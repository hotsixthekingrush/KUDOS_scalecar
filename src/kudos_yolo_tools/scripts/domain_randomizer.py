#!/usr/bin/env python3
"""
domain_randomizer.py
auto_labeler 로 수집하는 동안 Gazebo 장면을 주기적으로 바꿔서
한 번의 주행으로 다양한 학습 데이터를 얻게 해주는 노드.

★ webot_arena/worlds/limo_competition.world 기준 (9/28)

[무엇을 바꾸나]
1) 좌/우 마커 텍스처 위치 맞바꾸기
   월드에는 시작 표지판=우회전, 끝 표지판=좌회전으로 고정되어 있다.
   그대로 학습하면 모델이 "화살표"가 아니라 "배경/위치"로 좌우를 외울 수 있다.
   → 50% 확률로 두 텍스처를 서로의 자리로 옮긴다. (auto_labeler 는 모델 이름으로
     클래스를 정하므로 위치가 바뀌어도 라벨은 항상 정확하다)
2) 주차 표지판(기둥+판) 위치·각도 조금씩 흔들기 — 심사위원이 시설물을 옮길 수 있음

[바꾸지 않는 것]
- 회전교차로 차량, 차단기: webot_arena 의 mission_motion_controller.py 가 움직인다.
  수집할 때 그 노드를 같이 켜두면 된다. (여기서 또 움직이면 서로 싸운다)
- 조명: image_augment.py / 학습 augmentation(hsv_v) 으로 대신한다.

좌표를 코드에 박지 않고, 시작할 때 /gazebo/model_states 에서 원래 위치를 읽어온다.
→ 월드 파일에서 표지판 위치가 바뀌어도 이 코드는 고칠 필요 없음.

실행 (auto_labeler, mission_motion_controller 와 함께):
  ros2 run kudos_yolo_tools domain_randomizer.py --ros-args -p interval_sec:=2.0
"""

import math
import random
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState, ModelStates
from std_msgs.msg import Bool


# 서로 자리를 맞바꿀 두 모델 (targets.yaml 의 마커 이름과 같아야 함)
SWAP_PAIR = ('m4_sign_start_texture', 'm4_sign_end_texture')

# 함께 움직일 묶음: (모델 이름들, 위치 흔들 범위 m, yaw 흔들 범위 rad)
JITTER_GROUPS = [
    (('parking_sign_post', 'parking_sign_panel'), 0.08, 0.25),
]


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class DomainRandomizer(Node):
    def __init__(self):
        super().__init__('domain_randomizer')
        self.declare_parameter('interval_sec', 2.0)
        self.declare_parameter('swap_markers', True)
        self.declare_parameter('jitter_objects', True)
        self.interval = float(self.get_parameter('interval_sec').value)
        self.do_swap = bool(self.get_parameter('swap_markers').value)
        self.do_jitter = bool(self.get_parameter('jitter_objects').value)

        self.base = None          # 이름 -> 원래 Pose
        self.swapped = False
        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_subscription(ModelStates, '/gazebo/model_states', self.states_cb,
                                 qos_profile_sensor_data)
        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/gazebo/set_entity_state 없음 — 월드에 gazebo_ros_state 플러그인 확인')
        # 바꾸는 동안 auto_labeler 저장을 멈춘다 (이미지와 라벨이 어긋나는 것 방지)
        self.pub_enable = self.create_publisher(Bool, '/auto_labeler/enable', 10)
        self.reenable_at = None
        self.create_timer(self.interval, self.randomize)
        self.create_timer(0.05, self.check_reenable)
        self.get_logger().info('domain_randomizer 시작 (주기 %.1fs)' % self.interval)

    def states_cb(self, msg):
        if self.base is not None:
            return
        poses = dict(zip(msg.name, msg.pose))
        need = set(SWAP_PAIR)
        for names, _, _ in JITTER_GROUPS:
            need |= set(names)
        missing = [n for n in need if n not in poses]
        if missing:
            self.get_logger().warn('월드에 없는 모델(무시됨): %s' % ', '.join(sorted(missing)))
        self.base = {n: poses[n] for n in need if n in poses}
        self.get_logger().info('원래 위치 %d개 저장' % len(self.base))

    def set_pose(self, name, x, y, z, qx, qy, qz, qw):
        st = EntityState()
        st.name = name
        st.reference_frame = 'world'
        st.pose.position.x, st.pose.position.y, st.pose.position.z = float(x), float(y), float(z)
        st.pose.orientation.x, st.pose.orientation.y = float(qx), float(qy)
        st.pose.orientation.z, st.pose.orientation.w = float(qz), float(qw)
        req = SetEntityState.Request()
        req.state = st
        self.cli.call_async(req)

    def put(self, name, pose):
        p, q = pose.position, pose.orientation
        self.set_pose(name, p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    def set_labeler(self, on):
        m = Bool()
        m.data = bool(on)
        self.pub_enable.publish(m)

    def check_reenable(self):
        if self.reenable_at is not None and time.monotonic() >= self.reenable_at:
            self.reenable_at = None
            self.set_labeler(True)

    def randomize(self):
        if self.base is None:
            return
        self.set_labeler(False)
        self.reenable_at = time.monotonic() + 0.8

        # 1) 마커 맞바꾸기
        a, b = SWAP_PAIR
        if self.do_swap and a in self.base and b in self.base:
            self.swapped = random.random() < 0.5
            if self.swapped:
                self.put(a, self.base[b])
                self.put(b, self.base[a])
            else:
                self.put(a, self.base[a])
                self.put(b, self.base[b])

        # 2) 묶음 흔들기 — 묶음 전체를 첫 모델 위치 기준으로 같은 만큼 회전·이동
        if self.do_jitter:
            for names, dxy, dyaw in JITTER_GROUPS:
                names = [n for n in names if n in self.base]
                if not names:
                    continue
                ref = self.base[names[0]].position
                tx, ty = random.uniform(-dxy, dxy), random.uniform(-dxy, dxy)
                dth = random.uniform(-dyaw, dyaw)
                c, s = math.cos(dth), math.sin(dth)
                for n in names:
                    p = self.base[n].position
                    q = self.base[n].orientation
                    rx, ry = p.x - ref.x, p.y - ref.y
                    nx = ref.x + c * rx - s * ry + tx
                    ny = ref.y + s * rx + c * ry + ty
                    yaw = yaw_of(q) + dth
                    # 이 묶음의 모델들은 roll/pitch 가 0 이므로 yaw 만으로 재구성
                    self.set_pose(n, nx, ny, p.z, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def main(args=None):
    rclpy.init(args=args)
    node = DomainRandomizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
