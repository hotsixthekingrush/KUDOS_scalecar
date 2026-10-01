#!/usr/bin/env python3
"""
viewpoint_sampler.py
로봇을 미션 물체 앞 여러 위치·각도로 자동 순간이동시키며 auto_labeler 로 사진을 모으는 노드.
손으로 운전하면 표지판 앞을 지나는 몇 초만 저장되지만, 이 노드는 20분이면
클래스마다 수백 장을 모은다.

[한 자세마다 하는 일]
  1) auto_labeler 저장 중지  (/auto_labeler/enable = False)
  2) 좌/우 마커 텍스처 무작위 맞바꾸기, 주차 표지판 조금 흔들기
  3) 로봇을 config/viewpoints.yaml 의 구역 중 하나에서 뽑은 자세로 순간이동
  4) 순간이동 이후(시뮬 시간 기준 settle_sim_sec 뒤)에 찍힌 카메라 사진이 실제로 도착할 때까지 기다림
  5) auto_labeler 저장 재개 → 새 사진 capture_frames 장이 도착하면 다시 중지
  → 시계가 아니라 "사진의 촬영 시각"으로 판단하므로, Gazebo 가 느려도(카메라 1Hz 등)
    옮기기 전 사진에 새 라벨이 붙는 일이 없다. (auto_labeler 도 재개 시각 이전 사진은 버림)

실행 (Gazebo 를 먼저 띄울 것, teleop 은 끄기):
  ros2 launch kudos_yolo_tools collect_data.launch.py mode:=auto output_dir:=$HOME/yolo_data/dataset
"""

import math
import os
import random
import time

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState

from domain_randomizer import SWAP_PAIR, JITTER_GROUPS, yaw_of


def default_viewpoints_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('kudos_yolo_tools'),
                            'config', 'viewpoints.yaml')
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'config', 'viewpoints.yaml')


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def sample_pose(r, rng):
    """구역 설정 하나에서 (x, y, yaw) 하나를 뽑는다."""
    t = r['type']
    if t == 'box':
        x = rng.uniform(*r['x'])
        y = rng.uniform(*r['y'])
        yaw = r.get('yaw', 0.0) + rng.uniform(-1, 1) * r.get('yaw_jitter', 0.0)
        return x, y, wrap(yaw)
    if t == 'facing':
        tx, ty = r['target']
        n = r['normal_yaw']
        d = rng.uniform(*r['dist'])
        l = rng.uniform(*r['lateral'])
        nx, ny = math.cos(n), math.sin(n)
        lx, ly = -ny, nx          # 앞면 방향을 왼쪽으로 90도 돌린 방향
        x = tx + d * nx + l * lx
        y = ty + d * ny + l * ly
        yaw = math.atan2(ty - y, tx - x) + rng.uniform(-1, 1) * r.get('yaw_jitter', 0.0)
        return x, y, wrap(yaw)
    if t == 'ring':
        cx, cy = r['center']
        rad = rng.uniform(*r['radius'])
        ang = rng.uniform(-math.pi, math.pi)
        x = cx + rad * math.cos(ang)
        y = cy + rad * math.sin(ang)
        yaw = math.atan2(cy - y, cx - x) + rng.uniform(-1, 1) * r.get('yaw_jitter', 0.0)
        return x, y, wrap(yaw)
    raise ValueError('알 수 없는 type: %s' % t)


class ViewpointSampler(Node):
    def __init__(self):
        super().__init__('viewpoint_sampler')
        p = self.declare_parameter
        p('robot_model_name', 'mbot')
        p('viewpoints_file', '')
        p('num_poses', 600)          # 총 자세 수 (자세당 2~3장 저장)
        p('image_topic', '/rgb/image_raw')
        p('settle_sim_sec', 0.3)     # 순간이동 후 이만큼(시뮬 시간) 지난 뒤 찍힌 사진부터 인정
        p('capture_frames', 1)       # 재개 후 새 사진 몇 장 받고 멈출지 (labeler save_every_n=2 → 2장 저장)
        p('frame_timeout_sec', 20.0) # 이 시간(실제 시간) 동안 새 사진이 없으면 경고 후 다음 자세로
        p('swap_markers', True)
        p('jitter_objects', True)
        p('seed', -1)                # -1 이면 매번 다르게
        g = lambda n: self.get_parameter(n).value

        self.robot = g('robot_model_name')
        vfile = g('viewpoints_file') or default_viewpoints_file()
        with open(vfile) as f:
            cfg = yaml.safe_load(f)
        self.regions = cfg['regions']
        self.avoid = [((float(a['xy'][0]), float(a['xy'][1])), float(a['r']))
                      for a in (cfg.get('avoid') or [])]
        self.num_poses = int(g('num_poses'))
        self.settle_ns = int(float(g('settle_sim_sec')) * 1e9)
        self.capture_frames = int(g('capture_frames'))
        self.frame_timeout = float(g('frame_timeout_sec'))
        self.do_swap = bool(g('swap_markers'))
        self.do_jitter = bool(g('jitter_objects'))
        seed = int(g('seed'))
        self.rng = random.Random(None if seed < 0 else seed)

        self.weights = [float(r.get('weight', 1.0)) for r in self.regions]
        self.count = {r['name']: 0 for r in self.regions}
        self.base = None           # 마커·표지판 원래 위치
        self.robot_z = None        # 로봇이 바닥에 서 있을 때의 z
        self.done = 0
        self.phase = 'wait_states'
        self.deadline = 0.0          # 실제 시간 타임아웃
        self.t_move_ns = 0           # 순간이동 시각 (시뮬)
        self.t_enable_ns = 0         # 저장 재개 시각 (시뮬)
        self.last_img_ns = 0         # 가장 최근 도착한 사진의 촬영 시각
        self.cap_count = 0
        self.timeouts = 0

        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.pub_enable = self.create_publisher(Bool, '/auto_labeler/enable', 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.states_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, g('image_topic'), self.image_cb, qos_profile_sensor_data)
        self.create_timer(0.05, self.tick)
        self.get_logger().info('viewpoint_sampler 시작 | 구역 %d개 | 총 %d자세 | %s'
                               % (len(self.regions), self.num_poses, vfile))

    # ------------------------------------------------------------------
    def states_cb(self, msg):
        if self.base is not None:
            return
        poses = dict(zip(msg.name, msg.pose))
        if self.robot not in poses:
            self.get_logger().error('model_states 에 "%s" 없음. 있는 이름: %s'
                                    % (self.robot, ', '.join(sorted(poses))),
                                    throttle_duration_sec=5.0)
            return
        self.robot_z = poses[self.robot].position.z
        need = set(SWAP_PAIR)
        for names, _, _ in JITTER_GROUPS:
            need |= set(names)
        self.base = {n: poses[n] for n in need if n in poses}
        self.get_logger().info('로봇 기준 높이 z=%.3f, 물체 %d개 위치 저장'
                               % (self.robot_z, len(self.base)))

    def now_ns(self):
        return self.get_clock().now().nanoseconds   # use_sim_time 이면 시뮬 시간

    def image_cb(self, msg):
        st = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        self.last_img_ns = max(self.last_img_ns, st)
        if self.phase == 'capture' and st >= self.t_enable_ns:
            self.cap_count += 1

    def blocked(self, x, y):
        return any(math.hypot(x - ax, y - ay) < ar for (ax, ay), ar in self.avoid)

    def set_labeler(self, on):
        m = Bool()
        m.data = bool(on)
        self.pub_enable.publish(m)

    def set_pose(self, name, x, y, z, yaw, qx=0.0, qy=0.0, qz=None, qw=None):
        st = EntityState()
        st.name = name
        st.reference_frame = 'world'
        st.pose.position.x, st.pose.position.y, st.pose.position.z = float(x), float(y), float(z)
        if qz is None:
            qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        st.pose.orientation.x, st.pose.orientation.y = float(qx), float(qy)
        st.pose.orientation.z, st.pose.orientation.w = float(qz), float(qw)
        req = SetEntityState.Request()
        req.state = st
        self.cli.call_async(req)

    def put(self, name, pose):
        p, q = pose.position, pose.orientation
        self.set_pose(name, p.x, p.y, p.z, 0.0, q.x, q.y, q.z, q.w)

    def randomize_objects(self):
        a, b = SWAP_PAIR
        if self.do_swap and a in self.base and b in self.base:
            if self.rng.random() < 0.5:
                self.put(a, self.base[b])
                self.put(b, self.base[a])
            else:
                self.put(a, self.base[a])
                self.put(b, self.base[b])
        if self.do_jitter:
            for names, dxy, dyaw in JITTER_GROUPS:
                names = [n for n in names if n in self.base]
                if not names:
                    continue
                ref = self.base[names[0]].position
                tx, ty = self.rng.uniform(-dxy, dxy), self.rng.uniform(-dxy, dxy)
                dth = self.rng.uniform(-dyaw, dyaw)
                c, s = math.cos(dth), math.sin(dth)
                for n in names:
                    pp = self.base[n].position
                    rx, ry = pp.x - ref.x, pp.y - ref.y
                    self.set_pose(n, ref.x + c * rx - s * ry + tx, ref.y + s * rx + c * ry + ty,
                                  pp.z, yaw_of(self.base[n].orientation) + dth)

    # ------------------------------------------------------------------
    def tick(self):
        now = time.monotonic()
        if self.phase == 'wait_states':
            if self.base is None or not self.cli.service_is_ready():
                return
            self.phase = 'move'

        if self.phase == 'move':
            if self.done >= self.num_poses:
                self.get_logger().info('완료: %d자세 | 구역별 %s | 영상 지연으로 건너뜀 %d'
                                       % (self.done, self.count, self.timeouts))
                self.phase = 'finished'
                self.set_labeler(True)   # 사람이 이어서 운전하며 모을 수 있도록 다시 켬
                return
            self.set_labeler(False)
            self.randomize_objects()
            r = self.rng.choices(self.regions, weights=self.weights, k=1)[0]
            for _ in range(30):
                x, y, yaw = sample_pose(r, self.rng)
                if not self.blocked(x, y):
                    break
            self.set_pose(self.robot, x, y, self.robot_z + 0.01, yaw)
            self.count[r['name']] += 1
            self.t_move_ns = self.now_ns()
            self.phase, self.deadline = 'settle', now + self.frame_timeout

        elif self.phase == 'settle':
            # 순간이동 이후에 찍힌 사진이 실제로 도착했는가
            if self.last_img_ns > self.t_move_ns + self.settle_ns:
                self.t_enable_ns = self.now_ns()
                self.cap_count = 0
                self.set_labeler(True)
                self.phase, self.deadline = 'capture', now + self.frame_timeout
            elif now > self.deadline:
                self.skip('순간이동 후 새 카메라 영상이 %.0f초 동안 안 옴' % self.frame_timeout)

        elif self.phase == 'capture':
            if self.cap_count >= self.capture_frames:
                self.set_labeler(False)
                self.done += 1
                if self.done % 50 == 0:
                    self.get_logger().info('%d / %d 자세 | %s' % (self.done, self.num_poses, self.count))
                self.phase = 'move'
            elif now > self.deadline:
                self.skip('저장 중 새 카메라 영상이 %.0f초 동안 안 옴' % self.frame_timeout)

    def skip(self, why):
        self.set_labeler(False)
        self.timeouts += 1
        self.done += 1
        self.get_logger().warn('%s → 이 자세 건너뜀. Gazebo 가 켜져 있는지, '
                               '`ros2 topic hz /rgb/image_raw` 확인' % why)
        self.phase = 'move'


def main(args=None):
    rclpy.init(args=args)
    node = ViewpointSampler()
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
