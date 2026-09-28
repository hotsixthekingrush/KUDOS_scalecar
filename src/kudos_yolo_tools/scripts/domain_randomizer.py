#!/usr/bin/env python3
"""
domain_randomizer.py
Gazebo 환경을 주기적으로 무작위로 바꿔서, 한 번의 수집으로
다양한 조건의 학습 데이터를 얻게 해주는 노드.

[왜 필요한가]
시뮬레이션 이미지만으로 학습하면, 실제 카메라의 노이즈·조명 특성과 달라
실전에서 성능이 떨어진다(Sim-to-Real Gap). 조명·물체 배치 등
"인식과 무관해야 할 요소"를 일부러 다양하게 바꿔주면, 모델이 그런
변화에 둔감해져서 실제 환경에서도 잘 작동하게 된다.

[무엇을 무작위화하는가]
1. 조명 밝기·색온도·방향   <- 대회장 조명이 연습 환경과 다를 위험 대비
2. 표지판/차단기 위치·각도  <- 심사위원이 시설물을 옮길 수 있음(규정 명시)
3. 동적 차량 위치          <- 회전교차로 미션 다양한 상황

auto_labeler.py 와 동시에 실행하면, 무작위로 바뀐 장면들이
자동 라벨과 함께 계속 저장된다.

실행:
  ros2 run <pkg> domain_randomizer.py --ros-args -p interval_sec:=2.0
"""

import random
import math

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose


# 무작위로 위치를 바꿀 대상: (모델명, 기준위치x, 기준위치y, 흔들 범위 m)
MOVABLE = [
    ('sign_lane_change', 1.25,  1.00, 0.25),
    ('sign_parking',     2.18, -1.59, 0.20),
    ('dynamic_car_1',   -1.64,  0.80, 0.0),   # 회전교차로 — 원 궤도 위에서 회전
    ('dynamic_car_2',   -1.64, -0.64, 0.0),
]

# 회전교차로 중심 (동적 차량이 이 주위를 돈다)
ROUNDABOUT_CENTER = (-1.64, 0.08)
ROUNDABOUT_RADIUS = 0.72


class DomainRandomizer(Node):
    def __init__(self):
        super().__init__('domain_randomizer')

        self.declare_parameter('interval_sec', 2.0)
        self.declare_parameter('randomize_lighting', True)
        self.declare_parameter('randomize_objects', True)

        self.interval = self.get_parameter('interval_sec').value
        self.do_light = self.get_parameter('randomize_lighting').value
        self.do_obj = self.get_parameter('randomize_objects').value

        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.t = 0.0

        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                '/gazebo/set_entity_state 서비스를 찾을 수 없습니다. '
                '월드 파일에 gazebo_ros_state 플러그인이 있는지 확인하세요.')

        self.timer = self.create_timer(self.interval, self.randomize)
        self.get_logger().info(f'domain_randomizer 시작 (주기 {self.interval}s)')

    def set_pose(self, name, x, y, z, yaw):
        req = SetEntityState.Request()
        st = EntityState()
        st.name = name
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        p.orientation.z = math.sin(yaw / 2.0)
        p.orientation.w = math.cos(yaw / 2.0)
        st.pose = p
        st.reference_frame = 'world'
        req.state = st
        self.cli.call_async(req)

    def randomize(self):
        self.t += self.interval

        if self.do_obj:
            for name, bx, by, jitter in MOVABLE:
                if name.startswith('dynamic_car'):
                    # 회전교차로 궤도 위에서 일정 속도로 회전
                    phase = 0.0 if name.endswith('1') else math.pi
                    ang = self.t * 0.6 + phase
                    x = ROUNDABOUT_CENTER[0] + ROUNDABOUT_RADIUS * math.cos(ang)
                    y = ROUNDABOUT_CENTER[1] + ROUNDABOUT_RADIUS * math.sin(ang)
                    self.set_pose(name, x, y, 0.05, ang + math.pi / 2)
                else:
                    x = bx + random.uniform(-jitter, jitter)
                    y = by + random.uniform(-jitter, jitter)
                    yaw = random.uniform(-0.4, 0.4)
                    self.set_pose(name, x, y, 0.0, yaw)

        if self.do_light:
            # Gazebo Classic에서 조명 변경은 서비스로 직접 지원되지 않으므로,
            # 아래 안내대로 월드 파일의 <light> 값을 여러 벌 준비해 바꿔가며
            # 수집하거나, gazebo_ros_light 플러그인을 별도로 쓰는 방식을 권장.
            pass


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
