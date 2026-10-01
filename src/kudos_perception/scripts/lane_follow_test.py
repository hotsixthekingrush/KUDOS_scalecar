#!/usr/bin/env python3
"""
lane_follow_test.py — 차선 인식 확인용 "임시" 주행 노드 (Gazebo 전용)

/perception/lane_error 를 받아 /cmd_vel 을 내보내서 로봇이 차선을 따라 끝까지 가보게 한다.
차선 인식이 커브에서 제대로 나오는지 눈으로 확인하는 용도이고,
대회용 제어기(김서윤 담당)를 대신하는 코드가 아니다. main 에 넣지 않아도 된다.

[방식] Pure Pursuit
  LaneError 는 로봇(base_link) 앞 eval_dist_m(0.55m) 지점에서
    lateral_offset_m  : 차선 중앙이 로봇보다 왼쪽이면 +
    heading_error_rad : 차선이 왼쪽으로 꺾이면 +
  를 준다. 그 지점에서 차선 방향으로 lookahead_m 만큼 더 간 점을 목표로 잡고,
  뒷바퀴 축 기준으로 그 점을 지나는 원의 곡률 kappa = 2*y / (x^2 + y^2) 로 조향한다.
  ★ Gazebo 의 gazebo_ros_ackermann_drive 플러그인은 cmd_vel.angular.z 를
    "회전 속도(rad/s)"가 아니라 "앞바퀴 조향각(rad)"으로 해석한다 (9/30 확인, 플러그인 소스).
    → angular_is_steer:=true (기본): angular.z = atan(wheelbase * kappa)
      angular_is_steer:=false        : angular.z = v * kappa (일반 Twist 규약, 실차 확인 필요)

[안전]
  - 차선 인식이 timeout_s 동안 안 들어오거나 valid=false 가 계속되면 정지
  - Ctrl+C 로 끄면 정지 명령을 보내고 종료
  - 카메라가 느리므로(약 1.6Hz) 속도는 0.1 m/s 정도로 천천히

실행:
  ros2 run kudos_perception lane_follow_test.py --ros-args -p use_sim_time:=true
  (속도 바꾸기: -p speed:=0.15)
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ar_msgs.msg import LaneError


class LaneFollowTest(Node):
    def __init__(self):
        super().__init__('lane_follow_test')
        p = self.declare_parameter
        p('speed', 0.10)              # [m/s] 직진 속도
        p('eval_dist_m', 0.55)        # LaneError 가 측정된 지점 (base_link 앞). sim yaml 과 맞춤
        p('lookahead_m', 0.0)         # 그 지점에서 차선 방향으로 더 볼 거리 (0.25 는 U턴 출구에서 바깥으로 밀림, 10/1 시뮬)
        p('rear_axle_offset_m', 0.12) # base_link -> 뒷바퀴 축 거리 (wheelbase 0.24 / 2)
        p('wheelbase_m', 0.24)
        p('max_steer_rad', 0.5236)    # 30도 (ackermann.xacro max_steer)
        p('timeout_s', 2.0)           # 이 시간 동안 인식이 없으면 정지
        p('slow_on_curve', True)      # 커브에서 속도를 조금 줄임
        p('angular_is_steer', True)   # Gazebo ackermann 플러그인: angular.z = 조향각

        g = lambda n: self.get_parameter(n).value
        self.v = float(g('speed'))
        self.eval_d = float(g('eval_dist_m'))
        self.look = float(g('lookahead_m'))
        self.rear = float(g('rear_axle_offset_m'))
        self.kappa_max = math.tan(float(g('max_steer_rad'))) / float(g('wheelbase_m'))
        self.timeout = float(g('timeout_s'))
        self.slow_on_curve = bool(g('slow_on_curve'))
        self.angular_is_steer = bool(g('angular_is_steer'))
        self.wheelbase = float(g('wheelbase_m'))

        self.last = None          # 마지막 valid LaneError
        self.last_t = None        # 받은 시각 (노드 시계)
        self.n_invalid = 0

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaneError, '/perception/lane_error', self.cb, 10)
        self.create_timer(0.05, self.tick)   # 20Hz 로 마지막 명령 유지
        self.create_timer(1.0, self.log)
        self.get_logger().info(
            f'lane_follow_test 시작: v={self.v:.2f} m/s, lookahead={self.look:.2f} m, '
            f'angular.z={"조향각" if self.angular_is_steer else "회전속도"}, '
            f'최대 곡률 {self.kappa_max:.2f} 1/m (반경 {1/self.kappa_max:.2f} m)')

    def cb(self, msg):
        if not msg.valid:
            self.n_invalid += 1
            return
        self.n_invalid = 0
        self.last = msg
        self.last_t = self.get_clock().now()

    def compute(self):
        m = self.last
        # 목표점: 측정 지점 (eval_d, off) 에서 차선 방향(hdg)으로 look 만큼 더 간 점. 뒷바퀴 축 기준
        x = self.rear + self.eval_d + self.look * math.cos(m.heading_error_rad)
        y = m.lateral_offset_m + self.look * math.sin(m.heading_error_rad)
        kappa = 2.0 * y / (x * x + y * y)
        kappa = max(-self.kappa_max, min(self.kappa_max, kappa))
        v = self.v
        if self.slow_on_curve:
            v *= max(0.6, 1.0 - 0.25 * abs(kappa))   # 곡률 1.6(반경 0.6m)에서 60%
        if self.angular_is_steer:
            return v, math.atan(self.wheelbase * kappa), kappa
        return v, v * kappa, kappa

    def tick(self):
        cmd = Twist()
        if self.last_t is not None:
            age = (self.get_clock().now() - self.last_t).nanoseconds * 1e-9
            if age <= self.timeout:
                cmd.linear.x, cmd.angular.z, _ = self.compute()
        self.pub.publish(cmd)

    def log(self):
        if self.last is None:
            self.get_logger().warn('차선 인식 대기 중… (/perception/lane_error 가 안 오거나 valid=false)')
            return
        age = (self.get_clock().now() - self.last_t).nanoseconds * 1e-9
        if age > self.timeout:
            self.get_logger().warn(f'차선 놓침 {age:.1f}s — 정지 (invalid 연속 {self.n_invalid})')
            return
        v, w, k = self.compute()
        m = self.last
        self.get_logger().info(
            f'off={m.lateral_offset_m:+.3f}m hdg={math.degrees(m.heading_error_rad):+5.1f}° '
            f'-> v={v:.2f} {"steer" if self.angular_is_steer else "w"}={w:+.2f} (반경 {abs(1/k) if abs(k) > 1e-3 else float("inf"):.2f}m)'
            + (' [한쪽 선]' if m.single_lane_estimated else ''))

    def stop(self):
        self.pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            for _ in range(3):
                node.stop()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
