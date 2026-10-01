"""테스트용 가짜 인지 노드 + 간이 차량 시뮬레이터.

- 시나리오를 파라미터로 골라 인지 토픽을 대신 발행한다.
- /cmd_vel 을 적분해 /odom 을 발행한다 (주차 시나리오에서 로봇 이동 반영).
실차에서는 절대 함께 실행하지 말 것 (/odom 이 중복 발행됨).
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Float32MultiArray, Bool, Header
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from ar_msgs.msg import Obstacle, ObstacleArray, WallSegment, WallArray, LaneError


# 주차 포켓 (world 좌표: 로봇 시작점 원점, 진행방향 +x, 포켓은 왼쪽)
PARKING_WORLD = [
    (0.8, 0.75, 1.2, 0.75, WallSegment.ROLE_BACK),
    (0.8, 0.25, 0.8, 0.75, WallSegment.ROLE_SIDE),
    (1.2, 0.25, 1.2, 0.75, WallSegment.ROLE_SIDE),
]


class FakePerception(Node):

    def __init__(self):
        super().__init__('fake_perception')

        self.declare_parameter('scenario', 'lane')
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('lateral_error', 0.0)
        self.declare_parameter('heading_error', 0.0)
        self.declare_parameter('distance', 1.0)
        self.declare_parameter('gate_open', False)

        pub = self.create_publisher
        self.pub_lane = pub(LaneError, '/perception/lane_error', 10)
        self.pub_event = pub(String, '/perception/event', 10)
        self.pub_stopline = pub(Float32, '/perception/stopline_distance', 10)
        self.pub_gate_dist = pub(Float32, '/perception/gate_distance', 10)
        self.pub_gate_open = pub(Bool, '/perception/gate_open', 10)
        self.pub_obstacles = pub(ObstacleArray, '/perception/obstacles', 10)
        self.pub_marker_type = pub(String, '/perception/marker_type', 10)
        self.pub_marker_dist = pub(Float32, '/perception/marker_distance', 10)
        self.pub_walls = pub(WallArray, '/perception/parking_walls', 10)
        self.pub_odom = pub(Odometry, '/odom', 10)

        # --- 간이 시뮬레이터 ---
        self.sim_x = 0.0
        self.sim_y = 0.0
        self.sim_yaw = 0.0
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.sim_dt = 0.02
        self.last_scenario = None

        self.declare_parameter('publish_odom', True)
        if bool(self._p('publish_odom')):
            self.create_subscription(Twist, '/cmd_vel', self._cb_cmd, 10)
            self.create_timer(self.sim_dt, self._sim_step)

        rate = float(self._p('rate'))
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info('가짜 인지 시작 - scenario=%s' % self._p('scenario'))

    # ------------------------------------------------------------------
    # 시뮬레이터
    # ------------------------------------------------------------------
    def _cb_cmd(self, msg):
        self.cmd_v = msg.linear.x
        self.cmd_w = msg.angular.z

    def _sim_step(self):
        dt = self.sim_dt
        self.sim_x += self.cmd_v * math.cos(self.sim_yaw) * dt
        self.sim_y += self.cmd_v * math.sin(self.sim_yaw) * dt
        self.sim_yaw += self.cmd_w * dt

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.sim_x
        odom.pose.pose.position.y = self.sim_y
        odom.pose.pose.orientation.z = math.sin(self.sim_yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.sim_yaw / 2.0)
        odom.twist.twist.linear.x = self.cmd_v
        odom.twist.twist.angular.z = self.cmd_w
        self.pub_odom.publish(odom)

    def _reset_pose(self):
        self.sim_x = self.sim_y = self.sim_yaw = 0.0

    def _to_base(self, wx, wy):
        """world 좌표 -> 현재 로봇 기준(base_link) 좌표."""
        dx, dy = wx - self.sim_x, wy - self.sim_y
        c, s = math.cos(-self.sim_yaw), math.sin(-self.sim_yaw)
        return (c * dx - s * dy, s * dx + c * dy)

    # ------------------------------------------------------------------
    def _p(self, name):
        return self.get_parameter(name).value

    def _tick(self):
        scenario = self._p('scenario')
        if scenario != self.last_scenario:
            self._reset_pose()
            self.last_scenario = scenario
            self.get_logger().info('시나리오 %s - 시뮬레이터 위치 초기화' % scenario)

        lane = LaneError()
        lane.header = self._header()
        lane.valid = True
        lane.lateral_offset_m = float(self._p('lateral_error'))
        lane.heading_error_rad = float(self._p('heading_error'))
        lane.confidence = 1.0
        self.pub_lane.publish(lane)

        handler = getattr(self, '_s_' + scenario, None)
        if handler is None:
            self._event('none')
        else:
            handler()

    def _event(self, name):
        msg = String()
        msg.data = name
        self.pub_event.publish(msg)

    def _f32(self, pub, value):
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)

    def _header(self):
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = 'base_link'
        return h

    # ------------------------------------------------------------------
    # 시나리오
    # ------------------------------------------------------------------
    def _s_lane(self):
        self._event('none')

    def _s_red(self):
        self._event('red_zone')

    def _s_blue(self):
        self._event('blue_zone')

    def _s_crosswalk(self):
        self._event('crosswalk')
        self._f32(self.pub_stopline, self._p('distance'))

    def _s_gate(self):
        self._event('gate')
        self._f32(self.pub_gate_dist, self._p('distance'))
        msg = Bool()
        msg.data = bool(self._p('gate_open'))
        self.pub_gate_open.publish(msg)

    def _s_cone(self):
        """좌우 라바콘 3쌍을 앞쪽에 배치."""
        self._event('cone')
        arr = ObstacleArray()
        arr.header = self._header()
        for i in range(3):
            x = 0.6 + 0.45 * i
            for side in (1.0, -1.0):
                o = Obstacle()
                o.x = x
                o.y = side * 0.32
                o.radius = 0.09
                o.point_count = 12
                o.is_dynamic = False
                arr.obstacles.append(o)
        self.pub_obstacles.publish(arr)

    def _s_marker(self):
        self._event('marker')
        msg = String()
        msg.data = 'left'
        self.pub_marker_type.publish(msg)
        self._f32(self.pub_marker_dist, self._p('distance'))

    def _s_roundabout(self):
        """회전 차량 2대 + 중앙 아일랜드."""
        self._event('roundabout')
        t = self.get_clock().now().nanoseconds * 1e-9
        arr = ObstacleArray()
        arr.header = self._header()

        cx, cy, r, w = 0.5, 0.0, 0.5, 0.8
        for phase in (0.0, math.pi):
            a = w * t + phase
            o = Obstacle()
            o.x = cx + r * math.cos(a)
            o.y = cy + r * math.sin(a)
            o.radius = 0.10
            o.point_count = 15
            o.vx = -r * w * math.sin(a)
            o.vy = r * w * math.cos(a)
            o.is_dynamic = True
            arr.obstacles.append(o)

        island = Obstacle()
        island.x = cx
        island.y = cy
        island.radius = 0.20
        island.point_count = 25
        island.is_dynamic = False
        arr.obstacles.append(island)
        self.pub_obstacles.publish(arr)

    def _s_parking(self):
        """world 에 고정된 ㄷ자 포켓을 로봇 위치 기준으로 변환해 발행."""
        self._event('parking_sign')
        arr = WallArray()
        arr.header = self._header()
        for wx1, wy1, wx2, wy2, role in PARKING_WORLD:
            x1, y1 = self._to_base(wx1, wy1)
            x2, y2 = self._to_base(wx2, wy2)
            seg = WallSegment()
            seg.x1, seg.y1, seg.x2, seg.y2 = x1, y1, x2, y2
            seg.theta = math.atan2(y2 - y1, x2 - x1)
            seg.length = math.hypot(x2 - x1, y2 - y1)
            seg.role = role
            arr.walls.append(seg)
        self.pub_walls.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = FakePerception()
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