"""판단 + 상위 제어 통합 노드 (SMACH 기반 Rule-based FSM)."""

import math
import threading

import rclpy
import smach
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import String, Float32, Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from ar_msgs.msg import (ObstacleArray, WallArray, LaneError, ColorZone,
                         MissionObject, MissionObjectArray)

from ar_mission_supervisor.states.lane_following import LaneFollowing
from ar_mission_supervisor.states.stub import StubState
from ar_mission_supervisor.states.acc_dcc import AccDcc
from ar_mission_supervisor.states.crosswalk_stop import CrosswalkStop
from ar_mission_supervisor.states.gate_stop import GateStop
from ar_mission_supervisor.states.avoid_cone import AvoidCone
from ar_mission_supervisor.states.left_right import LeftRight
from ar_mission_supervisor.states.avoid_car import AvoidCar
from ar_mission_supervisor.states.parking import Parking


# ----------------------------------------------------------------------
# Transition 테이블
# ----------------------------------------------------------------------
EVENT_TO_OUTCOME = {
    'red_zone': 'to_acc_dcc',
    'blue_zone': 'to_acc_dcc',
    'crosswalk': 'to_crosswalk',
    'cone': 'to_cone',
    'marker': 'to_left_right',
    'roundabout': 'to_roundabout',
    'gate': 'to_gate',
    'parking_sign': 'to_parking',
}

OUTCOME_TO_STATE = {
    'to_acc_dcc': 'ACC_DCC',
    'to_crosswalk': 'CROSSWALK_STOP',
    'to_cone': 'AVOID_CONE',
    'to_left_right': 'LEFT_RIGHT',
    'to_roundabout': 'AVOID_CAR',
    'to_gate': 'GATE_STOP',
    'to_parking': 'PARKING',
}

MISSION_LABELS = list(OUTCOME_TO_STATE.values())

IMPLEMENTED = {
    'ACC_DCC': AccDcc,
    'CROSSWALK_STOP': CrosswalkStop,
    'AVOID_CONE': AvoidCone,
    'LEFT_RIGHT': LeftRight,
    'AVOID_CAR': AvoidCar,
    'GATE_STOP': GateStop,
    'PARKING': Parking,
}

# 이벤트 발행원. 'test' 는 기존 /perception/event (가짜 인지용)
EVENT_SOURCES = ('hsv', 'yolo', 'lidar', 'test')

# 여러 노드가 동시에 이벤트를 보낼 때의 우선순위
EVENT_PRIORITY = ('crosswalk', 'gate', 'roundabout', 'cone',
                  'marker', 'parking_sign', 'red_zone', 'blue_zone')


# ----------------------------------------------------------------------
# 파라미터 기본값 (YAML 에서 덮어씀). 실수형은 YAML 에서도 반드시 소수점 표기.
# ----------------------------------------------------------------------
PARAM_DEFAULTS = {
    '': {
        'control_rate': 20.0,
        'wheel_base': 0.2,
        'base_speed': 0.3,
        'max_steer': 0.4636,
    },
    'lane': {
        'kp_lateral': 1.2,
        'kd_lateral': 0.15,
        'kp_heading': 0.8,
    },
    'failsafe': {
        'hold_sec': 1.0,
        'creep_speed': 0.12,
        'stop_after_sec': 5.0,
        'wall_sector_deg': 30.0,
        'wall_max_range': 0.8,
    },
    'perception': {
        'stale_sec': 0.5,
        'debounce_count': 3,
        'lockout_sec': 8.0,
        'exit_miss_count': 3,
    },
    'zone': {
        'red_speed': 0.13,
        'blue_speed': 0.6,
    },
    'crosswalk': {
        'approach_speed': 0.2,
        'min_speed': 0.08,
        'slowdown_distance': 0.6,
        'stop_distance': 0.15,
        'wait_sec': 6.0,
        'approach_timeout': 8.0,
    },
    'gate': {
        'approach_speed': 0.2,
        'min_speed': 0.08,
        'slowdown_distance': 0.6,
        'stop_distance': 0.25,
        'open_confirm_sec': 0.3,
        'max_wait_sec': 30.0,
        'approach_timeout': 8.0,
    },
    'cone': {
        'speed': 0.25,
        'creep_speed': 0.10,
        'lookahead': 0.55,
        'curve_gain': 0.35,
        'roi_min_x': 0.05,
        'roi_max_x': 2.0,
        'roi_max_y': 1.0,
        'max_radius': 0.18,
        'pair_max_dx': 0.45,
        'min_gate_width': 0.30,
        'side_offset': 0.22,
        'robot_width': 0.22,
        'max_duration': 40.0,
    },
    'left_right': {
        'trigger_distance': 0.8,
        'approach_speed': 0.25,
        'shift_speed': 0.25,
        'lane_width': 0.40,
        'shift_duration': 2.5,
        'settle_sec': 1.0,
        'approach_timeout': 10.0,
        'max_duration': 20.0,
    },
    'roundabout': {
        'approach_speed': 0.22,
        'min_speed': 0.08,
        'cross_speed': 0.30,
        'slowdown_distance': 0.8,
        'entry_distance': 0.55,
        'island_min_radius': 0.15,
        'track_max_range': 2.0,
        'conflict_x': 0.5,
        'conflict_radius': 0.35,
        'cross_duration': 2.5,
        'safety_buffer': 0.8,
        'fallback_after': 9.0,
        'fallback_ratio': 0.4,
        'emergency_distance': 0.30,
        'emergency_hold': 1.5,
        'clear_margin': 1.6,
        'min_speed_for_prediction': 0.05,
        'approach_timeout': 8.0,
        'max_duration': 45.0,
    },
    'parking': {
        'side': 'left',
        'scan_speed': 0.15,
        'turn_offset': 0.40,
        'reverse_speed': 0.12,
        'align_speed': 0.08,
        'yaw_tolerance': 0.05,
        'clearance': 0.10,
        'robot_length': 0.322,
        'k_heading': 1.0,
        'k_lateral': 1.5,
        'phase_pause': 0.5,
        'scan_timeout': 25.0,
        'turn_timeout': 12.0,
        'align_timeout': 10.0,
    },
    'debug': {
        'enabled': True,
    },
    'yolo': {
        'event_range': 1.5,       # 횡단보도·차단기는 이 거리 안에 들어와야 이벤트 발생
        'gate_open_angle': 1.0,   # 차단기 바 기울기(rad)가 이 이상이면 열림 (약 57도)
    },
}

# 파라미터 그룹 -> 노드 속성 이름 (다른 경우만)
GROUP_ATTR = {'lane': 'lane_gains'}


# ----------------------------------------------------------------------
# 보조 클래스
# ----------------------------------------------------------------------
class Buffer(object):
    """최신 인지값 + 수신 시각. 오래되면 무효 처리.

    (값, 시각)을 한 덩어리로 저장해, 수신 스레드와 상태머신 스레드가
    동시에 접근해도 새 값과 옛 시각이 섞여 읽히지 않게 한다.
    """

    def __init__(self, stale_sec):
        self._data = (None, -1e9)
        self.stale_sec = stale_sec

    @property
    def value(self):
        return self._data[0]

    @value.setter
    def value(self, v):
        self._data = (v, self._data[1])

    def set(self, value, now):
        self._data = (value, now)

    def get(self, now):
        value, stamp = self._data
        if value is None or (now - stamp) > self.stale_sec:
            return None
        return value


class Debouncer(object):
    """같은 값이 N회 연속 확인될 때만 통과시킨다."""

    def __init__(self, count):
        self.count = count
        self.last = None
        self.n = 0

    def push(self, value):
        if value == self.last:
            self.n += 1
        else:
            self.last = value
            self.n = 1
        return value if self.n >= self.count else None


def _yaw_from_odom(msg):
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))

ZONE_EVENT = {ColorZone.ZONE_RED: 'red_zone', ColorZone.ZONE_BLUE: 'blue_zone'}


def _zone_to_event(msg):
    return ZONE_EVENT.get(msg.zone, 'none')


# ----------------------------------------------------------------------
# 노드
# ----------------------------------------------------------------------
class MissionSupervisor(Node):

    BUFFER_KEYS = ('lane', 'scan', 'stopline', 'gate_dist', 'gate_open',
                   'obstacles', 'marker_type', 'marker_dist', 'walls', 'odom',
                   'event_hsv', 'event_yolo', 'event_lidar', 'event_test',
                   'gate_dist_yolo', 'gate_open_yolo')

    def __init__(self):
        super().__init__('mission_supervisor')
        self._load_params()

        # --- 발행 ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.state_pub = self.create_publisher(String, '/supervisor/state', latched)
        self.current_label = 'INIT'
        self.last_v = 0.0          # 마지막으로 보낸 선속도 (지연 보정용)
        self._debug_pubs = {}
        self._debug_lock = threading.Lock()

        # --- 인지 버퍼 ---
        stale = self.perception['stale_sec']
        self.buf = dict((k, Buffer(stale)) for k in self.BUFFER_KEYS)

        # --- 구독 ---
        store = self._store
        sub = self.create_subscription
        sub(LaneError, '/perception/lane_error', self._cb_lane, 10)

        sub(LaserScan, '/scan', store('scan', lambda m: m), qos_profile_sensor_data)
        # 색상 구역은 카메라 인지가 ColorZone 으로 보냄 -> 이벤트 문자열로 변환
        sub(ColorZone, '/perception/color_zone', store('event_hsv', _zone_to_event), 10)
        for src in ('lidar', 'test'):
            topic = '/perception/event' if src == 'test' else '/perception/event/' + src
            sub(String, topic, store('event_' + src, lambda m: m.data), 10)
        sub(MissionObjectArray, '/perception/mission_objects', self._cb_mission_objects, 10)

        sub(Float32, '/perception/stopline_distance',
            store('stopline', lambda m: float(m.data)), 10)
        sub(Float32, '/perception/gate_distance',
            store('gate_dist', lambda m: float(m.data)), 10)
        sub(Bool, '/perception/gate_open',
            store('gate_open', lambda m: bool(m.data)), 10)
        sub(ObstacleArray, '/perception/obstacles',
            store('obstacles', lambda m: list(m.obstacles)), 10)
        sub(String, '/perception/marker_type',
            store('marker_type', lambda m: m.data), 10)
        sub(Float32, '/perception/marker_distance',
            store('marker_dist', lambda m: float(m.data)), 10)
        sub(WallArray, '/perception/parking_walls',
            store('walls', lambda m: list(m.walls)), 10)
        sub(Odometry, '/odom', store('odom', _yaw_from_odom), 10)
        sub(Bool, '/supervisor/reset', self._cb_reset, 10)

        # --- 전이 판정 ---
        self.debouncer = Debouncer(self.perception['debounce_count'])
        self.lockout = {}
        self._reset_flag = False
        self._lock = threading.Lock()

        self.create_timer(0.5, self._publish_state_tick)
        self.get_logger().info('mission_supervisor 준비 완료 (%.1f Hz)'
                               % self.control_rate)

    # ------------------------------------------------------------------
    # 파라미터
    # ------------------------------------------------------------------
    def _load_params(self):
        for group, items in PARAM_DEFAULTS.items():
            values = {}
            for key, default in items.items():
                name = '%s.%s' % (group, key) if group else key
                self.declare_parameter(name, default)
                values[key] = type(default)(self.get_parameter(name).value)
            if group:
                setattr(self, GROUP_ATTR.get(group, group), values)
            else:
                for key, value in values.items():
                    setattr(self, key, value)

        self.control_period = 1.0 / self.control_rate
        self.zone_speed = {
            'red_zone': self.zone['red_speed'],
            'blue_zone': self.zone['blue_speed'],
        }

    # ------------------------------------------------------------------
    # 콜백
    # ------------------------------------------------------------------
    def _store(self, key, extract):
        def callback(msg):
            self.buf[key].set(extract(msg), self.clock_sec())
        return callback

    def _cb_lane(self, msg):
        if msg.valid:
            self.buf['lane'].set((float(msg.lateral_offset_m),
                                  float(msg.heading_error_rad)), self.clock_sec())
        else:
            self.buf['lane'].value = None        # 차선 유실 -> 페일세이프

    def _cb_reset(self, msg):
        if msg.data:
            with self._lock:
                self._reset_flag = True
            self.lockout.clear()
            self.get_logger().warn('외부 리셋 요청 수신')

    def _cb_mission_objects(self, msg):
        """YOLO 객체 배열 -> 이벤트, 정지선 거리, 마커, 차단기 정보로 분해."""
        now = self.clock_sec()
        rng = self.yolo['event_range']

        best = {}                                    # 클래스별 가장 확실한 1개
        for o in msg.objects:
            if o.class_id == MissionObject.DYNAMIC_CAR:
                continue                             # 회전교차로는 라이다가 담당
            prev = best.get(o.class_id)
            if prev is None or o.confidence > prev.confidence:
                best[o.class_id] = o

        def near(o):
            # 너무 가까우면 화면 밖이라 위치가 invalid -> 가까운 것으로 간주
            return (not o.position_valid) or o.x_m <= rng

        cw = best.get(MissionObject.CROSSWALK)
        gate = best.get(MissionObject.GATE_BAR)
        park = best.get(MissionObject.PARKING_SIGN)
        markers = [o for o in (best.get(MissionObject.MARKER_LEFT),
                               best.get(MissionObject.MARKER_RIGHT)) if o]
        marker = max(markers, key=lambda o: o.confidence) if markers else None

        if cw and near(cw):
            event = 'crosswalk'
        elif gate and near(gate):
            event = 'gate'
        elif marker:
            event = 'marker'
        elif park:
            event = 'parking_sign'
        else:
            event = 'none'
        self.buf['event_yolo'].set(event, now)

        # 사진 촬영 후 지금까지 이동한 거리만큼 보정 (추론·전송 지연 보상)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = max(0.0, now - stamp) if stamp > 0.0 else 0.0
        self.dbg('yolo/age', age)

        if cw and cw.position_valid:
            self.buf['stopline'].set(cw.x_m - self.last_v * age, now)
        if marker:
            left = marker.class_id == MissionObject.MARKER_LEFT
            self.buf['marker_type'].set('left' if left else 'right', now)
            if marker.position_valid:
                self.buf['marker_dist'].set(marker.x_m - self.last_v * age, now)
        if gate:
            if gate.position_valid:
                self.buf['gate_dist_yolo'].set(gate.x_m, now)
            if gate.angle_valid:
                self.buf['gate_open_yolo'].set(
                    gate.angle_rad >= self.yolo['gate_open_angle'], now)

    # ------------------------------------------------------------------
    # 공통 헬퍼
    # ------------------------------------------------------------------
    def ok(self):
        return rclpy.ok()

    def clock_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _latest(self, key):
        return self.buf[key].get(self.clock_sec())

    def consume_reset(self):
        with self._lock:
            flag = self._reset_flag
            self._reset_flag = False
        return flag

    def set_lockout(self, label):
        self.lockout[label] = self.clock_sec() + self.perception['lockout_sec']

    def dbg(self, name, value):
        """튜닝용 내부 값 발행. /supervisor/debug/<name>
        숫자·bool 은 Float32, 문자열은 String 으로 나간다."""
        if not self.debug['enabled'] or value is None:
            return
        is_text = isinstance(value, str)
        with self._debug_lock:
            pub = self._debug_pubs.get(name)
            if pub is None:
                msg_type = String if is_text else Float32
                pub = self.create_publisher(msg_type, '/supervisor/debug/' + name, 10)
                self._debug_pubs[name] = pub
        msg = String() if is_text else Float32()
        msg.data = value if is_text else float(value)
        pub.publish(msg)

    def publish_state(self, label):
        self.current_label = label
        msg = String()
        msg.data = label
        self.state_pub.publish(msg)

    def _publish_state_tick(self):
        msg = String()
        msg.data = self.current_label
        self.state_pub.publish(msg)

    # ------------------------------------------------------------------
    # 인지 조회 (State 들이 사용)
    # ------------------------------------------------------------------
    def lane_error(self, now=None):
        if now is None:
            now = self.clock_sec()
        return self.buf['lane'].get(now)

    def merged_event(self, now=None):
        """여러 인지 노드의 이벤트를 하나로 합친다. 해당 없으면 None."""
        if now is None:
            now = self.clock_sec()
        found = []
        for src in EVENT_SOURCES:
            value = self.buf['event_' + src].get(now)
            if value is not None and value != 'none':
                found.append(value)
        if not found:
            return None
        for name in EVENT_PRIORITY:
            if name in found:
                return name
        return found[0]

    def current_event(self):
        return self.merged_event()

    def detect_mission(self):
        """Supervisor 전이 판정: 유효시간 + 디바운스 + 재진입 차단."""
        now = self.clock_sec()
        raw = self.merged_event(now)
        self.dbg('event', raw if raw else 'none')
        event = self.debouncer.push(raw)
        if event is None:
            return None
        outcome = EVENT_TO_OUTCOME.get(event)
        if outcome is None:
            return None
        if self.lockout.get(OUTCOME_TO_STATE[outcome], 0.0) > now:
            return None
        return outcome

    def stopline_distance(self):
        return self._latest('stopline')

    def gate_distance(self):
        d = self._latest('gate_dist')
        return d if d is not None else self._latest('gate_dist_yolo')

    def gate_is_open(self):
        return bool(self._latest('gate_open')) or bool(self._latest('gate_open_yolo'))

    def obstacles(self):
        return self._latest('obstacles')

    def marker_type(self):
        return self._latest('marker_type')

    def marker_distance(self):
        return self._latest('marker_dist')

    def parking_walls(self):
        return self._latest('walls')

    def odom_yaw(self):
        return self._latest('odom')

    def wall_distances(self):
        """좌/우 섹터 최소거리. 양쪽 모두 유효할 때만 반환."""
        scan = self._latest('scan')
        if scan is None:
            return None
        half = math.radians(self.failsafe['wall_sector_deg']) * 0.5
        max_r = self.failsafe['wall_max_range']
        left, right = [], []
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r <= scan.range_min or r > max_r:
                continue
            a = scan.angle_min + i * scan.angle_increment
            if abs(a - math.pi / 2.0) < half:
                left.append(r)
            elif abs(a + math.pi / 2.0) < half:
                right.append(r)
        if not left or not right:
            return None
        return (min(left), min(right))

    def forward_scan_points(self, max_range):
        """전방 스캔을 (각도, 거리) 목록으로. Follow the Gap 용."""
        scan = self._latest('scan')
        if scan is None:
            return []
        points = []
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r <= scan.range_min:
                continue
            a = scan.angle_min + i * scan.angle_increment
            if abs(a) > math.pi / 2.0:
                continue
            points.append((a, min(r, max_range)))
        points.sort(key=lambda p: p[0])
        return points

    # ------------------------------------------------------------------
    # 제어 출력
    # ------------------------------------------------------------------
    def publish_cmd(self, v, omega):
        if not (math.isfinite(v) and math.isfinite(omega)):
            self.get_logger().warn('비정상 제어값 - 정지')
            self.publish_stop()
            return
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(omega)
        self.cmd_pub.publish(msg)
        self.last_v = float(v)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())
        self.last_v = 0.0


# ----------------------------------------------------------------------
# 상태머신 구성
# ----------------------------------------------------------------------
def build_state_machine(node):
    sm = smach.StateMachine(outcomes=['shutdown'])
    back = {'done': 'LANE_FOLLOWING', 'aborted': 'LANE_FOLLOWING'}

    with sm:
        lane = LaneFollowing(node)
        transitions = dict((o, OUTCOME_TO_STATE[o]) for o in lane.OUTCOMES)
        transitions['aborted'] = 'LANE_FOLLOWING'
        smach.StateMachine.add('LANE_FOLLOWING', lane, transitions=transitions)

        for label in MISSION_LABELS:
            cls = IMPLEMENTED.get(label)
            state = cls(node) if cls else StubState(node, label)
            smach.StateMachine.add(label, state, transitions=back)

    return sm


def main(args=None):
    rclpy.init(args=args)
    node = MissionSupervisor()

    # SMACH execute() 가 블로킹이므로 ROS 콜백은 별도 스레드에서 처리
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    sm = build_state_machine(node)
    try:
        sm.execute()
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()