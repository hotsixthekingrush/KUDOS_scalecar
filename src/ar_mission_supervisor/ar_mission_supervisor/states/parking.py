"""미션 8 - 주차 (1단계: 템플릿 기동).

T자 포켓에 후진으로 진입한다.
  SCAN  : 저속 차선 추종하며 벽 관측, 포켓을 지나친 지점에서 정지
  TURN  : 최대 조향으로 후진하며 90도 회전
  ALIGN : 옆벽 기준으로 방향/중앙을 보정하며 후진, 뒷벽 거리에서 정지
  PARKED: 정지 유지 (마지막 미션이므로 기본 주행으로 돌아가지 않음)
실패 시 정지 후 외부 리셋을 기다린다.
"""

import math

from ar_msgs.msg import WallSegment

from ar_mission_supervisor.base_state import SafeState
from ar_mission_supervisor.controllers import clamp, steering_to_omega
from ar_mission_supervisor.geometry import (
    normalize_angle, axis_angle, point_line_distance, line_y_at_x)
from ar_mission_supervisor.lane_driver import LaneDriver


class Parking(SafeState):

    OUTCOMES = ['done']

    SCAN, TURN, ALIGN, PARKED, FAILED = range(5)

    PHASE_NAMES = ('SCAN', 'TURN', 'ALIGN', 'PARKED', 'FAILED')

    def __init__(self, node):
        SafeState.__init__(self, node, 'PARKING', self.OUTCOMES)
        self.driver = LaneDriver(node)
        self.phase = self.SCAN
        self.phase_started = 0.0
        self.turn_start_yaw = None

    def on_enter(self):
        self.driver.reset()
        self.phase = self.SCAN
        self.phase_started = self.node.clock_sec()
        self.turn_start_yaw = None

    def run(self):
        node = self.node
        cfg = node.parking
        while node.ok():
            if self.reset_requested():
                return 'aborted'

            now = node.clock_sec()
            node.dbg('parking/phase', self.PHASE_NAMES[self.phase])
            if self.phase == self.SCAN:
                self._scan(now, cfg)
            elif self.phase == self.TURN:
                self._turn(now, cfg)
            elif self.phase == self.ALIGN:
                self._align(now, cfg)
            else:
                node.publish_stop()          # PARKED / FAILED : 정지 유지

            self.sleep()
        return 'aborted'

    # ------------------------------------------------------------------
    # 단계별 동작
    # ------------------------------------------------------------------
    def _scan(self, now, cfg):
        node = self.node
        if now - self.phase_started > cfg['scan_timeout']:
            self._fail('포켓을 찾지 못함')
            return

        walls = node.parking_walls()
        cx = self._pocket_center_x(walls) if walls else None
        node.dbg('parking/pocket_x', cx)

        if cx is not None and cx <= -cfg['turn_offset']:
            node.publish_stop()
            self.turn_start_yaw = None
            self._set_phase(self.TURN, now,
                            '포켓 통과 (중심이 %.2fm 뒤) - 후진 회전' % (-cx))
            return

        self.driver.step(cfg['scan_speed'])

    def _turn(self, now, cfg):
        node = self.node
        elapsed = now - self.phase_started

        if elapsed < cfg['phase_pause']:
            node.publish_stop()              # 완전히 멈춘 뒤 후진 시작
            return
        if elapsed > cfg['turn_timeout']:
            self._fail('회전 제한시간 초과')
            return

        if self.turn_start_yaw is None:
            self.turn_start_yaw = node.odom_yaw()

        yaw = node.odom_yaw()
        if yaw is not None and self.turn_start_yaw is not None:
            turned = abs(normalize_angle(yaw - self.turn_start_yaw))
        else:
            # odom 이 없으면 기구학으로 회전량 추정
            rate = cfg['reverse_speed'] * math.tan(node.max_steer) / node.wheel_base
            turned = (elapsed - cfg['phase_pause']) * rate

        node.dbg('parking/turned_deg', math.degrees(turned))

        if turned >= math.pi / 2.0 - cfg['yaw_tolerance']:
            node.publish_stop()
            self._set_phase(self.ALIGN, now,
                            '회전 완료 (%.0f도) - 정렬 후진' % math.degrees(turned))
            return

        v = -cfg['reverse_speed']
        steer = self._side(cfg) * node.max_steer
        node.publish_cmd(v, steering_to_omega(v, steer, node.wheel_base))

    def _align(self, now, cfg):
        node = self.node
        elapsed = now - self.phase_started

        if elapsed < cfg['phase_pause']:
            node.publish_stop()
            return
        if elapsed > cfg['align_timeout']:
            self._fail('정렬 제한시간 초과')
            return

        walls = node.parking_walls()
        if not walls:
            node.publish_stop()              # 벽이 안 보이면 멈춰서 재관측
            return

        back, sides = self._classify(walls, back_parallel=False)

        # 정지 판정: 뒷벽까지 거리 (차체 절반 + 여유)
        if back is not None:
            d = point_line_distance(0.0, 0.0, back.x1, back.y1, back.x2, back.y2)
            node.dbg('parking/back_dist', d)
            behind = 0.5 * (back.x1 + back.x2) < 0.0
            stop_dist = 0.5 * cfg['robot_length'] + cfg['clearance']
            if behind and d <= stop_dist:
                node.publish_stop()
                self._set_phase(self.PARKED, now, '주차 완료 (뒷벽까지 %.2fm)' % d)
                return

        # 옆벽 기준 보정 (라이다 실측 기반이라 odom 누적 오차와 무관)
        steer = 0.0
        if sides:
            angles = [axis_angle(s.x2 - s.x1, s.y2 - s.y1) for s in sides]
            heading_err = sum(angles) / len(angles)
            node.dbg('parking/heading_err', heading_err)
            steer -= cfg['k_heading'] * heading_err          # 후진이라 부호 반전

            ys = [line_y_at_x(0.0, s.x1, s.y1, s.x2, s.y2) for s in sides]
            ys = [y for y in ys if y is not None]
            if len(ys) >= 2:
                center = 0.5 * (max(ys) + min(ys))            # 포켓 중앙의 횡방향 위치
                node.dbg('parking/center', center)
                steer += cfg['k_lateral'] * center

        steer = clamp(steer, -node.max_steer, node.max_steer)
        v = -cfg['align_speed']
        node.publish_cmd(v, steering_to_omega(v, steer, node.wheel_base))

    # ------------------------------------------------------------------
    # 보조 함수
    # ------------------------------------------------------------------
    def _classify(self, walls, back_parallel):
        """role 이 있으면 그대로, 없으면 로봇 축과의 각도로 뒷벽을 판별.

        back_parallel=True  : 뒷벽이 로봇 진행축과 나란함 (SCAN 단계)
        back_parallel=False : 뒷벽이 로봇 진행축과 수직   (ALIGN 단계)
        """
        back, sides, unknown = None, [], []
        for w in walls:
            if w.role == WallSegment.ROLE_BACK:
                back = w
            elif w.role == WallSegment.ROLE_SIDE:
                sides.append(w)
            else:
                unknown.append(w)

        if back is None and unknown:
            def parallel(w):
                return abs(math.cos(axis_angle(w.x2 - w.x1, w.y2 - w.y1)))
            if back_parallel:
                back = max(unknown, key=parallel)
            else:
                back = min(unknown, key=parallel)
            sides += [w for w in unknown if w is not back]

        return back, sides

    def _pocket_center_x(self, walls):
        back, sides = self._classify(walls, back_parallel=True)
        if back is not None:
            return 0.5 * (back.x1 + back.x2)
        if sides:
            xs = [0.5 * (s.x1 + s.x2) for s in sides]
            return sum(xs) / len(xs)
        return None

    def _side(self, cfg):
        return 1.0 if cfg['side'] == 'left' else -1.0

    def _set_phase(self, phase, now, msg):
        self.phase = phase
        self.phase_started = now
        self.node.get_logger().info('[PARKING] %s' % msg)

    def _fail(self, reason):
        self.node.publish_stop()
        self.phase = self.FAILED
        self.node.get_logger().warn('[PARKING] 실패: %s - 정지 후 리셋 대기' % reason)