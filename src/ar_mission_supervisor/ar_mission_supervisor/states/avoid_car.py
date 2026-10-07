"""미션 5 - 회전교차로 통과.

규정상 차선 이탈이 실패 조건이므로 자유 경로계획을 하지 않는다.
조향은 차선 추종을 그대로 유지하고, 회전 차량의 교차 지점 통과
예정 시각을 추정해 진입 여부(속도)만 제어한다.
(경로-속도 분해 방식)
"""

import math

from ar_mission_supervisor.base_state import SafeState
from ar_mission_supervisor.geometry import distance
from ar_mission_supervisor.lane_driver import LaneDriver


class AvoidCar(SafeState):

    OUTCOMES = ['done']

    APPROACH = 0
    WAIT = 1
    CROSS = 2

    def __init__(self, node):
        SafeState.__init__(self, node, 'AVOID_CAR', self.OUTCOMES)
        self.driver = LaneDriver(node)
        self.phase = self.APPROACH
        self.entered_at = None
        self.wait_started = None
        self.cross_started = None
        self.emergency_until = 0.0

    def on_enter(self):
        self.driver.reset()
        self.phase = self.APPROACH
        self.entered_at = self.node.clock_sec()
        self.wait_started = None
        self.cross_started = None
        self.emergency_until = 0.0

    def run(self):
        node = self.node
        cfg = node.roundabout

        while node.ok():
            if self.reset_requested():
                return 'aborted'

            now = node.clock_sec()
            node.dbg('roundabout/phase', ('APPROACH', 'WAIT', 'CROSS')[self.phase])
            if now - self.entered_at > cfg['max_duration']:
                node.get_logger().warn('[ROUNDABOUT] 제한시간 초과 - 복귀')
                return 'done'

            if self.phase == self.APPROACH:
                self._approach(now, cfg)
            elif self.phase == self.WAIT:
                self._wait(now, cfg)
            else:
                if self._cross(now, cfg):
                    return 'done'

            self.sleep()

        return 'aborted'

    # ------------------------------------------------------------------
    def _approach(self, now, cfg):
        """중앙 아일랜드까지 거리로 진입 지점 판단."""
        node = self.node
        island = self._find_island(cfg)

        if island is None:
            # 아직 안 보임 - 감속 주행하며 접근
            if now - self.entered_at > cfg['approach_timeout']:
                node.get_logger().warn('[ROUNDABOUT] 아일랜드 미검출 - 현 위치에서 판단')
                self.phase = self.WAIT
                self.wait_started = now
                return
            self.driver.step(cfg['approach_speed'])
            return

        d = distance(island.x, island.y)
        if d <= cfg['entry_distance']:
            node.get_logger().info('[ROUNDABOUT] 진입 지점 도달 (%.2fm) - 판단 시작' % d)
            self.phase = self.WAIT
            self.wait_started = now
            node.publish_stop()
            return

        # 남은 거리에 비례해 감속
        ratio = min(1.0, d / cfg['slowdown_distance'])
        speed = max(cfg['min_speed'], cfg['approach_speed'] * ratio)
        self.driver.step(speed)

    def _wait(self, now, cfg):
        node = self.node
        node.publish_stop()

        if now < self.emergency_until:
            return          # 긴급 정지 직후에는 재판단하지 않음

        waited = now - self.wait_started
        cars = self._find_cars(cfg)

        safe = self._gap_is_safe(cars, cfg, cfg['safety_buffer'])
        node.dbg('roundabout/gap_safe', safe)
        if safe:
            node.get_logger().info('[ROUNDABOUT] 안전 갭 확보 - 진입')
            self.phase = self.CROSS
            self.cross_started = now
            return

        # 제한시간 내 갭이 안 나오면 여유를 줄여 재판단 (폴백)
        if waited > cfg['fallback_after']:
            relaxed = cfg['safety_buffer'] * cfg['fallback_ratio']
            if self._gap_is_safe(cars, cfg, relaxed):
                node.get_logger().warn('[ROUNDABOUT] 폴백 진입 (여유 축소)')
                self.phase = self.CROSS
                self.cross_started = now

    def _cross(self, now, cfg):
        """차선 추종으로 통과. 통과 중에도 실시간 충돌 감시."""
        node = self.node

        # 예측이 빗나갔을 때를 대비한 최후 안전망
        cars = self._find_cars(cfg)
        for c in cars:
            if distance(c.x, c.y) < cfg['emergency_distance']:
                node.get_logger().warn('[ROUNDABOUT] 근접 감지 - 긴급 정지')
                node.publish_stop()
                self.phase = self.WAIT
                self.wait_started = now
                self.emergency_until = now + cfg['emergency_hold']
                return False

        self.driver.step(cfg['cross_speed'])
        return (now - self.cross_started) >= cfg['cross_duration']

    # ------------------------------------------------------------------
    def _find_island(self, cfg):
        """가장 큰 정적 클러스터를 중앙 아일랜드로 간주."""
        obstacles = self.node.obstacles()
        if not obstacles:
            return None
        candidates = [o for o in obstacles
                      if not o.is_dynamic and o.radius >= cfg['island_min_radius']]
        if not candidates:
            return None
        return max(candidates, key=lambda o: o.radius)

    def _find_cars(self, cfg):
        obstacles = self.node.obstacles()
        if not obstacles:
            return []
        return [o for o in obstacles
                if o.is_dynamic and distance(o.x, o.y) <= cfg['track_max_range']]

    def _gap_is_safe(self, cars, cfg, buffer):
        """각 차량이 교차 지점을 통과할 예정 시각과
        로봇의 통과 소요시간이 겹치지 않는지 검사."""
        if not cars:
            return True

        t_cross = cfg['cross_duration']
        for c in cars:
            # 이미 위험 거리 안에 있으면 예측과 무관하게 불가
            if distance(c.x, c.y) < cfg['emergency_distance'] * cfg['clear_margin']:
                return False
    
            t_enter, t_exit = self._occupancy_window(c, cfg)
            if t_enter is None:
                # 속도 추정 불가 - 거리로만 보수적 판단
                if distance(c.x, c.y) < cfg['conflict_radius'] * 2.0:
                    return False
                continue
            # 로봇 점유구간 [0, t_cross] 와 겹치면 위험
            if not (t_exit < -buffer or t_enter > t_cross + buffer):
                return False
        return True

    def _occupancy_window(self, car, cfg):
        """차량이 교차 지점(로봇 전방)을 점유하는 시간대 [t_enter, t_exit].

        교차 지점을 로봇 전방 conflict_x 지점으로 두고,
        차량이 등속 직선으로 접근한다고 근사한다.
        """
        speed = math.hypot(car.vx, car.vy)
        if speed < cfg['min_speed_for_prediction']:
            return (None, None)

        cx, cy = cfg['conflict_x'], 0.0
        r = cfg['conflict_radius']

        # 차량에서 교차 지점까지 상대 위치
        dx = cx - car.x
        dy = cy - car.y

        # 진행 방향 성분 (접근 중이면 양수)
        along = (dx * car.vx + dy * car.vy) / speed
        if along < 0.0:
            # 이미 지나감
            return (-999.0, -998.0)

        # 최근접 거리
        perp = abs(dx * car.vy - dy * car.vx) / speed
        if perp > r:
            # 교차 지점 근처를 지나지 않음
            return (999.0, 1000.0)

        half_chord = math.sqrt(max(0.0, r * r - perp * perp))
        t_enter = (along - half_chord) / speed
        t_exit = (along + half_chord) / speed
        return (t_enter, t_exit)