"""미션 4 - 갈림길 차선 변경.

마커를 인식하면 진입하고, 마커까지 거리가 임계값 이하가 되는 시점에
목표 차선 방향으로 횡방향 오프셋을 점진적으로 적용해 차선을 옮긴다.
인식 직후 즉시 조향하면 분기 지점과 어긋나 차선을 이탈할 수 있으므로
거리 기반 트리거를 사용한다.
"""

from ar_mission_supervisor.base_state import SafeState
from ar_mission_supervisor.controllers import clamp
from ar_mission_supervisor.lane_driver import LaneDriver


class LeftRight(SafeState):

    OUTCOMES = ['done']

    APPROACH = 0
    SHIFT = 1
    SETTLE = 2

    def __init__(self, node):
        SafeState.__init__(self, node, 'LEFT_RIGHT', self.OUTCOMES)
        self.driver = LaneDriver(node)
        self.phase = self.APPROACH
        self.entered_at = None
        self.shift_started = None
        self.settle_started = None
        self.direction = 0.0

    def on_enter(self):
        self.driver.reset()
        self.phase = self.APPROACH
        self.entered_at = self.node.clock_sec()
        self.shift_started = None
        self.settle_started = None
        self.direction = 0.0

    def run(self):
        node = self.node
        cfg = node.left_right

        while node.ok():
            if self.reset_requested():
                return 'aborted'

            now = node.clock_sec()
            if now - self.entered_at > cfg['max_duration']:
                node.get_logger().warn('[LEFT_RIGHT] 제한시간 초과 - 복귀')
                return 'done'

            if self.phase == self.APPROACH:
                if self._approach(now, cfg):
                    continue
            elif self.phase == self.SHIFT:
                self._shift(now, cfg)
            else:
                if self._settle(now, cfg):
                    return 'done'

            self.sleep()

        return 'aborted'

    # ------------------------------------------------------------------
    def _approach(self, now, cfg):
        """마커까지 접근. 트리거 거리에 도달하면 SHIFT 로 전환."""
        node = self.node
        marker = node.marker_type()
        dist = node.marker_distance()

        if marker is not None:
            self.direction = 1.0 if marker == 'left' else -1.0

        if dist is not None and dist <= cfg['trigger_distance']:
            if self.direction == 0.0:
                node.get_logger().warn('[LEFT_RIGHT] 마커 종류 불명 - 복귀')
                return False
            node.get_logger().info(
                '[LEFT_RIGHT] %s 차선 변경 시작 (마커 %.2fm)'
                % ('좌' if self.direction > 0 else '우', dist))
            self.phase = self.SHIFT
            self.shift_started = now
            return False

        if now - self.entered_at > cfg['approach_timeout']:
            node.get_logger().warn('[LEFT_RIGHT] 마커 거리 미확보 - 복귀')
            return False

        self.driver.step(cfg['approach_speed'])
        return False

    def _shift(self, now, cfg):
        """오프셋을 점진적으로 키워 목표 차선으로 이동."""
        elapsed = now - self.shift_started
        progress = clamp(elapsed / cfg['shift_duration'], 0.0, 1.0)
        offset = self.direction * cfg['lane_width'] * progress

        self.driver.step(cfg['shift_speed'], offset)

        if progress >= 1.0:
            self.phase = self.SETTLE
            self.settle_started = now
            self.node.get_logger().info('[LEFT_RIGHT] 이동 완료 - 안정화')

    def _settle(self, now, cfg):
        """새 차선에서 잠시 유지한 뒤 종료."""
        offset = self.direction * cfg['lane_width']
        self.driver.step(cfg['shift_speed'], offset)
        return (now - self.settle_started) >= cfg['settle_sec']