"""미션 7 - 차단기 앞 정지 후 재출발.

차단기 앞에 정차하고, 열림 신호를 받으면 출발한다.
규정: 차단기가 올라간 후 15초 이내 출발 필수.
"""

from ar_mission_supervisor.base_state import SafeState
from ar_mission_supervisor.lane_driver import LaneDriver


class GateStop(SafeState):

    OUTCOMES = ['done']

    def __init__(self, node):
        SafeState.__init__(self, node, 'GATE_STOP', self.OUTCOMES)
        self.driver = LaneDriver(node)
        self.stopped_at = None
        self.approach_started = None
        self.open_since = None

    def on_enter(self):
        self.driver.reset()
        self.stopped_at = None
        self.approach_started = self.node.clock_sec()
        self.open_since = None

    def run(self):
        node = self.node
        cfg = node.gate

        while node.ok():
            if self.reset_requested():
                return 'aborted'

            now = node.clock_sec()

            # --- 1단계: 차단기까지 접근 ---
            if self.stopped_at is None:
                dist = node.gate_distance()

                if dist is None:
                    if now - self.approach_started > cfg['approach_timeout']:
                        node.get_logger().warn(
                            '[GATE] 차단기 미검출 - 현 위치에서 정지')
                        self._begin_stop(now)
                    else:
                        self.driver.step(cfg['approach_speed'])
                elif dist <= cfg['stop_distance']:
                    node.get_logger().info('[GATE] 차단기 %.2fm - 정차' % dist)
                    self._begin_stop(now)
                else:
                    ratio = min(1.0, dist / cfg['slowdown_distance'])
                    speed = max(cfg['min_speed'], cfg['approach_speed'] * ratio)
                    self.driver.step(speed)

                self.sleep()
                continue

            # --- 2단계: 열림 대기 ---
            node.publish_stop()

            if node.gate_is_open():
                if self.open_since is None:
                    self.open_since = now
                    node.get_logger().info('[GATE] 열림 감지')
                # 오검출 방지를 위해 잠시 유지되는지 확인
                elif now - self.open_since >= cfg['open_confirm_sec']:
                    node.get_logger().info('[GATE] 출발')
                    return 'done'
            else:
                self.open_since = None

            # 안전장치: 너무 오래 대기하면 미션 실패 방지를 위해 복귀
            if now - self.stopped_at > cfg['max_wait_sec']:
                node.get_logger().warn('[GATE] 대기 제한시간 초과 - 기본 주행 복귀')
                return 'done'

            self.sleep()
        return 'aborted'

    def _begin_stop(self, now):
        self.stopped_at = now
        self.node.publish_stop()