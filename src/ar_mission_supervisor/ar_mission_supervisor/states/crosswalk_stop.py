"""미션 2 - 횡단보도 정지 후 출발.

YOLO 의 횡단보도 거리는 base_link ~ 횡단보도 앞쪽 끝.
가까워지면 화면 밖으로 빠져 거리가 끊기므로, 마지막 측정값에서
주행 거리를 빼는 방식으로 남은 거리를 추정한다.
규정: 횡단보도 ±30cm 안에 정지, 정지 후 5초 경과부터 출발, 15초 내 출발.
"""

from ar_mission_supervisor.base_state import SafeState
from ar_mission_supervisor.lane_driver import LaneDriver


class CrosswalkStop(SafeState):

    OUTCOMES = ['done']

    def __init__(self, node):
        SafeState.__init__(self, node, 'CROSSWALK_STOP', self.OUTCOMES)
        self.driver = LaneDriver(node)
        self.stopped_at = None
        self.approach_started = None
        self.est_dist = None
        self.last_time = None
        self.last_speed = 0.0

    def on_enter(self):
        self.driver.reset()
        self.stopped_at = None
        self.approach_started = self.node.clock_sec()
        self.est_dist = None
        self.last_time = None
        self.last_speed = 0.0

    def run(self):
        node = self.node
        cfg = node.crosswalk

        while node.ok():
            if self.reset_requested():
                return 'aborted'

            now = node.clock_sec()

            # --- 1단계: 정지 지점까지 접근 ---
            if self.stopped_at is None:
                self._update_estimate(now)
                node.dbg('crosswalk/est_dist', self.est_dist)

                if self.est_dist is None:
                    if now - self.approach_started > cfg['approach_timeout']:
                        node.get_logger().warn('[CROSSWALK] 거리 미확보 - 현 위치에서 정지')
                        self._begin_stop(now)
                    else:
                        self._drive(cfg['approach_speed'], now)
                elif self.est_dist <= cfg['stop_distance']:
                    node.get_logger().info('[CROSSWALK] 남은 거리 %.2fm - 정차' % self.est_dist)
                    self._begin_stop(now)
                else:
                    ratio = min(1.0, self.est_dist / cfg['slowdown_distance'])
                    speed = max(cfg['min_speed'], cfg['approach_speed'] * ratio)
                    self._drive(speed, now)

                self.sleep()
                continue

            # --- 2단계: 정차 대기 ---
            node.publish_stop()
            waited = now - self.stopped_at
            if waited >= cfg['wait_sec']:
                node.get_logger().info('[CROSSWALK] %.1f초 대기 완료 - 출발' % waited)
                return 'done'

            self.sleep()
        return 'aborted'

    def _update_estimate(self, now):
        measured = self.node.stopline_distance()
        if measured is not None:
            self.est_dist = measured
        elif self.est_dist is not None and self.last_time is not None:
            self.est_dist -= self.last_speed * (now - self.last_time)
        self.last_time = now

    def _drive(self, speed, now):
        self.driver.step(speed)
        self.last_speed = speed

    def _begin_stop(self, now):
        self.stopped_at = now
        self.last_speed = 0.0
        self.node.publish_stop()