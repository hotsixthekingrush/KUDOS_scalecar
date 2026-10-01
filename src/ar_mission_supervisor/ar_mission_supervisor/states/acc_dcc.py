"""미션 1 - 차로 색상에 따른 감/가속.

조향은 차선 추종을 그대로 유지하고 목표 선속도만 전환한다.
색상 구간이 사라지면 기본 주행으로 복귀한다.
"""

from ar_mission_supervisor.base_state import SafeState
from ar_mission_supervisor.lane_driver import LaneDriver


class AccDcc(SafeState):

    OUTCOMES = ['done']

    def __init__(self, node):
        # 레드존과 블루존이 인접해 있을 수 있으므로 재진입을 막지 않는다.
        SafeState.__init__(self, node, 'ACC_DCC', self.OUTCOMES,
                           lockout_on_exit=False)
        self.driver = LaneDriver(node)
        self.miss = 0
        self.speed = node.base_speed
        self.zone = None

    def on_enter(self):
        self.driver.reset()
        self.miss = 0
        self.speed = self.node.base_speed
        self.zone = None

    def run(self):
        node = self.node
        exit_after = node.perception['exit_miss_count']

        while node.ok():
            if self.reset_requested():
                return 'aborted'

            event = node.current_event()
            target = node.zone_speed.get(event)

            if target is None:
                # 색상 구간을 벗어남 (일시적 미검출 대비해 몇 회 유예)
                self.miss += 1
                if self.miss >= exit_after:
                    node.get_logger().info('[ACC_DCC] 색상 구간 종료')
                    return 'done'
            else:
                self.miss = 0
                if event != self.zone:
                    self.zone = event
                    node.get_logger().info(
                        '[ACC_DCC] %s 진입 - 목표속도 %.2f m/s' % (event, target))
                self.speed = target

            self.driver.step(self.speed)
            self.sleep()
        return 'aborted'