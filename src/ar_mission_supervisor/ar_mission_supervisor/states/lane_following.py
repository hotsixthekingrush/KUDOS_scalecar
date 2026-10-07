"""기본 주행 State. 터널 대응 페일세이프는 LaneDriver 내부에서 처리한다."""

from ar_mission_supervisor.base_state import SafeState
from ar_mission_supervisor.lane_driver import LaneDriver


class LaneFollowing(SafeState):

    OUTCOMES = ['to_acc_dcc', 'to_crosswalk', 'to_cone', 'to_left_right',
                'to_roundabout', 'to_gate', 'to_parking']

    def __init__(self, node):
        SafeState.__init__(self, node, 'LANE_FOLLOWING', self.OUTCOMES,
                           lockout_on_exit=False)
        self.driver = LaneDriver(node)

    def on_enter(self):
        self.driver.reset()

    def run(self):
        node = self.node
        while node.ok():
            if self.reset_requested():
                return 'aborted'

            outcome = node.detect_mission()
            if outcome is not None:
                return outcome

            self.driver.step(node.base_speed)
            self.sleep()
        return 'aborted'