"""아직 구현하지 않은 미션 State 자리채움."""

from ar_mission_supervisor.base_state import SafeState


class StubState(SafeState):

    def __init__(self, node, label):
        SafeState.__init__(self, node, label, ['done'])

    def run(self):
        self.node.get_logger().warn('[%s] 미구현 State - 정지 후 복귀' % self.label)
        self.node.publish_stop()
        return 'done'