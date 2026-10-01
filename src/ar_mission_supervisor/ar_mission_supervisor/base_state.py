"""모든 미션 State의 공통 부모.

SMACH의 execute()는 블로킹이므로, 각 State는 run() 안에서 while 루프를 돈다.
예외 처리와 리셋 확인을 여기 한 곳에 모아 개별 State 구현을 단순하게 유지한다.
"""

import time

import smach


class SafeState(smach.State):

    def __init__(self, node, label, outcomes, lockout_on_exit=True):
        outcomes = list(outcomes)
        if 'aborted' not in outcomes:
            outcomes.append('aborted')
        smach.State.__init__(self, outcomes=outcomes)
        self.node = node
        self.label = label
        self.lockout_on_exit = lockout_on_exit

    # -- SMACH 진입점 --------------------------------------------------
    def execute(self, userdata):
        self.node.get_logger().info('전이: -> %s' % self.label)
        self.node.publish_state(self.label)
        self.node.publish_state(self.label)
        try:
            self.on_enter()
            return self.run()
        except Exception as exc:
            self.node.get_logger().error('[%s] 예외: %r' % (self.label, exc))
            self.node.publish_stop()
            return 'aborted'
        finally:
            if self.lockout_on_exit:
                self.node.set_lockout(self.label)

    # -- 하위 클래스가 구현 --------------------------------------------
    def on_enter(self):
        """진입 시 1회. 내부 변수 초기화용."""
        pass

    def run(self):
        """while 루프. outcomes 중 하나를 문자열로 반환하며 종료."""
        raise NotImplementedError

    # -- 편의 함수 -----------------------------------------------------
    def sleep(self):
        time.sleep(self.node.control_period)

    def reset_requested(self):
        """외부 리셋 요청 확인. True면 즉시 'aborted' 반환할 것."""
        return self.node.consume_reset()