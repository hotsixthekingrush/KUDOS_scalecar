"""차선 추종 주행 로직. lane_following 과 미션 State 들이 공유한다."""

from ar_mission_supervisor.controllers import pd_steering, steering_to_omega


class LaneDriver(object):

    def __init__(self, node):
        self.node = node
        self.reset()

    def reset(self):
        self.prev_lateral = 0.0
        self.prev_time = None
        self.lost_since = None
        self.last_cmd = (0.0, 0.0)

    def step(self, target_speed, offset=0.0):
        """차선이 보이면 추종, 유실이면 페일세이프. 매 주기 1회 호출.

        offset: 차선 중앙 기준 목표 횡방향 위치(m, 왼쪽 +). 차선 변경에 사용.
        """
        node = self.node
        now = node.clock_sec()
        lane = node.lane_error(now)
        if lane is not None:
            self.lost_since = None
            self._drive(lane, now, target_speed, offset)
        else:
            if self.lost_since is None:
                self.lost_since = now
            self._failsafe(now - self.lost_since)

    # ------------------------------------------------------------------
    def _drive(self, lane, now, target_speed, offset=0.0):
        node = self.node
        lateral, heading = lane
        # lateral = 로봇 기준 차선 중앙의 위치 (왼쪽 +).
        # 목표 지점은 차선 중앙에서 offset 만큼 옆이므로 로봇 기준 위치는 lateral + offset.
        lateral = lateral + offset
        dt = 0.0 if self.prev_time is None else (now - self.prev_time)

        steer = pd_steering(lateral, heading, self.prev_lateral, dt,
                            node.lane_gains, node.max_steer)
        omega = steering_to_omega(target_speed, steer, node.wheel_base)

        self.prev_lateral = lateral
        self.prev_time = now
        self.last_cmd = (target_speed, omega)
        node.publish_cmd(target_speed, omega)

        node.dbg('lane/mode', 'track')
        node.dbg('lane/lateral', lateral)
        node.dbg('lane/heading', heading)
        node.dbg('lane/steer', steer)

    def _failsafe(self, lost_sec):
        """유실 지속시간에 따른 단계적 대응."""
        node = self.node
        cfg = node.failsafe
        node.dbg('lane/lost_sec', lost_sec)

        # (1) 단기 유실 - 직전 명령 유지
        if lost_sec < cfg['hold_sec']:
            node.dbg('lane/mode', 'hold')
            v, omega = self.last_cmd
            node.publish_cmd(v, omega)
            return

        # (2) 양측 벽 감지 - 벽 추종 (터널 구간 포함)
        walls = node.wall_distances()
        if walls is not None:
            left, right = walls
            node.dbg('lane/mode', 'wall')
            node.dbg('lane/wall_err', left - right)
            steer = node.lane_gains['kp_lateral'] * (left - right)
            steer = max(-node.max_steer, min(node.max_steer, steer))
            v = cfg['creep_speed']
            node.publish_cmd(v, steering_to_omega(v, steer, node.wheel_base))
            return

        # (3) 벽도 없음 - 각속도 0으로 저속 직진
        if lost_sec < cfg['stop_after_sec']:
            node.dbg('lane/mode', 'creep')
            node.publish_cmd(cfg['creep_speed'], 0.0)
            return

        # (4) 장시간 유실 - 정차 후 재검출 대기
        node.dbg('lane/mode', 'stop')
        node.publish_stop()