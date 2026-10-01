"""미션 3 - 라바콘 회피.

차선 인식이 아닌 라이다 기반으로 주행한다.
좌우 라바콘을 짝지어 게이트 중점을 만들고 Pure Pursuit 로 추종하며,
페어링이 불안정하면 Follow the Gap 으로 전환한다.
"""

import math

from ar_mission_supervisor.base_state import SafeState
from ar_mission_supervisor.controllers import steering_to_omega, clamp
from ar_mission_supervisor.geometry import (
    distance, pure_pursuit_curvature, pick_lookahead, find_largest_gap)


class AvoidCone(SafeState):

    OUTCOMES = ['done']

    def __init__(self, node):
        SafeState.__init__(self, node, 'AVOID_CONE', self.OUTCOMES)
        self.miss = 0
        self.entered_at = None
        self.last_steer = 0.0

    def on_enter(self):
        self.miss = 0
        self.entered_at = self.node.clock_sec()
        self.last_steer = 0.0

    def run(self):
        node = self.node
        cfg = node.cone

        while node.ok():
            if self.reset_requested():
                return 'aborted'

            now = node.clock_sec()
            if now - self.entered_at > cfg['max_duration']:
                node.get_logger().warn('[CONE] 제한시간 초과 - 복귀')
                return 'done'

            cones = self._collect_cones()
            node.dbg('cone/count', len(cones))

            if not cones:
                # 라바콘이 안 보임 - 구간 종료로 판단 (몇 회 유예)
                self.miss += 1
                if self.miss >= node.perception['exit_miss_count']:
                    node.get_logger().info('[CONE] 라바콘 구간 종료')
                    return 'done'
                node.publish_cmd(cfg['speed'], 
                                 steering_to_omega(cfg['speed'], self.last_steer,
                                                   node.wheel_base))
                self.sleep()
                continue

            self.miss = 0
            target = self._make_target(cones, cfg)

            if target is None:
                node.get_logger().warn('[CONE] 경로 없음 - 감속 직진')
                node.publish_cmd(cfg['creep_speed'], 0.0)
                self.sleep()
                continue

            self._drive_to(target, cfg)
            self.sleep()

        return 'aborted'

    # ------------------------------------------------------------------
    def _collect_cones(self):
        """전방 ROI 안의 정적 클러스터만 라바콘 후보로 추린다."""
        node = self.node
        cfg = node.cone
        obstacles = node.obstacles()
        if obstacles is None:
            return []

        cones = []
        for o in obstacles:
            if o.is_dynamic:
                continue
            if o.x < cfg['roi_min_x'] or o.x > cfg['roi_max_x']:
                continue
            if abs(o.y) > cfg['roi_max_y']:
                continue
            if o.radius > cfg['max_radius']:
                continue
            cones.append(o)
        return cones

    def _make_target(self, cones, cfg):
        """게이트 중점 시퀀스 -> lookahead 목표점."""
        left = sorted([c for c in cones if c.y > 0.0], key=lambda c: c.x)
        right = sorted([c for c in cones if c.y <= 0.0], key=lambda c: c.x)

        gates = []
        for l, r in zip(left, right):
            # 전후 거리가 크게 어긋나면 잘못된 짝
            if abs(l.x - r.x) > cfg['pair_max_dx']:
                break
            width = distance(l.x, l.y, r.x, r.y) - (l.radius + r.radius)
            if width < cfg['min_gate_width']:
                break
            gates.append((0.5 * (l.x + r.x), 0.5 * (l.y + r.y)))

        if gates:
            self.node.dbg('cone/mode', 'gate')
            return pick_lookahead(gates, cfg['lookahead'])

        # 한쪽만 보이는 경우 - 고정 오프셋으로 가상 목표
        single = left if left else right
        if single:
            self.node.dbg('cone/mode', 'single')
            c = single[0]
            sign = -1.0 if c.y > 0.0 else 1.0
            offset = c.radius + cfg['side_offset']
            return (c.x, c.y + sign * offset)

        self.node.dbg('cone/mode', 'gap')
        return self._follow_the_gap(cfg)

    def _follow_the_gap(self, cfg):
        """페어링 실패 시 폴백."""
        node = self.node
        points = node.forward_scan_points(cfg['roi_max_x'])
        result = find_largest_gap(points, cfg['robot_width'], cfg['lookahead'])
        if result is None:
            return None
        angle, rng = result
        return (rng * math.cos(angle), rng * math.sin(angle))

    def _drive_to(self, target, cfg):
        node = self.node
        tx, ty = target
        kappa = pure_pursuit_curvature(tx, ty)

        # 곡률 -> 조향각 (애커만 기구학 역변환)
        steer = math.atan(kappa * node.wheel_base)
        steer = clamp(steer, -node.max_steer, node.max_steer)
        self.last_steer = steer

        # 곡률이 클수록 감속
        speed = cfg['speed'] / (1.0 + cfg['curve_gain'] * abs(kappa))
        speed = max(cfg['creep_speed'], speed)

        node.publish_cmd(speed, steering_to_omega(speed, steer, node.wheel_base))

        node.dbg('cone/target_x', tx)
        node.dbg('cone/target_y', ty)
        node.dbg('cone/kappa', kappa)