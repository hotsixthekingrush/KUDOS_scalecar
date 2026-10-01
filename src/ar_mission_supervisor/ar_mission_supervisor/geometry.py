"""경로 계획용 기하 함수."""

import math


def distance(ax, ay, bx=0.0, by=0.0):
    return math.hypot(ax - bx, ay - by)


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def axis_angle(dx, dy):
    """방향 없는 직선의 각도를 (-pi/2, pi/2] 로 정규화."""
    a = math.atan2(dy, dx)
    if a > math.pi / 2.0:
        a -= math.pi
    elif a <= -math.pi / 2.0:
        a += math.pi
    return a


def point_line_distance(px, py, x1, y1, x2, y2):
    """점에서 직선까지 수직 거리."""
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return math.hypot(px - x1, py - y1)
    return abs(dy * (px - x1) - dx * (py - y1)) / n


def line_y_at_x(x, x1, y1, x2, y2):
    """직선이 주어진 x 에서 갖는 y. 세로선이면 None."""
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return None
    return y1 + (x - x1) / dx * (y2 - y1)


def pure_pursuit_curvature(target_x, target_y):
    """base_link 기준 목표점까지의 추종 곡률. kappa = 2y / L^2"""
    l2 = target_x * target_x + target_y * target_y
    if l2 < 1e-6:
        return 0.0
    return 2.0 * target_y / l2


def pick_lookahead(points, lookahead):
    """lookahead 거리를 처음 넘는 점. 없으면 가장 먼 점."""
    if not points:
        return None
    for px, py in points:
        if distance(px, py) >= lookahead:
            return (px, py)
    return points[-1]


def find_largest_gap(scan_points, robot_width, max_range):
    """전방 점들 사이 통과 가능한 가장 넓은 빈 구간의 (중앙 각도, 거리)."""
    if not scan_points:
        return None
    best = None
    prev_a, prev_r = scan_points[0]
    for a, r in scan_points[1:]:
        gap = math.sqrt(prev_r ** 2 + r ** 2
                        - 2.0 * prev_r * r * math.cos(a - prev_a))
        if gap > robot_width:
            mid_a = 0.5 * (prev_a + a)
            mid_r = min(prev_r, r)
            if best is None or gap > best[2]:
                best = (mid_a, mid_r, gap)
        prev_a, prev_r = a, r
    if best is None:
        return None
    return (best[0], min(best[1], max_range))