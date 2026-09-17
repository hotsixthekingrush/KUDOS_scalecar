 
#!/usr/bin/env python3

from pathlib import Path
import math
import os
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ============================================================
# Path
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PACKAGE_DIR.parent.parent

PACKAGE_WORLD_PATH = PACKAGE_DIR / "worlds" / "limo_competition.world"
ROOT_WORLD_PATH = WORKSPACE_DIR / "worlds" / "limo_competition.world"

# ============================================================
# Arena / common
# ============================================================

ARENA_LENGTH = 16.0
ARENA_WIDTH = 9.5

GROUND_Z = -0.025
GROUND_THICKNESS = 0.05

ROAD_WIDTH = 0.80

EDGE_LINE_WIDTH = 0.025
WHITE_LINE_WIDTH = 0.035
LINE_HEIGHT = 0.004
LINE_Z = LINE_HEIGHT / 2.0 + 0.001

# ============================================================
# Colors
# ============================================================

COLOR_GROUND = (0.08, 0.08, 0.08, 1.0)

COLOR_YELLOW = (0.88, 0.82, 0.05, 1.0)
COLOR_WHITE = (0.95, 0.95, 0.95, 1.0)
COLOR_BLACK = (0.03, 0.03, 0.03, 1.0)

COLOR_RED = (0.78, 0.16, 0.12, 1.0)
COLOR_BLUE = (0.18, 0.24, 0.70, 1.0)
COLOR_ORANGE = (0.98, 0.33, 0.02, 1.0)

COLOR_GREY = (0.35, 0.35, 0.35, 1.0)
COLOR_DARK_GREY = (0.12, 0.12, 0.12, 1.0)
COLOR_P_BLUE = (0.05, 0.22, 0.78, 1.0)

# ============================================================
# Global geometry (Z-shape base)
# ============================================================

# Top straight
TOP_X_START = -7.25
TOP_X_END = 5.05
TOP_Y_TOP = 4.35
TOP_Y_BOTTOM = 3.55

# Right U-turn
RIGHT_UTURN_CX = 5.05
RIGHT_UTURN_CY = 3.20
RIGHT_UTURN_OUTER_R = 1.15
RIGHT_UTURN_INNER_R = 0.35
RIGHT_UTURN_START_DEG = 90
RIGHT_UTURN_END_DEG = -72

# M4 lane change
M4_CX = 1.20
M4_CY = 1.25
M4_YAW = math.radians(18.0)
M4_LANE_WIDTH = ROAD_WIDTH
M4_DOUBLE_WIDTH = 1.60
M4_TOTAL_LENGTH = 4.00
M4_TRANSITION_LENGTH = 0.80

# Roundabout
ROUNDABOUT_CX = -2.15
ROUNDABOUT_CY = 0.20
ROUNDABOUT_OUTER_R = 1.00
ROUNDABOUT_ISLAND_R = 0.32

# Left U-turn
LEFT_UTURN_CX = -6.00
LEFT_UTURN_CY = -2.55
LEFT_UTURN_OUTER_R = 1.40
LEFT_UTURN_INNER_R = 0.60
LEFT_UTURN_START_DEG = 108
LEFT_UTURN_END_DEG = 270

# Bottom straight
BOTTOM_X_START = LEFT_UTURN_CX
BOTTOM_X_END = 7.25
BOTTOM_Y_TOP = -3.15
BOTTOM_Y_BOTTOM = -3.95


# ============================================================
# XML helpers
# ============================================================

name_counter = {}


def unique_name(prefix):
    if prefix not in name_counter:
        name_counter[prefix] = 0
    name_counter[prefix] += 1
    return f"{prefix}_{name_counter[prefix]}"


def rgba_text(color):
    return " ".join(str(v) for v in color)


def pose_text(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    return f"{x} {y} {z} {roll} {pitch} {yaw}"


def add_material(parent, color):
    material = ET.SubElement(parent, "material")

    ambient = ET.SubElement(material, "ambient")
    ambient.text = rgba_text(color)

    diffuse = ET.SubElement(material, "diffuse")
    diffuse.text = rgba_text(color)

    specular = ET.SubElement(material, "specular")
    specular.text = "0.05 0.05 0.05 1"


def add_box_model(
    world,
    name,
    x,
    y,
    z,
    length,
    width,
    height,
    yaw=0.0,
    color=COLOR_WHITE,
    collision=True
):
    model = ET.SubElement(world, "model", {"name": name})

    static = ET.SubElement(model, "static")
    static.text = "true"

    pose = ET.SubElement(model, "pose")
    pose.text = pose_text(x, y, z, 0, 0, yaw)

    link = ET.SubElement(model, "link", {"name": "link"})

    if collision:
        collision_element = ET.SubElement(link, "collision", {"name": "collision"})
        geometry = ET.SubElement(collision_element, "geometry")
        box = ET.SubElement(geometry, "box")
        size = ET.SubElement(box, "size")
        size.text = f"{length} {width} {height}"

    visual = ET.SubElement(link, "visual", {"name": "visual"})
    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box")
    size = ET.SubElement(box, "size")
    size.text = f"{length} {width} {height}"
    add_material(visual, color)

    return model


def add_cylinder_model(
    world,
    name,
    x,
    y,
    z,
    radius,
    length,
    color,
    collision=True
):
    model = ET.SubElement(world, "model", {"name": name})

    static = ET.SubElement(model, "static")
    static.text = "true"

    pose = ET.SubElement(model, "pose")
    pose.text = pose_text(x, y, z)

    link = ET.SubElement(model, "link", {"name": "link"})

    if collision:
        collision_element = ET.SubElement(link, "collision", {"name": "collision"})
        geometry = ET.SubElement(collision_element, "geometry")
        cylinder = ET.SubElement(geometry, "cylinder")

        radius_element = ET.SubElement(cylinder, "radius")
        radius_element.text = str(radius)

        length_element = ET.SubElement(cylinder, "length")
        length_element.text = str(length)

    visual = ET.SubElement(link, "visual", {"name": "visual"})
    geometry = ET.SubElement(visual, "geometry")
    cylinder = ET.SubElement(geometry, "cylinder")

    radius_element = ET.SubElement(cylinder, "radius")
    radius_element.text = str(radius)

    length_element = ET.SubElement(cylinder, "length")
    length_element.text = str(length)

    add_material(visual, color)
    return model



# ============================================================
# Oriented primitive / sign helper
# ============================================================

def add_box_model_pose(
    world,
    name,
    x,
    y,
    z,
    length,
    width,
    height,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    color=COLOR_WHITE,
    collision=True
):
    model = ET.SubElement(world, "model", {"name": name})

    static = ET.SubElement(model, "static")
    static.text = "true"

    pose = ET.SubElement(model, "pose")
    pose.text = pose_text(x, y, z, roll, pitch, yaw)

    link = ET.SubElement(model, "link", {"name": "link"})

    if collision:
        collision_element = ET.SubElement(link, "collision", {"name": "collision"})
        geometry = ET.SubElement(collision_element, "geometry")
        box = ET.SubElement(geometry, "box")
        size = ET.SubElement(box, "size")
        size.text = f"{length} {width} {height}"

    visual = ET.SubElement(link, "visual", {"name": "visual"})
    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box")
    size = ET.SubElement(box, "size")
    size.text = f"{length} {width} {height}"

    add_material(visual, color)
    return model


def add_cylinder_model_pose(
    world,
    name,
    x,
    y,
    z,
    radius,
    length,
    color,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    collision=True
):
    model = ET.SubElement(world, "model", {"name": name})

    static = ET.SubElement(model, "static")
    static.text = "true"

    pose = ET.SubElement(model, "pose")
    pose.text = pose_text(x, y, z, roll, pitch, yaw)

    link = ET.SubElement(model, "link", {"name": "link"})

    if collision:
        collision_element = ET.SubElement(link, "collision", {"name": "collision"})
        geometry = ET.SubElement(collision_element, "geometry")
        cylinder = ET.SubElement(geometry, "cylinder")

        radius_element = ET.SubElement(cylinder, "radius")
        radius_element.text = str(radius)

        length_element = ET.SubElement(cylinder, "length")
        length_element.text = str(length)

    visual = ET.SubElement(link, "visual", {"name": "visual"})
    geometry = ET.SubElement(visual, "geometry")
    cylinder = ET.SubElement(geometry, "cylinder")

    radius_element = ET.SubElement(cylinder, "radius")
    radius_element.text = str(radius)

    length_element = ET.SubElement(cylinder, "length")
    length_element.text = str(length)

    add_material(visual, color)
    return model


def resolve_m4_sign_dirs():
    """
    M4_SIGN_MODE:
      - random : make_arena.py 실행할 때마다 랜덤
      - right  : 시작 right, 종료 left
      - left   : 시작 left, 종료 right
    """
    mode = os.getenv("M4_SIGN_MODE", "random").strip().lower()

    if mode == "right":
        start_dir = "right"
    elif mode == "left":
        start_dir = "left"
    else:
        start_dir = random.choice(["left", "right"])

    end_dir = "left" if start_dir == "right" else "right"
    return start_dir, end_dir



def add_standing_circular_sign(world, prefix, x, y, face_yaw, direction="right"):
    """
    갈림길용 세워진 원형 표지판.

    수정사항:
    - 파란 원판 크기 증가
    - 원판 두께 증가
    - 봉이 원판 앞쪽으로 튀어나와 보이지 않도록 높이 조정
    - 화살표를 원판 앞면에 크게 표시
    """

    # --------------------------------------------------------
    # 봉
    # 원판 중심보다 아래까지만 올라오게 한다.
    # --------------------------------------------------------

    pole_height = 0.78
    pole_z = pole_height / 2.0

    add_box_model(
        world,
        unique_name(prefix + "_pole"),
        x,
        y,
        pole_z,
        0.055,
        0.055,
        pole_height,
        color=COLOR_GREY,
        collision=True
    )

    # --------------------------------------------------------
    # 파란 원판
    # --------------------------------------------------------

    panel_z = 0.91

    # 기존보다 크게
    panel_radius = 0.24

    # 기존보다 두껍게
    panel_thickness = 0.10

    add_cylinder_model_pose(
        world,
        unique_name(prefix + "_disc"),
        x,
        y,
        panel_z,
        panel_radius,
        panel_thickness,
        COLOR_BLUE,

        # 원기둥 축을 수평으로 눕혀 세운 원판으로 만듦
        roll=0.0,
        pitch=math.pi / 2.0,
        yaw=face_yaw,

        collision=True
    )

    # --------------------------------------------------------
    # 화살표
    #
    # 표지판 앞쪽에 흰색 box 3개를 붙여
    # shaft + 위쪽 head + 아래쪽 head로 만든다.
    # --------------------------------------------------------

    # 표지판 정면 방향
    fx = math.cos(face_yaw)
    fy = math.sin(face_yaw)

    # 표지판 면 위의 좌우 방향
    rx = -math.sin(face_yaw)
    ry = math.cos(face_yaw)

    # 원판 앞면보다 살짝 앞으로
    front_offset = panel_thickness / 2.0 + 0.012

    direction_sign = 1.0 if direction == "right" else -1.0

    # --------------------------------------------------------
    # 화살표 몸통
    # --------------------------------------------------------

    shaft_center_side = -0.025 * direction_sign

    shaft_x = x + fx * front_offset + rx * shaft_center_side
    shaft_y = y + fy * front_offset + ry * shaft_center_side

    add_box_model_pose(
        world,
        unique_name(prefix + "_arrow_shaft"),
        shaft_x,
        shaft_y,
        panel_z,

        # box dimensions
        0.018,
        0.16,
        0.030,

        roll=0.0,
        pitch=0.0,
        yaw=face_yaw,

        color=COLOR_WHITE,
        collision=False
    )

    # --------------------------------------------------------
    # 화살표 머리
    #
    # 끝점 위치
    # --------------------------------------------------------

    tip_side = 0.072 * direction_sign

    tip_x = x + fx * front_offset + rx * tip_side
    tip_y = y + fy * front_offset + ry * tip_side

    # 위쪽 대각선
    add_box_model_pose(
        world,
        unique_name(prefix + "_arrow_head_upper"),
        tip_x,
        tip_y,
        panel_z + 0.032,

        0.018,
        0.075,
        0.025,

        roll=0.0,
        pitch=0.0,
        yaw=face_yaw + 0.62 * direction_sign,

        color=COLOR_WHITE,
        collision=False
    )

    # 아래쪽 대각선
    add_box_model_pose(
        world,
        unique_name(prefix + "_arrow_head_lower"),
        tip_x,
        tip_y,
        panel_z - 0.032,

        0.018,
        0.075,
        0.025,

        roll=0.0,
        pitch=0.0,
        yaw=face_yaw - 0.62 * direction_sign,

        color=COLOR_WHITE,
        collision=False
    )

def arc_point(cx, cy, r, deg):
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def local_to_world(s, t, cx, cy, yaw):
    x = cx + s * math.cos(yaw) - t * math.sin(yaw)
    y = cy + s * math.sin(yaw) + t * math.cos(yaw)
    return x, y


def add_line_segment(
    world,
    prefix,
    x1,
    y1,
    x2,
    y2,
    width=EDGE_LINE_WIDTH,
    color=COLOR_YELLOW,
    z=LINE_Z,
    collision=False
):
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length < 1e-8:
        return

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    yaw = math.atan2(dy, dx)

    add_box_model(
        world=world,
        name=unique_name(prefix),
        x=cx,
        y=cy,
        z=z,
        length=length,
        width=width,
        height=LINE_HEIGHT,
        yaw=yaw,
        color=color,
        collision=collision
    )


def add_polyline(world, prefix, points, width=EDGE_LINE_WIDTH, color=COLOR_YELLOW):
    for i in range(len(points) - 1):
        add_line_segment(
            world,
            prefix,
            points[i][0],
            points[i][1],
            points[i + 1][0],
            points[i + 1][1],
            width=width,
            color=color
        )


def add_arc(
    world,
    prefix,
    cx,
    cy,
    radius,
    angle_start,
    angle_end,
    segments=24,
    width=EDGE_LINE_WIDTH,
    color=COLOR_YELLOW
):
    points = []

    for i in range(segments + 1):
        ratio = i / segments
        deg = angle_start + (angle_end - angle_start) * ratio
        points.append(arc_point(cx, cy, radius, deg))

    add_polyline(world, prefix, points, width=width, color=color)


def add_local_line(
    world,
    prefix,
    s1,
    t1,
    s2,
    t2,
    cx,
    cy,
    yaw,
    width=EDGE_LINE_WIDTH,
    color=COLOR_YELLOW
):
    x1, y1 = local_to_world(s1, t1, cx, cy, yaw)
    x2, y2 = local_to_world(s2, t2, cx, cy, yaw)

    add_line_segment(
        world,
        prefix,
        x1,
        y1,
        x2,
        y2,
        width=width,
        color=color
    )


def get_m4_geometry():
    lane_half = M4_LANE_WIDTH / 2.0
    double_half = M4_DOUBLE_WIDTH / 2.0

    s0 = -M4_TOTAL_LENGTH / 2.0
    s1 = s0 + M4_TRANSITION_LENGTH
    s2 = -s1
    s3 = -s0

    left_upper = local_to_world(s0, lane_half, M4_CX, M4_CY, M4_YAW)
    left_lower = local_to_world(s0, -lane_half, M4_CX, M4_CY, M4_YAW)
    right_upper = local_to_world(s3, lane_half, M4_CX, M4_CY, M4_YAW)
    right_lower = local_to_world(s3, -lane_half, M4_CX, M4_CY, M4_YAW)

    return {
        "lane_half": lane_half,
        "double_half": double_half,
        "s0": s0,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "left_upper": left_upper,
        "left_lower": left_lower,
        "right_upper": right_upper,
        "right_lower": right_lower,
    }


def get_left_uturn_start_points():
    outer_start = arc_point(LEFT_UTURN_CX, LEFT_UTURN_CY, LEFT_UTURN_OUTER_R, LEFT_UTURN_START_DEG)
    inner_start = arc_point(LEFT_UTURN_CX, LEFT_UTURN_CY, LEFT_UTURN_INNER_R, LEFT_UTURN_START_DEG)
    return outer_start, inner_start


# ============================================================
# Arena primitives
# ============================================================

def create_ground(world):
    add_box_model(
        world,
        "arena_ground",
        0.0,
        0.0,
        GROUND_Z,
        ARENA_LENGTH,
        ARENA_WIDTH,
        GROUND_THICKNESS,
        color=COLOR_GROUND,
        collision=True
    )


def create_top_straight(world):
    add_line_segment(world, "top_outer", TOP_X_START, TOP_Y_TOP, TOP_X_END, TOP_Y_TOP)
    add_line_segment(world, "top_inner", TOP_X_START, TOP_Y_BOTTOM, TOP_X_END, TOP_Y_BOTTOM)

    # 시작 검은 구역
    add_box_model(
        world,
        "start_zone_black",
        -6.90,
        (TOP_Y_TOP + TOP_Y_BOTTOM) / 2.0,
        LINE_Z,
        0.55,
        0.62,
        LINE_HEIGHT,
        color=COLOR_BLACK,
        collision=False
    )

    # 시작 흰색 라인 (명확히 보이게 2개)
    add_line_segment(
        world,
        "start_line_main",
        -6.28,
        TOP_Y_BOTTOM,
        -6.28,
        TOP_Y_TOP,
        width=WHITE_LINE_WIDTH,
        color=COLOR_WHITE
    )

    add_line_segment(
        world,
        "start_line_back",
        -7.05,
        TOP_Y_BOTTOM,
        -7.05,
        TOP_Y_TOP,
        width=WHITE_LINE_WIDTH,
        color=COLOR_WHITE
    )


def create_speed_zones(world):
    zone_y = (TOP_Y_TOP + TOP_Y_BOTTOM) / 2.0
    zone_width = 0.72

    add_box_model(
        world,
        "red_speed_zone",
        -4.60,
        zone_y,
        LINE_Z,
        2.75,
        zone_width,
        LINE_HEIGHT,
        color=COLOR_RED,
        collision=False
    )

    add_box_model(
        world,
        "blue_speed_zone",
        -2.35,
        zone_y,
        LINE_Z,
        1.35,
        zone_width,
        LINE_HEIGHT,
        color=COLOR_BLUE,
        collision=False
    )


def create_crosswalk(world):
    center_x = 0.65
    stripe_width = 0.06
    stripe_length = 0.68
    stripe_gap = 0.11
    count = 7
    center_y = (TOP_Y_TOP + TOP_Y_BOTTOM) / 2.0

    # 횡단보도 미션 시작선:
    # 횡단보도 진입 직전에 도로 전체 폭을 가로지르는 흰색 선
    mission_start_x = center_x - 0.60

    add_line_segment(
        world,
        "crosswalk_mission_start",
        mission_start_x,
        TOP_Y_BOTTOM,
        mission_start_x,
        TOP_Y_TOP,
        width=WHITE_LINE_WIDTH,
        color=COLOR_WHITE
    )

    # 횡단보도 줄무늬
    for i in range(count):
        x = center_x + (i - (count - 1) / 2.0) * stripe_gap

        add_box_model(
            world,
            unique_name("crosswalk"),
            x,
            center_y,
            LINE_Z,
            stripe_width,
            stripe_length,
            LINE_HEIGHT,
            color=COLOR_WHITE,
            collision=False
        )


def create_right_uturn(world):
    add_arc(
        world,
        "right_uturn_outer",
        RIGHT_UTURN_CX,
        RIGHT_UTURN_CY,
        RIGHT_UTURN_OUTER_R,
        RIGHT_UTURN_START_DEG,
        RIGHT_UTURN_END_DEG,
        segments=44
    )

    add_arc(
        world,
        "right_uturn_inner",
        RIGHT_UTURN_CX,
        RIGHT_UTURN_CY,
        RIGHT_UTURN_INNER_R,
        RIGHT_UTURN_START_DEG,
        RIGHT_UTURN_END_DEG,
        segments=36
    )

    m4 = get_m4_geometry()

    outer_end = arc_point(
        RIGHT_UTURN_CX,
        RIGHT_UTURN_CY,
        RIGHT_UTURN_OUTER_R,
        RIGHT_UTURN_END_DEG
    )

    inner_end = arc_point(
        RIGHT_UTURN_CX,
        RIGHT_UTURN_CY,
        RIGHT_UTURN_INNER_R,
        RIGHT_UTURN_END_DEG
    )

    # --------------------------------------------------------
    # 첫 번째 커브와 M4를 교차 없이 바로 연결
    #
    # RIGHT_UTURN_END_DEG = -72 deg 로 잡았기 때문에
    # 커브 끝의 접선 방향이 M4의 주행축과 거의 동일하다.
    # --------------------------------------------------------

    add_line_segment(
        world,
        "right_curve_to_m4_outer",
        outer_end[0],
        outer_end[1],
        m4["right_lower"][0],
        m4["right_lower"][1]
    )

    add_line_segment(
        world,
        "right_curve_to_m4_inner",
        inner_end[0],
        inner_end[1],
        m4["right_upper"][0],
        m4["right_upper"][1]
    )


def create_m4_lane_change(world):
    m4 = get_m4_geometry()

    s0 = m4["s0"]
    s1 = m4["s1"]
    s2 = m4["s2"]
    s3 = m4["s3"]
    lane_half = m4["lane_half"]
    double_half = m4["double_half"]

    # 위 경계
    add_local_line(world, "m4_top_1", s0, lane_half, s1, double_half, M4_CX, M4_CY, M4_YAW)
    add_local_line(world, "m4_top_2", s1, double_half, s2, double_half, M4_CX, M4_CY, M4_YAW)
    add_local_line(world, "m4_top_3", s2, double_half, s3, lane_half, M4_CX, M4_CY, M4_YAW)

    # 아래 경계
    add_local_line(world, "m4_bottom_1", s0, -lane_half, s1, -double_half, M4_CX, M4_CY, M4_YAW)
    add_local_line(world, "m4_bottom_2", s1, -double_half, s2, -double_half, M4_CX, M4_CY, M4_YAW)
    add_local_line(world, "m4_bottom_3", s2, -double_half, s3, -lane_half, M4_CX, M4_CY, M4_YAW)

    # 중앙 분리선
    add_local_line(
        world,
        "m4_center_divider",
        s1 + 0.25,
        0.0,
        s2 - 0.25,
        0.0,
        M4_CX,
        M4_CY,
        M4_YAW,
        width=WHITE_LINE_WIDTH,
        color=COLOR_WHITE
    )

    # 진입/진출 기준선
    add_local_line(
        world,
        "m4_entry_white",
        s0,
        -lane_half,
        s0,
        lane_half,
        M4_CX,
        M4_CY,
        M4_YAW,
        width=WHITE_LINE_WIDTH,
        color=COLOR_WHITE
    )

    add_local_line(
        world,
        "m4_exit_white",
        s3,
        -lane_half,
        s3,
        lane_half,
        M4_CX,
        M4_CY,
        M4_YAW,
        width=WHITE_LINE_WIDTH,
        color=COLOR_WHITE
    )


def create_roundabout(world):
    add_arc(
        world,
        "roundabout_outer",
        ROUNDABOUT_CX,
        ROUNDABOUT_CY,
        ROUNDABOUT_OUTER_R,
        0,
        360,
        segments=72
    )

    # --------------------------------------------------------
    # 중앙 원기둥
    # 로봇보다 충분히 높도록 유지
    # --------------------------------------------------------

    island_height = 0.56

    add_cylinder_model(
        world,
        "roundabout_island",
        ROUNDABOUT_CX,
        ROUNDABOUT_CY,
        island_height / 2.0,
        ROUNDABOUT_ISLAND_R,
        island_height,
        COLOR_GREY,
        collision=True
    )

    add_arc(
        world,
        "roundabout_island_edge",
        ROUNDABOUT_CX,
        ROUNDABOUT_CY,
        ROUNDABOUT_ISLAND_R,
        0,
        360,
        segments=48,
        width=EDGE_LINE_WIDTH,
        color=COLOR_YELLOW
    )

    m4 = get_m4_geometry()

    # --------------------------------------------------------
    # M4 -> roundabout 진행방향
    #
    # M4는 좌 -> 우 기준으로 +18 deg.
    # 실제 이 구간 주행은 우 -> 좌이므로 +180 deg.
    # --------------------------------------------------------

    road_yaw = M4_YAW + math.pi

    dx = math.cos(road_yaw)
    dy = math.sin(road_yaw)

    # --------------------------------------------------------
    # ray와 원의 첫 번째 교점을 구하는 함수
    # --------------------------------------------------------

    def ray_circle_first_hit(px, py):
        ox = px - ROUNDABOUT_CX
        oy = py - ROUNDABOUT_CY

        b = 2.0 * (ox * dx + oy * dy)

        c = (
            ox * ox
            + oy * oy
            - ROUNDABOUT_OUTER_R * ROUNDABOUT_OUTER_R
        )

        disc = b * b - 4.0 * c

        if disc < 0:
            return None

        root = math.sqrt(disc)

        t1 = (-b - root) / 2.0
        t2 = (-b + root) / 2.0

        positive = [
            t for t in [t1, t2]
            if t >= 0
        ]

        if not positive:
            return None

        t = min(positive)

        return (
            px + t * dx,
            py + t * dy,
            t
        )

    upper_start = m4["left_upper"]
    lower_start = m4["left_lower"]

    upper_hit = ray_circle_first_hit(
        upper_start[0],
        upper_start[1]
    )

    lower_hit = ray_circle_first_hit(
        lower_start[0],
        lower_start[1]
    )

    if upper_hit is None or lower_hit is None:
        raise RuntimeError(
            "M4 -> roundabout 연결선이 원과 만나지 않습니다."
        )

    # 두 차선을 원까지 정확히 연장
    add_line_segment(
        world,
        "m4_to_roundabout_upper",
        upper_start[0],
        upper_start[1],
        upper_hit[0],
        upper_hit[1]
    )

    add_line_segment(
        world,
        "m4_to_roundabout_lower",
        lower_start[0],
        lower_start[1],
        lower_hit[0],
        lower_hit[1]
    )

    # --------------------------------------------------------
    # 회전교차로 미션 시작선
    #
    # 두 차선의 동일한 진행거리 지점을 연결하면
    # M4 종료선과 정확히 평행하다.
    # --------------------------------------------------------

    marker_t = min(
        upper_hit[2],
        lower_hit[2]
    ) * 0.56

    su = (
        upper_start[0] + marker_t * dx,
        upper_start[1] + marker_t * dy
    )

    sl = (
        lower_start[0] + marker_t * dx,
        lower_start[1] + marker_t * dy
    )

    add_line_segment(
        world,
        "roundabout_mission_start",
        sl[0],
        sl[1],
        su[0],
        su[1],
        width=WHITE_LINE_WIDTH,
        color=COLOR_WHITE
    )


def create_roundabout_to_left(world):
    outer_start, inner_start = get_left_uturn_start_points()

    # roundabout 전후 직선이 서로 평행하도록
    # 같은 축(M4_YAW 계열)을 사용한다.
    axis_yaw = M4_YAW
    dx = math.cos(axis_yaw)
    dy = math.sin(axis_yaw)

    def ray_circle_first_hit(px, py):
        ox = px - ROUNDABOUT_CX
        oy = py - ROUNDABOUT_CY

        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - ROUNDABOUT_OUTER_R * ROUNDABOUT_OUTER_R

        disc = b * b - 4.0 * c
        if disc < 0:
            return None

        root = math.sqrt(disc)
        t1 = (-b - root) / 2.0
        t2 = (-b + root) / 2.0

        positive = [t for t in [t1, t2] if t >= 0]
        if not positive:
            return None

        t = min(positive)
        return (px + t * dx, py + t * dy, t)

    outer_hit = ray_circle_first_hit(outer_start[0], outer_start[1])
    inner_hit = ray_circle_first_hit(inner_start[0], inner_start[1])

    if outer_hit is None or inner_hit is None:
        raise RuntimeError("roundabout -> left curve 연결 실패")

    # 회전교차로 종료 직선 2개
    add_line_segment(
        world,
        "roundabout_to_left_outer",
        outer_hit[0], outer_hit[1],
        outer_start[0], outer_start[1]
    )

    add_line_segment(
        world,
        "roundabout_to_left_inner",
        inner_hit[0], inner_hit[1],
        inner_start[0], inner_start[1]
    )

    # 종료선: M4 종료선과 평행
    marker_t = min(outer_hit[2], inner_hit[2]) * 0.64

    end_outer = (
        outer_start[0] + marker_t * dx,
        outer_start[1] + marker_t * dy
    )
    end_inner = (
        inner_start[0] + marker_t * dx,
        inner_start[1] + marker_t * dy
    )

    add_line_segment(
        world,
        "roundabout_mission_end",
        end_inner[0], end_inner[1],
        end_outer[0], end_outer[1],
        width=WHITE_LINE_WIDTH,
        color=COLOR_WHITE
    )


def create_left_uturn(world):
    add_arc(
        world,
        "left_uturn_outer",
        LEFT_UTURN_CX,
        LEFT_UTURN_CY,
        LEFT_UTURN_OUTER_R,
        LEFT_UTURN_START_DEG,
        LEFT_UTURN_END_DEG,
        segments=36
    )

    add_arc(
        world,
        "left_uturn_inner",
        LEFT_UTURN_CX,
        LEFT_UTURN_CY,
        LEFT_UTURN_INNER_R,
        LEFT_UTURN_START_DEG,
        LEFT_UTURN_END_DEG,
        segments=30
    )


def create_bottom_straight(world):
    add_line_segment(world, "bottom_upper", BOTTOM_X_START, BOTTOM_Y_TOP, 6.15, BOTTOM_Y_TOP)
    add_line_segment(world, "bottom_lower", BOTTOM_X_START, BOTTOM_Y_BOTTOM, BOTTOM_X_END + 0.50, BOTTOM_Y_BOTTOM)

    # roundabout 종료부 기준선만 유지
    for x in [-0.35]:
        add_line_segment(
            world,
            "bottom_white",
            x,
            BOTTOM_Y_BOTTOM,
            x,
            BOTTOM_Y_TOP,
            width=WHITE_LINE_WIDTH,
            color=COLOR_WHITE
        )


def create_tunnel(world):
    center_x = -2.55
    center_y = -3.55

    tunnel_length = 2.00
    tunnel_width = 1.08

    wall_thickness = 0.12
    wall_height = 0.65
    roof_height = 0.10

    side_offset = (
        tunnel_width / 2.0
        - wall_thickness / 2.0
    )

    # --------------------------------------------------------
    # 위쪽 측벽
    # X 방향으로 길게
    # --------------------------------------------------------

    add_box_model(
        world,
        "tunnel_side_wall_upper",
        center_x,
        center_y + side_offset,
        wall_height / 2.0,
        tunnel_length,
        wall_thickness,
        wall_height,
        color=COLOR_GREY,
        collision=True
    )

    # --------------------------------------------------------
    # 아래쪽 측벽
    # --------------------------------------------------------

    add_box_model(
        world,
        "tunnel_side_wall_lower",
        center_x,
        center_y - side_offset,
        wall_height / 2.0,
        tunnel_length,
        wall_thickness,
        wall_height,
        color=COLOR_GREY,
        collision=True
    )

    # --------------------------------------------------------
    # 지붕
    #
    # 입구와 출구는 완전히 개방됨
    # --------------------------------------------------------

    add_box_model(
        world,
        "tunnel_roof",
        center_x,
        center_y,
        wall_height + roof_height / 2.0,
        tunnel_length,
        tunnel_width,
        roof_height,
        color=COLOR_DARK_GREY,
        collision=True
    )


def create_barrier(world):
    arm_center_x = 3.55
    arm_center_y = (
        BOTTOM_Y_TOP + BOTTOM_Y_BOTTOM
    ) / 2.0

    # --------------------------------------------------------
    # 차단봉
    #
    # box:
    # length = X 방향
    # width  = Y 방향
    #
    # 차량 진행방향 X축에 수직이 되도록
    # Y축 길이를 1.10 m로 증가.
    # --------------------------------------------------------

    add_box_model(
        world,
        "moving_barrier_arm",
        arm_center_x,
        arm_center_y,
        0.43,
        0.07,
        1.10,
        0.07,
        color=COLOR_YELLOW,
        collision=True
    )

    # --------------------------------------------------------
    # 검은색 지지 봉
    # LiDAR 인식을 위해 기존보다 약간 크게
    # --------------------------------------------------------

    add_cylinder_model(
        world,
        "barrier_post",
        arm_center_x,
        BOTTOM_Y_TOP + 0.03,
        0.35,
        0.055,
        0.70,
        COLOR_BLACK,
        collision=True
    )




def create_parking_zone(world):
    """
    주차구역 모양:
    - 꺾인 형태 유지
    - 주차구역 입구에는 선 없음
    """

    # 아래 메인 차선의 위쪽 경계는 이미 bottom_upper 로 존재
    # 따라서 여기서는 주차공간 윤곽만 만든다.

    points = [
        # 입구 왼쪽 시작점 (메인 차선 위쪽 경계에서 바로 시작)
        (6.20, BOTTOM_Y_TOP),

        # 위로 올라가는 첫 번째 꺾임
        (6.20, -2.35),

        # 상단 수평
        (7.18, -2.35),

        # 다시 아래로
        (7.18, BOTTOM_Y_TOP),

        # 오른쪽으로 짧게
        (7.62, BOTTOM_Y_TOP),

        # 우측 세로
        (7.62, BOTTOM_Y_BOTTOM),
    ]

    add_polyline(
        world,
        "parking_shape",
        points
    )

    # 중요:
    # 주차구역 입구를 막는 세로선/흰색선은 생성하지 않음.



def add_png_sign_panel(
    world,
    name,
    x,
    y,
    z,
    width,
    height,
    yaw,
    material_name,
    thickness=0.015,
    collision=False
):
    """
    Gazebo Classic용 PNG texture 표지판 panel.

    - local X축 = 표지판 두께 방향
    - local Y축 = 표지판 가로
    - local Z축 = 표지판 세로
    """

    model = ET.SubElement(
        world,
        "model",
        {"name": name}
    )

    ET.SubElement(model, "static").text = "true"

    ET.SubElement(
        model,
        "pose"
    ).text = f"{x} {y} {z} 0 0 {yaw}"

    link = ET.SubElement(
        model,
        "link",
        {"name": "link"}
    )

    visual = ET.SubElement(
        link,
        "visual",
        {"name": "visual"}
    )

    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box")

    ET.SubElement(
        box,
        "size"
    ).text = f"{thickness} {width} {height}"

    material = ET.SubElement(visual, "material")
    script = ET.SubElement(material, "script")

    # make_arena.py가 source에서 실행되면 source/media,
    # install에서 실행되면 install/.../media를 자동 사용
    package_dir = Path(__file__).resolve().parent.parent

    script_dir = (
        package_dir
        / "media"
        / "materials"
        / "scripts"
    )

    texture_dir = (
        package_dir
        / "media"
        / "materials"
        / "textures"
    )

    ET.SubElement(
        script,
        "uri"
    ).text = script_dir.as_uri()

    ET.SubElement(
        script,
        "uri"
    ).text = texture_dir.as_uri()

    ET.SubElement(
        script,
        "name"
    ).text = material_name

    if collision:
        collision_el = ET.SubElement(
            link,
            "collision",
            {"name": "collision"}
        )

        geometry = ET.SubElement(
            collision_el,
            "geometry"
        )

        box = ET.SubElement(
            geometry,
            "box"
        )

        ET.SubElement(
            box,
            "size"
        ).text = f"{thickness} {width} {height}"


def create_parking_sign(world):
    """
    PNG 기반 주차 표지판.

    하단 직선에서 좌 -> 우로 접근하는 LIMO가
    정면에서 볼 수 있도록 -X 방향을 향한다.
    """

    # ========================================================
    # 위치
    # ========================================================
    x = 6.02
    y = -3.00

    # ========================================================
    # 여기 숫자들만 바꾸면 나중에 크기/높이 조절 가능
    # ========================================================

    # 표지판 중심 높이
    panel_z = 0.43

    # 표지판 PNG 패널 크기
    panel_width = 0.36
    panel_height = 0.36

    # 표지판 두께
    panel_thickness = 0.018

    # 봉 굵기
    post_width = 0.035

    # 표지판 아래까지 봉이 올라오도록 계산
    panel_bottom_z = panel_z - panel_height / 2.0

    post_height = panel_bottom_z

    # ========================================================
    # 봉
    # ========================================================

    add_box_model(
        world,
        "parking_sign_post",
        x,
        y,
        post_height / 2.0,
        post_width,
        post_width,
        post_height,
        color=COLOR_GREY,
        collision=True
    )

    # ========================================================
    # PNG 표지판
    #
    # yaw = pi
    # -> 표지판 앞면이 -X 방향을 바라봄
    # ========================================================

    add_png_sign_panel(
        world,
        "parking_sign_panel",
        x,
        y,
        panel_z,
        panel_width,
        panel_height,
        math.pi,
        "KUDOS/Parking",
        thickness=panel_thickness,
        collision=False
    )


def create_cone(world, x, y):
    """
    단일 mesh로 구성된 주황색 고깔형 라바콘.

    높이: 0.34 m
    밑반지름: 0.10 m
    흰색 띠 없음
    """

    model = ET.SubElement(
        world,
        "model",
        {"name": unique_name("cone")}
    )

    ET.SubElement(
        model,
        "static"
    ).text = "true"

    # STL 자체가 z=0부터 시작하므로
    # 그대로 지면 위에 배치
    ET.SubElement(
        model,
        "pose"
    ).text = pose_text(
        x,
        y,
        0.0
    )

    link = ET.SubElement(
        model,
        "link",
        {"name": "link"}
    )

    mesh_path = (
        PACKAGE_DIR
        / "meshes"
        / "traffic_cone.stl"
    )

    mesh_uri = mesh_path.as_uri()

    # --------------------------------------------------------
    # collision
    # --------------------------------------------------------

    collision = ET.SubElement(
        link,
        "collision",
        {"name": "collision"}
    )

    geometry = ET.SubElement(
        collision,
        "geometry"
    )

    mesh = ET.SubElement(
        geometry,
        "mesh"
    )

    ET.SubElement(
        mesh,
        "uri"
    ).text = mesh_uri

    # --------------------------------------------------------
    # visual
    # --------------------------------------------------------

    visual = ET.SubElement(
        link,
        "visual",
        {"name": "visual"}
    )

    geometry = ET.SubElement(
        visual,
        "geometry"
    )

    mesh = ET.SubElement(
        geometry,
        "mesh"
    )

    ET.SubElement(
        mesh,
        "uri"
    ).text = mesh_uri

    # 전체 주황색
    add_material(
        visual,
        COLOR_ORANGE
    )

def create_curve_cones(world):

    # ========================================================
    # 첫 번째 커브 - 우측 상단
    # ========================================================

    # 커브 양쪽에 라바콘 배치
    #
    # inner cone radius:
    #   안쪽 차선 경계보다 약간 바깥
    #
    # outer cone radius:
    #   바깥 차선 경계보다 약간 안쪽

    right_inner_cone_r = RIGHT_UTURN_INNER_R + 0.10
    right_outer_cone_r = RIGHT_UTURN_OUTER_R - 0.10

    right_angles = [
        70,
        40,
        10,
        -20,
        -50,
        -80,
        -110
    ]

    for deg in right_angles:
        # 안쪽
        x, y = arc_point(
            RIGHT_UTURN_CX,
            RIGHT_UTURN_CY,
            right_inner_cone_r,
            deg
        )
        create_cone(world, x, y)

        # 바깥쪽
        x, y = arc_point(
            RIGHT_UTURN_CX,
            RIGHT_UTURN_CY,
            right_outer_cone_r,
            deg
        )
        create_cone(world, x, y)

    # ========================================================
    # 두 번째 커브 - 좌측 하단
    # ========================================================

    left_inner_cone_r = LEFT_UTURN_INNER_R + 0.10
    left_outer_cone_r = LEFT_UTURN_OUTER_R - 0.10

    left_angles = [
        160,
        185,
        210,
        235,
        260
    ]

    for deg in left_angles:
        # 안쪽
        x, y = arc_point(
            LEFT_UTURN_CX,
            LEFT_UTURN_CY,
            left_inner_cone_r,
            deg
        )
        create_cone(world, x, y)

        # 바깥쪽
        x, y = arc_point(
            LEFT_UTURN_CX,
            LEFT_UTURN_CY,
            left_outer_cone_r,
            deg
        )
        create_cone(world, x, y)





def add_circular_png_sign(
    world,
    name,
    x,
    y,
    z,
    radius,
    yaw,
    material_name,
    cylinder_thickness=0.025
):
    """
    원형 표지판.

    구조:
      1) 얇은 파란 원기둥 = 실제 표지판 몸체
      2) 원기둥 앞에 투명 PNG panel = 화살표 그림
    """

    # ========================================================
    # 원형 본체
    # ========================================================

    model = ET.SubElement(
        world,
        "model",
        {"name": f"{name}_disc"}
    )

    ET.SubElement(model, "static").text = "true"

    # cylinder 기본 축은 Z축.
    # pitch=pi/2로 세워서 원형 면이 수직이 되게 함.
    ET.SubElement(
        model,
        "pose"
    ).text = (
        f"{x} {y} {z} "
        f"0 {math.pi / 2.0} {yaw}"
    )

    link = ET.SubElement(
        model,
        "link",
        {"name": "link"}
    )

    visual = ET.SubElement(
        link,
        "visual",
        {"name": "visual"}
    )

    geometry = ET.SubElement(visual, "geometry")

    cylinder = ET.SubElement(
        geometry,
        "cylinder"
    )

    ET.SubElement(
        cylinder,
        "radius"
    ).text = str(radius)

    ET.SubElement(
        cylinder,
        "length"
    ).text = str(cylinder_thickness)

    # 파란 원형 본체
    material = ET.SubElement(
        visual,
        "material"
    )

    script = ET.SubElement(
        material,
        "script"
    )

    ET.SubElement(
        script,
        "uri"
    ).text = "file://media/materials/scripts/gazebo.material"

    ET.SubElement(
        script,
        "name"
    ).text = "Gazebo/Blue"


    # ========================================================
    # PNG 화살표
    # ========================================================

    # 표지판 앞면 방향 벡터
    nx = math.cos(yaw)
    ny = math.sin(yaw)

    # 원기둥 앞면보다 2 mm 앞으로 빼서
    # z-fighting 방지
    offset = cylinder_thickness / 2.0 + 0.002

    px = x + nx * offset
    py = y + ny * offset

    diameter = radius * 2.0

    add_png_sign_panel(
        world,
        f"{name}_texture",
        px,
        py,
        z,
        diameter,
        diameter,
        yaw,
        material_name,
        thickness=0.002,
        collision=False
    )


def create_m4_direction_signs(world):
    """
    M4 차선변경 원형 PNG 표지판.
    """

    m4 = get_m4_geometry()

    start_dir, end_dir = resolve_m4_sign_dirs()

    sign_end_s = m4["s1"] + 0.06
    sign_start_s = m4["s2"] - 0.06

    lateral_offset = 0.00

    x_start, y_start = local_to_world(
        sign_start_s,
        lateral_offset,
        M4_CX,
        M4_CY,
        M4_YAW
    )

    x_end, y_end = local_to_world(
        sign_end_s,
        lateral_offset,
        M4_CX,
        M4_CY,
        M4_YAW
    )

    # ========================================================
    # 여기 숫자들로 나중에 위치/크기/높이 조정
    # ========================================================

    # 표지판 중심 높이
    sign_z = 0.27

    # 원형 표지판 반지름
    sign_radius = 0.115

    # 원형 판 두께
    sign_thickness = 0.025

    # 봉 굵기
    post_width = 0.025

    # 기존 위치에서 global +X 방향으로 약간 이동
    x_offset = 0.03

    x_start += x_offset
    x_end += x_offset

    # 원 아래까지 봉이 연결되도록 자동 계산
    sign_bottom_z = sign_z - sign_radius

    post_height = max(
        sign_bottom_z,
        0.08
    )

    # ========================================================
    # 표지판 앞면 방향
    #
    # 이전 표지판이 뒤집혀 있었으므로
    # 접근 차량을 바라보는 쪽으로 변경.
    # ========================================================

    face_yaw = M4_YAW


    def add_m4_sign(
        name,
        x,
        y,
        direction
    ):

        # ----------------------------------------------------
        # pole
        # ----------------------------------------------------

        add_box_model(
            world,
            f"{name}_post",
            x,
            y,
            post_height / 2.0,
            post_width,
            post_width,
            post_height,
            color=COLOR_GREY,
            collision=True
        )

        # ----------------------------------------------------
        # PNG material 결정
        # ----------------------------------------------------

        if direction == "left":
            material_name = "KUDOS/M4_Left"
        else:
            material_name = "KUDOS/M4_Right"

        # ----------------------------------------------------
        # 원형 표지판
        # ----------------------------------------------------

        add_circular_png_sign(
            world,
            name,
            x,
            y,
            sign_z,
            sign_radius,
            face_yaw,
            material_name,
            cylinder_thickness=sign_thickness
        )


    add_m4_sign(
        "m4_sign_start",
        x_start,
        y_start,
        start_dir
    )

    add_m4_sign(
        "m4_sign_end",
        x_end,
        y_end,
        end_dir
    )

    print(
        f"[M4 circular PNG signs] "
        f"start={start_dir}, end={end_dir}"
    )

def add_world_environment(world):

    # --------------------------------------------------------
    # gravity
    # --------------------------------------------------------

    gravity = ET.SubElement(world, "gravity")
    gravity.text = "0 0 -9.81"

    # --------------------------------------------------------
    # physics
    # --------------------------------------------------------

    physics = ET.SubElement(
        world,
        "physics",
        {
            "name": "default_physics",
            "type": "ode"
        }
    )

    max_step_size = ET.SubElement(
        physics,
        "max_step_size"
    )
    max_step_size.text = "0.001"

    real_time_factor = ET.SubElement(
        physics,
        "real_time_factor"
    )
    real_time_factor.text = "1.0"

    # --------------------------------------------------------
    # sun
    # --------------------------------------------------------

    include = ET.SubElement(world, "include")

    uri = ET.SubElement(include, "uri")
    uri.text = "model://sun"

    # --------------------------------------------------------
    # scene
    # --------------------------------------------------------

    scene = ET.SubElement(world, "scene")

    ambient = ET.SubElement(scene, "ambient")
    ambient.text = "0.5 0.5 0.5 1"

    background = ET.SubElement(scene, "background")
    background.text = "0.72 0.72 0.72 1"

    shadows = ET.SubElement(scene, "shadows")
    shadows.text = "true"

    # --------------------------------------------------------
    # ROS2 Gazebo state plugin
    #
    # /gazebo/set_entity_state
    # /gazebo/get_entity_state
    #
    # 서비스를 제공한다.
    # --------------------------------------------------------

    plugin = ET.SubElement(
        world,
        "plugin",
        {
            "name": "gazebo_ros_state",
            "filename": "libgazebo_ros_state.so"
        }
    )

    # ROS namespace
    ros = ET.SubElement(plugin, "ros")

    namespace = ET.SubElement(ros, "namespace")
    namespace.text = "/gazebo"



def create_roundabout_moving_vehicle(world):
    """회전교차로를 도는 LIMO 크기의 차량형 동적 장애물 2대."""

    radius = 0.68

    vehicles = [
        (
            "roundabout_dynamic_obstacle_1",
            ROUNDABOUT_CX + radius,
            ROUNDABOUT_CY,
            0.0
        ),
        (
            "roundabout_dynamic_obstacle_2",
            ROUNDABOUT_CX - radius,
            ROUNDABOUT_CY,
            math.pi
        ),
    ]

    for model_name, start_x, start_y, start_yaw in vehicles:

        model = ET.SubElement(
            world,
            "model",
            {"name": model_name}
        )

        ET.SubElement(
            model,
            "pose"
        ).text = (
            f"{start_x} {start_y} 0 "
            f"0 0 {start_yaw}"
        )

        ET.SubElement(model, "static").text = "false"

        link = ET.SubElement(
            model,
            "link",
            {"name": "vehicle_link"}
        )

        # ROS2 controller가 직접 위치를 갱신
        ET.SubElement(link, "gravity").text = "false"
        ET.SubElement(link, "kinematic").text = "true"

        # =========================
        # 차체
        # =========================
        body = ET.SubElement(
            link,
            "visual",
            {"name": "body_visual"}
        )

        ET.SubElement(
            body,
            "pose"
        ).text = "0 0 0.125 0 0 0"

        geometry = ET.SubElement(body, "geometry")
        box = ET.SubElement(geometry, "box")

        # LIMO와 비슷한 작은 차량 크기
        ET.SubElement(
            box,
            "size"
        ).text = "0.32 0.20 0.18"

        material = ET.SubElement(body, "material")
        script = ET.SubElement(material, "script")

        ET.SubElement(
            script,
            "uri"
        ).text = "file://media/materials/scripts/gazebo.material"

        ET.SubElement(
            script,
            "name"
        ).text = "Gazebo/Orange"

        # collision
        collision = ET.SubElement(
            link,
            "collision",
            {"name": "body_collision"}
        )

        ET.SubElement(
            collision,
            "pose"
        ).text = "0 0 0.125 0 0 0"

        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")

        ET.SubElement(
            box,
            "size"
        ).text = "0.32 0.20 0.18"

        # =========================
        # 바퀴 4개
        # =========================
        wheel_positions = [
            ("front_left",   0.105,  0.115),
            ("front_right",  0.105, -0.115),
            ("rear_left",   -0.105,  0.115),
            ("rear_right",  -0.105, -0.115),
        ]

        for name, x, y in wheel_positions:

            wheel = ET.SubElement(
                link,
                "visual",
                {"name": f"{name}_wheel"}
            )

            ET.SubElement(
                wheel,
                "pose"
            ).text = f"{x} {y} 0.055 1.5708 0 0"

            geometry = ET.SubElement(
                wheel,
                "geometry"
            )

            cylinder = ET.SubElement(
                geometry,
                "cylinder"
            )

            ET.SubElement(
                cylinder,
                "radius"
            ).text = "0.045"

            ET.SubElement(
                cylinder,
                "length"
            ).text = "0.032"

            material = ET.SubElement(
                wheel,
                "material"
            )

            script = ET.SubElement(
                material,
                "script"
            )

            ET.SubElement(
                script,
                "uri"
            ).text = "file://media/materials/scripts/gazebo.material"

            ET.SubElement(
                script,
                "name"
            ).text = "Gazebo/Black"


def apply_final_geometry_fixes(world):
    """
    최종 경기장 위치/주차구역 보정.

    1. 첫 번째 커브의 가장 오른쪽 라바콘을 +X 방향으로 이동
    2. bottom_upper_1을 줄여 주차구역 입구 확보
    3. parking_shape_1, parking_shape_3 길이 증가
    4. parking_shape_1~3 위치를 따라 주차 벽 3면 생성
    """

    # ========================================================
    # 여기 숫자들만 수정하면 나중에 미세조정 가능
    # ========================================================

    # 문제의 라바콘을 오른쪽(+X)으로 이동
    CONE_X_SHIFT = 0.25

    # bottom_upper_1 길이를 이만큼 감소
    BOTTOM_UPPER_SHRINK = 0.0

    # parking_shape_1 / 3 길이를 이만큼 증가
    PARKING_SIDE_EXTEND = 0.0

    # 주차 벽
    PARKING_WALL_HEIGHT = 0.55
    PARKING_WALL_THICKNESS = 0.08


    # ========================================================
    # helper
    # ========================================================

    def get_model(name):
        for model in world.findall("model"):
            if model.attrib.get("name") == name:
                return model

        return None


    def get_pose(model):
        pose = model.find("pose")

        if pose is None or not pose.text:
            return None

        values = [float(v) for v in pose.text.split()]

        while len(values) < 6:
            values.append(0.0)

        return values


    def set_pose(model, values):
        model.find("pose").text = " ".join(
            str(v) for v in values
        )


    def get_box_size(model):
        size = model.find(".//box/size")

        if size is None or not size.text:
            return None, None

        values = [float(v) for v in size.text.split()]

        return size, values


    # ========================================================
    # 1. cone 14를 정확히 지정해서 +X 방향으로 이동
    #
    # 이전처럼 '가장 오른쪽 cone'을 자동 선택하지 않는다.
    # cone 14의 base / body / stripe 등이 별도 model이면
    # 모두 같은 거리만큼 이동한다.
    # ========================================================

    import re

    CONE_TARGET_NUMBER = 14
    CONE_X_SHIFT = 0.25

    cone_pattern = re.compile(
        rf'(^|[^0-9]){CONE_TARGET_NUMBER}([^0-9]|$)'
    )

    moved_cone_parts = []

    for model in world.findall("model"):

        name = model.attrib.get("name", "")
        lower_name = name.lower()

        if "cone" not in lower_name:
            continue

        # 이름 안에 독립된 숫자 14가 있는 cone만 선택
        if not cone_pattern.search(lower_name):
            continue

        pose = get_pose(model)

        if pose is None:
            continue

        # Gazebo global +X = 빨간 축 방향
        pose[0] += CONE_X_SHIFT

        set_pose(
            model,
            pose
        )

        moved_cone_parts.append(name)

    if moved_cone_parts:

        print(
            f"[FIX] cone {CONE_TARGET_NUMBER} "
            f"+X {CONE_X_SHIFT:.2f} m"
        )

        for name in moved_cone_parts:
            print("      ->", name)

    else:

        print(
            f"[WARN] cone {CONE_TARGET_NUMBER} "
            "관련 model을 찾지 못했습니다."
        )


    # ========================================================
    # 2. bottom_upper_1 줄이기
    #
    # 주차구역 입구와 가까운 쪽 끝을 줄이고
    # 반대쪽 끝은 그대로 유지한다.
    # ========================================================

    model = get_model("bottom_upper_1")

    if model is not None:

        pose = get_pose(model)
        size_el, size = get_box_size(model)

        if pose is not None and size is not None:

            x, y, z, roll, pitch, yaw = pose

            # X/Y 중 긴 쪽이 line 진행방향
            if size[0] >= size[1]:

                old_length = size[0]

                new_length = max(
                    0.10,
                    old_length - BOTTOM_UPPER_SHRINK
                )

                # local X axis
                ux = math.cos(yaw)
                uy = math.sin(yaw)

                long_axis_index = 0

            else:

                old_length = size[1]

                new_length = max(
                    0.10,
                    old_length - BOTTOM_UPPER_SHRINK
                )

                # local Y axis
                ux = -math.sin(yaw)
                uy = math.cos(yaw)

                long_axis_index = 1


            # 기존 양 끝점
            p1 = (
                x + ux * old_length / 2.0,
                y + uy * old_length / 2.0
            )

            p2 = (
                x - ux * old_length / 2.0,
                y - uy * old_length / 2.0
            )

            # 주차 입구 근처 좌표
            entrance_x = 6.02
            entrance_y = -3.00

            d1 = math.hypot(
                p1[0] - entrance_x,
                p1[1] - entrance_y
            )

            d2 = math.hypot(
                p2[0] - entrance_x,
                p2[1] - entrance_y
            )

            # 입구에서 먼 endpoint를 고정
            fixed = p1 if d1 > d2 else p2

            # fixed endpoint 기준 새로운 center 계산
            if fixed == p1:

                new_x = fixed[0] - ux * new_length / 2.0
                new_y = fixed[1] - uy * new_length / 2.0

            else:

                new_x = fixed[0] + ux * new_length / 2.0
                new_y = fixed[1] + uy * new_length / 2.0


            pose[0] = new_x
            pose[1] = new_y

            size[long_axis_index] = new_length

            set_pose(model, pose)

            size_el.text = " ".join(
                str(v) for v in size
            )

            print(
                f"[FIX] bottom_upper_1 "
                f"{old_length:.2f} -> "
                f"{new_length:.2f}"
            )

    else:
        print(
            "[WARN] bottom_upper_1 not found"
        )


    # ========================================================
    # 3. parking_shape_1 / parking_shape_3 길이 증가
    # ========================================================

    for model_name in [
        "parking_shape_1",
        "parking_shape_3"
    ]:

        model = get_model(model_name)

        if model is None:
            print(
                f"[WARN] {model_name} not found"
            )
            continue

        size_el, size = get_box_size(model)

        if size is None:
            continue

        # 평면상의 긴 축만 늘림
        if size[0] >= size[1]:

            old_length = size[0]
            size[0] += PARKING_SIDE_EXTEND
            new_length = size[0]

        else:

            old_length = size[1]
            size[1] += PARKING_SIDE_EXTEND
            new_length = size[1]

        size_el.text = " ".join(
            str(v) for v in size
        )

        print(
            f"[FIX] {model_name} "
            f"{old_length:.2f} -> "
            f"{new_length:.2f}"
        )


    # ========================================================
    # 4. 주차구역 벽 3면
    #
    # parking_shape_1/2/3의 위치와 방향을 그대로 이용해서
    # 그 위에 높은 wall을 세운다.
    # ========================================================

    for index, model_name in enumerate(
        [
            "parking_shape_1",
            "parking_shape_2",
            "parking_shape_3"
        ],
        start=1
    ):

        source_model = get_model(
            model_name
        )

        if source_model is None:
            print(
                f"[WARN] {model_name} missing, wall skipped"
            )
            continue

        pose = get_pose(source_model)
        _, size = get_box_size(source_model)

        if pose is None or size is None:
            continue

        x = pose[0]
        y = pose[1]
        yaw = pose[5]

        # 기존 yellow line의 긴 방향은 유지하고
        # 짧은 방향만 wall 두께로 변경
        if size[0] >= size[1]:

            wall_x = size[0]
            wall_y = PARKING_WALL_THICKNESS

        else:

            wall_x = PARKING_WALL_THICKNESS
            wall_y = size[1]

        add_box_model(
            world,
            f"parking_wall_{index}",
            x,
            y,
            PARKING_WALL_HEIGHT / 2.0,
            wall_x,
            wall_y,
            PARKING_WALL_HEIGHT,
            yaw=yaw,
            color=COLOR_GREY,
            collision=True
        )

    print(
        f"[FIX] parking walls created: "
        f"height={PARKING_WALL_HEIGHT:.2f} m"
    )

def build_world():
    sdf = ET.Element("sdf", {"version": "1.6"})
    world = ET.SubElement(sdf, "world", {"name": "limo_competition"})

    add_world_environment(world)

    create_ground(world)

    create_top_straight(world)
    create_speed_zones(world)
    create_crosswalk(world)

    create_right_uturn(world)
    create_m4_lane_change(world)
    create_m4_direction_signs(world)

    create_roundabout(world)
    create_roundabout_moving_vehicle(world)
    create_roundabout_to_left(world)
    create_left_uturn(world)

    create_bottom_straight(world)
    create_tunnel(world)
    create_curve_cones(world)

    create_barrier(world)

    create_parking_zone(world)
    create_parking_sign(world)

    # 최종 경기장 위치/주차구역 보정
    apply_final_geometry_fixes(world)

    return sdf


def prettify_xml(element):
    rough = ET.tostring(element, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding="utf-8")


def main():
    PACKAGE_WORLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROOT_WORLD_PATH.parent.mkdir(parents=True, exist_ok=True)

    sdf = build_world()
    xml_data = prettify_xml(sdf)

    PACKAGE_WORLD_PATH.write_bytes(xml_data)
    ROOT_WORLD_PATH.write_bytes(xml_data)

    print("")
    print("============================================")
    print(" LIMO Competition Arena 생성 완료")
    print("============================================")
    print("")
    print("Saved world files:")
    print(f"  1) {PACKAGE_WORLD_PATH}")
    print(f"  2) {ROOT_WORLD_PATH}")
    print("")
    print("실행:")
    print("  cd ~/Workspace/limo_competition_ws")
    print("  source /opt/ros/humble/setup.bash")
    print("  source install/setup.bash")
    print("  gzserver worlds/limo_competition.world --verbose")
    print("")
    print("다른 터미널:")
    print("  source /opt/ros/humble/setup.bash")
    print("  gzclient")
    print("")


if __name__ == "__main__":
    main()


