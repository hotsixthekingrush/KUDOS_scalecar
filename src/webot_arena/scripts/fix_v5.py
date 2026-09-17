from pathlib import Path
import re

path = Path("make_arena.py")
text = path.read_text(encoding="utf-8")


# ============================================================
# 1) FIRST CURVE -> ROUNDABOUT
#
# 기존:
#   좁은 1차선 + 잘못된 사각형
#
# 수정:
#   2차선 도로
#   = 위쪽 외곽선
#   = 가운데 차선
#   = 아래쪽 외곽선
#
# 파란 표지판은 가운데 차선 위에 설치
# ============================================================

new_lane_change = r'''
# ============================================================
# FIRST CURVE -> ROUNDABOUT
# TWO-LANE LANE CHANGE SECTION
# ============================================================

# road center region
LC_LEFT = -0.05
LC_RIGHT = 1.75

LC_TOP = 1.78
LC_CENTER = 1.10
LC_BOTTOM = 0.42


# ------------------------------------------------------------
# 2-LANE ROAD
# ------------------------------------------------------------

# upper boundary
line(
    "lanechange_upper",
    LC_LEFT,
    LC_TOP,
    LC_RIGHT,
    LC_TOP,
    0.055,
    WHITE
)

# CENTER DIVIDER
#
# This is the line separating the two lanes.
#
line(
    "lanechange_center",
    LC_LEFT,
    LC_CENTER,
    LC_RIGHT,
    LC_CENTER,
    0.055,
    WHITE
)

# lower boundary
line(
    "lanechange_lower",
    LC_LEFT,
    LC_BOTTOM,
    LC_RIGHT,
    LC_BOTTOM,
    0.055,
    WHITE
)


# ============================================================
# BLUE RIGHT-DIRECTION SIGN
#
# IMPORTANT:
# sign is placed ON the CENTER DIVIDER.
#
# Vehicle approaches from the right-side curve and sees
# the FRONT FACE of the sign.
#
# Therefore the sign board is thin in X and wide in Y.
# ============================================================

SIGN_X = 1.40
SIGN_Y = LC_CENTER


# pole
box(
    "direction_sign_pole",
    SIGN_X,
    SIGN_Y,
    0.055,
    0.055,
    0.72,
    GRAY,
    0.36,
    collision=True
)


# blue board
#
# sx = thin
# sy = wide
# sz = high
#
# => board face looks toward +/- X
#
box(
    "direction_sign_board",
    SIGN_X,
    SIGN_Y,
    0.08,
    0.38,
    0.28,
    BLUE,
    0.72,
    collision=False
)


# white right arrow
box(
    "direction_arrow_body",
    SIGN_X - 0.005,
    SIGN_Y,
    0.012,
    0.15,
    0.025,
    WHITE,
    0.72,
    collision=False
)

box(
    "direction_arrow_head",
    SIGN_X,
    SIGN_Y,
    0.012,
    0.055,
    0.075,
    WHITE,
    0.72,
    collision=False
)
'''


pattern = re.compile(
    r"# ============================================================\n"
    r"# D\. LONG LANE-CHANGE RECTANGULAR AREA.*?"
    r"(?=# ============================================================\n"
    r"# E\.|# F\.)",
    re.S
)

text, n1 = pattern.subn(
    new_lane_change + "\n\n",
    text,
    count=1
)


# ============================================================
# 2) ROUNDABOUT -> SECOND CURVE
#
# 기존의 따로 떨어진 선 제거.
# 회전교차로 출구점과 LEFT U-TURN 시작점을 직접 공유.
# ============================================================

new_round_to_curve = r'''
# ============================================================
# ROUNDABOUT -> SECOND CURVE
# ============================================================
#
# The roundabout exit and second curve share EXACT endpoints.
#
# Left curve center:
#   (-3.55, -1.55)
#
# Inner radius:
#   0.42
#
# Outer radius:
#   1.30
#
# ============================================================

LEFT_CX = -3.55
LEFT_CY = -1.55

LEFT_INNER_R = 0.42
LEFT_OUTER_R = 1.30


# ------------------------------------------------------------
# ROUNDABOUT EXIT
#
# exact endpoints of left U-turn
# ------------------------------------------------------------

# upper road edge -> OUTER curve endpoint
polyline(
    "roundabout_to_left_outer",
    [
        (-2.45, 0.62),
        (-2.82, 0.42),
        (-3.18, 0.08),
        (
            LEFT_CX,
            LEFT_CY + LEFT_OUTER_R
        )
    ],
    0.055,
    WHITE
)


# lower road edge -> INNER curve endpoint
polyline(
    "roundabout_to_left_inner",
    [
        (-2.45, -0.38),
        (-2.82, -0.48),
        (-3.18, -0.70),
        (
            LEFT_CX,
            LEFT_CY + LEFT_INNER_R
        )
    ],
    0.055,
    WHITE
)


# ============================================================
# LEFT U-TURN
# ============================================================

arc(
    "left_outer_curve",
    LEFT_CX,
    LEFT_CY,
    LEFT_OUTER_R,
    90,
    270,
    0.055,
    WHITE,
    80
)


arc(
    "left_inner_curve",
    LEFT_CX,
    LEFT_CY,
    LEFT_INNER_R,
    90,
    270,
    0.055,
    WHITE,
    64
)


# ============================================================
# BOTTOM STRAIGHT
#
# Starts from EXACT U-turn endpoints.
# ============================================================

BOTTOM_UPPER = LEFT_CY - LEFT_INNER_R
BOTTOM_LOWER = LEFT_CY - LEFT_OUTER_R

line(
    "bottom_upper",
    LEFT_CX,
    BOTTOM_UPPER,
    4.40,
    BOTTOM_UPPER,
    0.055,
    WHITE
)

line(
    "bottom_lower",
    LEFT_CX,
    BOTTOM_LOWER,
    4.40,
    BOTTOM_LOWER,
    0.055,
    WHITE
)
'''


pattern = re.compile(
    r"# ============================================================\n"
    r"# H\. ROUNDABOUT -> LEFT CURVE.*?"
    r"(?=# ============================================================\n"
    r"# M1 RED)",
    re.S
)

text, n2 = pattern.subn(
    new_round_to_curve + "\n\n",
    text,
    count=1
)


# ============================================================
# 3) CURVE CONES
#
# 임의의 좌표를 없애고 실제 원호를 따라 자동 배치.
# ============================================================

new_cones = r'''
# ============================================================
# M3 LIDAR CONES
#
# Cones follow BOTH SIDES of the actual curves.
#
# IMPORTANT:
# The cone locations are calculated from the same circle
# used for the road lines.
# ============================================================

def place_curve_cones(
    prefix,
    cx,
    cy,
    road_center_radius,
    road_half_width,
    start_deg,
    end_deg,
    count
):

    # outer side
    outer_r = road_center_radius + road_half_width + 0.18

    # inner side
    inner_r = road_center_radius - road_half_width - 0.18

    for i in range(count):

        if count == 1:
            t = 0.5
        else:
            t = i / (count - 1)

        deg = (
            start_deg +
            (end_deg - start_deg) * t
        )

        a = math.radians(deg)


        # outer
        ox = cx + outer_r * math.cos(a)
        oy = cy + outer_r * math.sin(a)

        cone(
            f"{prefix}_outer_{i}",
            ox,
            oy
        )


        # inner
        ix = cx + inner_r * math.cos(a)
        iy = cy + inner_r * math.sin(a)

        cone(
            f"{prefix}_inner_{i}",
            ix,
            iy
        )


# ------------------------------------------------------------
# RIGHT U-TURN
# ------------------------------------------------------------

place_curve_cones(
    "right_curve_cone",
    RIGHT_CX,
    RIGHT_CY,
    (RIGHT_OUTER_R + RIGHT_INNER_R) / 2,
    (RIGHT_OUTER_R - RIGHT_INNER_R) / 2,
    75,
    -75,
    5
)


# ------------------------------------------------------------
# LEFT LOWER U-TURN
# ------------------------------------------------------------

place_curve_cones(
    "left_curve_cone",
    LEFT_CX,
    LEFT_CY,
    (LEFT_OUTER_R + LEFT_INNER_R) / 2,
    (LEFT_OUTER_R - LEFT_INNER_R) / 2,
    105,
    255,
    5
)
'''


# 기존 cone 좌표 블록을 찾아 제거
pattern = re.compile(
    r"# ============================================================\n"
    r"# M3 LIDAR CONES.*?"
    r"(?=# ============================================================\n"
    r"# M5 ROUNDABOUT VEHICLES|# M5)",
    re.S
)

text, n3 = pattern.subn(
    new_cones + "\n\n",
    text,
    count=1
)


path.write_text(
    text,
    encoding="utf-8"
)


print()
print("==============================================")
print(" V5 GEOMETRY PATCH")
print("==============================================")
print("Lane-change section :", n1)
print("Roundabout -> curve :", n2)
print("Curve cones         :", n3)
print()
print("2-LANE ROAD         : OK")
print("CENTER DIVIDER      : OK")
print("BLUE SIGN ON LINE   : OK")
print("CURVE CONNECTION    : EXACT ENDPOINT")
print("CURVE CONES         : AUTO-GENERATED")
print("==============================================")
print()
