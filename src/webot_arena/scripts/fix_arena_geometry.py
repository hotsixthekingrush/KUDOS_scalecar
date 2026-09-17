from pathlib import Path
import re

path = Path("make_arena.py")
text = path.read_text(encoding="utf-8")


# ============================================================
# 1. LANE CHANGE AREA
#    기존 좁은 직사각형을 삭제하고
#    넓어진 2차선 구간 + 중앙 분리선으로 변경
# ============================================================

new_lane_change = r'''
# ============================================================
# D. TWO-LANE LANE-CHANGE SECTION
#
# This section widens to TWO LANES.
#
# Road boundary:
#
#        upper boundary
#   ─────────────────────────
#
#        lane 1
#
#   - - - - - - - - - - - - -
#        CENTER LINE
#
#        lane 2
#
#   ─────────────────────────
#        lower boundary
#
# Blue direction sign is mounted
# ON the center divider line near the right end.
# ============================================================

LC_LEFT = -0.10
LC_RIGHT = 1.60

LC_TOP = 1.75
LC_CENTER = 1.10
LC_BOTTOM = 0.45


# upper road boundary
line(
    "lanechange_upper",
    LC_LEFT,
    LC_TOP,
    LC_RIGHT,
    LC_TOP
)


# CENTER LANE DIVIDER
line(
    "lanechange_center",
    LC_LEFT,
    LC_CENTER,
    LC_RIGHT,
    LC_CENTER,
    0.055,
    WHITE
)


# lower road boundary
line(
    "lanechange_lower",
    LC_LEFT,
    LC_BOTTOM,
    LC_RIGHT,
    LC_BOTTOM
)


# ============================================================
# BLUE RIGHT-DIRECTION SIGN
#
# Mounted directly on the CENTER DIVIDER LINE.
# ============================================================

SIGN_X = 1.28
SIGN_Y = LC_CENTER


# pole
box(
    "direction_sign_pole",
    SIGN_X,
    SIGN_Y,
    0.055,
    0.055,
    0.70,
    GRAY,
    0.35
)


# blue sign board
box(
    "direction_sign_board",
    SIGN_X,
    SIGN_Y,
    0.38,
    0.08,
    0.26,
    BLUE,
    0.72,
    collision=False
)


# white arrow shaft
box(
    "direction_arrow_shaft",
    SIGN_X - 0.02,
    SIGN_Y,
    0.14,
    0.015,
    0.025,
    WHITE,
    0.72,
    collision=False
)


# white arrow head
box(
    "direction_arrow_head",
    SIGN_X + 0.075,
    SIGN_Y,
    0.055,
    0.015,
    0.075,
    WHITE,
    0.72,
    collision=False
)
'''


pattern_lane = re.compile(
    r"# ============================================================\n"
    r"# D\. LONG LANE-CHANGE RECTANGULAR AREA.*?"
    r"(?=# ============================================================\n"
    r"# F\. ROAD -> ROUNDABOUT)",
    re.S
)

text, count_lane = pattern_lane.subn(
    new_lane_change + "\n\n",
    text
)


# ============================================================
# 2. ROUNDABOUT -> LEFT CURVE
#    실제 LEFT U-TURN ARC endpoint와 정확히 연결
# ============================================================

new_round_to_left = r'''
# ============================================================
# H. ROUNDABOUT -> SECOND / LEFT CURVE
#
# IMPORTANT:
# The endpoints below are EXACTLY the start points
# of the left U-turn arcs.
#
# LEFT U-TURN:
#   center = (-3.55, -1.55)
#   outer R = 1.30
#   inner R = 0.42
#
# Therefore:
#
# outer upper endpoint = (-3.55, -0.25)
# inner upper endpoint = (-3.55, -1.13)
#
# These are used directly.
# ============================================================

LEFT_CX = -3.55
LEFT_CY = -1.55

LEFT_OUTER_R = 1.30
LEFT_INNER_R = 0.42


# ------------------------------------------------------------
# ROUNDABOUT EXIT -> OUTER CURVE
# exact endpoint
# ------------------------------------------------------------

polyline(
    "roundabout_exit_outer",
    [
        (-2.48, 0.59),
        (-2.82, 0.42),
        (-3.18, 0.05),
        (
            LEFT_CX,
            LEFT_CY + LEFT_OUTER_R
        )
    ]
)


# ------------------------------------------------------------
# ROUNDABOUT EXIT -> INNER CURVE
# exact endpoint
# ------------------------------------------------------------

polyline(
    "roundabout_exit_inner",
    [
        (-2.48, -0.40),
        (-2.82, -0.48),
        (-3.18, -0.72),
        (
            LEFT_CX,
            LEFT_CY + LEFT_INNER_R
        )
    ]
)


# ============================================================
# I. LEFT LOWER U-TURN
# ============================================================

arc(
    "left_inner_curve",
    LEFT_CX,
    LEFT_CY,
    LEFT_INNER_R,
    90,
    270,
    LINE_W,
    WHITE,
    72
)

arc(
    "left_outer_curve",
    LEFT_CX,
    LEFT_CY,
    LEFT_OUTER_R,
    90,
    270,
    LINE_W,
    WHITE,
    72
)
'''


pattern_round = re.compile(
    r"# ============================================================\n"
    r"# H\. ROUNDABOUT -> LEFT CURVE.*?"
    r"(?=# ============================================================\n"
    r"# J\. BOTTOM STRAIGHT)",
    re.S
)

text, count_round = pattern_round.subn(
    new_round_to_left + "\n\n",
    text
)


# ============================================================
# 3. PARKING AREA + P SIGN
# ============================================================

new_parking = r'''
# ============================================================
# M8 PARKING
#
# After the barrier.
# Parking area is on the LEFT side of the road.
# Orientation = VERTICAL.
# ============================================================

PARK_X = 2.70
PARK_Y = -1.20

PARK_W = 0.72
PARK_L = 1.55


# left boundary
line(
    "parking_left",
    PARK_X - PARK_W / 2,
    PARK_Y - PARK_L / 2,
    PARK_X - PARK_W / 2,
    PARK_Y + PARK_L / 2,
    0.055
)

# right boundary
line(
    "parking_right",
    PARK_X + PARK_W / 2,
    PARK_Y - PARK_L / 2,
    PARK_X + PARK_W / 2,
    PARK_Y + PARK_L / 2,
    0.055
)

# top boundary
line(
    "parking_top",
    PARK_X - PARK_W / 2,
    PARK_Y + PARK_L / 2,
    PARK_X + PARK_W / 2,
    PARK_Y + PARK_L / 2,
    0.055
)

# bottom boundary
line(
    "parking_bottom",
    PARK_X - PARK_W / 2,
    PARK_Y - PARK_L / 2,
    PARK_X + PARK_W / 2,
    PARK_Y - PARK_L / 2,
    0.055
)


# ============================================================
# BLUE PARKING SIGN
#
# Located BEFORE the parking turn.
# On the LEFT side of the road.
# ============================================================

P_SIGN_X = 2.10
P_SIGN_Y = -1.50


# pole
box(
    "parking_sign_pole",
    P_SIGN_X,
    P_SIGN_Y,
    0.055,
    0.055,
    0.85,
    GRAY,
    0.425
)


# blue sign
box(
    "parking_sign_board",
    P_SIGN_X,
    P_SIGN_Y,
    0.40,
    0.08,
    0.32,
    BLUE,
    0.95,
    collision=False
)


# ============================================================
# WHITE "P" SYMBOL
# ============================================================

# vertical stem
box(
    "parking_P_vertical",
    P_SIGN_X - 0.075,
    P_SIGN_Y,
    0.025,
    0.018,
    0.20,
    WHITE,
    0.95,
    collision=False
)

# top horizontal
box(
    "parking_P_top",
    P_SIGN_X - 0.015,
    P_SIGN_Y,
    0.12,
    0.018,
    0.025,
    WHITE,
    1.035,
    collision=False
)

# upper right vertical
box(
    "parking_P_right",
    P_SIGN_X + 0.045,
    P_SIGN_Y,
    0.025,
    0.018,
    0.10,
    WHITE,
    0.985,
    collision=False
)
'''


pattern_parking = re.compile(
    r"# ============================================================\n"
    r"# M8 PARKING.*?"
    r"(?=# ============================================================\n"
    r"# FINISH)",
    re.S
)

text, count_parking = pattern_parking.subn(
    new_parking + "\n\n",
    text
)


# ============================================================
# SAVE
# ============================================================

path.write_text(
    text,
    encoding="utf-8"
)


print()
print("==============================================")
print(" GEOMETRY PATCH COMPLETE")
print("==============================================")
print("Lane change replacement :", count_lane)
print("Roundabout connection   :", count_round)
print("Parking replacement     :", count_parking)
print()
print("2-LANE SECTION          : OK")
print("CENTER DIVIDER SIGN     : OK")
print("ROUNDABOUT -> CURVE     : EXACT ENDPOINT")
print("VERTICAL PARKING        : OK")
print("BLUE P SIGN             : OK")
print("==============================================")
print()
