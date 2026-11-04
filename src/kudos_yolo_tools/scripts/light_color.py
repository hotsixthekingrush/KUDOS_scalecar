#!/usr/bin/env python3
"""
light_color.py — 신호등 박스 안의 불 색 판정 (YOLO 는 신호등 1클래스, 색은 여기서)

판정 순서
  1) 박스 안에서 "밝고 색이 진한" 픽셀만 남긴다 (켜진 램프).
  2) 그 픽셀의 색상(hue)으로 빨강/노랑/초록 표를 센다.
  3) 켜진 픽셀이 너무 적거나 두 색이 비슷하면 → 박스 세로 3등분 중 가장 밝은 칸
     (위=빨강, 가운데=노랑, 아래=초록) 으로 한 번 더 본다.
  4) 그래도 애매하면 0(모름). 판단 쪽은 모름 = 정지로 처리하므로 틀린 색보다 낫다.

반환값은 MissionObject.light_color 규약과 같다: 0 모름, 1 빨강, 2 노랑, 3 초록
ROS 없이 import 할 수 있다 (오프라인 시험용).
"""
import numpy as np
import cv2

UNKNOWN, RED, YELLOW, GREEN = 0, 1, 2, 3
NAMES = {UNKNOWN: 'unknown', RED: 'red', YELLOW: 'yellow', GREEN: 'green'}

# OpenCV HSV: H 0~179
_HUE = {RED: [(0, 10), (165, 179)], YELLOW: [(14, 38)], GREEN: [(40, 100)]}


def _hue_votes(hsv, mask):
    h = hsv[..., 0][mask]
    votes = {}
    for c, ranges in _HUE.items():
        votes[c] = int(sum(((h >= a) & (h <= b)).sum() for a, b in ranges))
    return votes


def classify(bgr, box, s_min=90, v_min=150, min_lit_frac=0.01, dominance=1.8):
    """bgr: 전체 영상, box: (x1, y1, x2, y2) → (색 코드, 근거 문자열)"""
    h_img, w_img = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 - x1 < 3 or y2 - y1 < 6:
        return UNKNOWN, 'box too small'
    roi = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lit = (hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min)
    n_lit = int(lit.sum())

    if n_lit >= max(4, min_lit_frac * lit.size):
        votes = _hue_votes(hsv, lit)
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        (c1, n1), (_, n2) = ranked[0], ranked[1]
        if n1 > 0 and n1 >= dominance * max(n2, 1):
            return c1, 'hue %s' % votes

    # 위치로 보조 판정: 세로 3등분 중 밝은 칸 (가로로 누운 신호등이면 가로 3등분)
    v = hsv[..., 2].astype(np.float32)
    vertical = roi.shape[0] >= roi.shape[1]
    parts = np.array_split(v, 3, axis=0 if vertical else 1)
    means = [float(p.mean()) if p.size else 0.0 for p in parts]
    order = np.argsort(means)[::-1]
    if means[order[0]] > 1.4 * max(means[order[1]], 1.0) and means[order[0]] > 90:
        return (RED, YELLOW, GREEN)[int(order[0])], 'thirds %s' % [round(m) for m in means]
    return UNKNOWN, 'unclear lit=%d' % n_lit
