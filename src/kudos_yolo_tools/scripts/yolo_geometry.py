#!/usr/bin/env python3
"""
yolo_geometry.py
auto_labeler / yolo_detector_node 가 같이 쓰는 기하 계산 모음.

ROS 의존성이 없으므로 ROS 없이도 단위 테스트할 수 있다.
(Python 3.8 / Foxy 호환 문법만 사용)

좌표 규약
  - T_a_b : b 좌표계의 점을 a 좌표계로 옮기는 4x4 동차변환  (p_a = T_a_b @ p_b)
  - 카메라 광학 좌표계(optical): z 전방, x 오른쪽, y 아래
  - base_link: x 전방, y 왼쪽, z 위 (ROS 표준)
"""

import math

import cv2
import numpy as np


# ---------------------------------------------------------------------
# 회전 / 변환
# ---------------------------------------------------------------------
def quat_to_rot(x, y, z, w):
    """쿼터니언 -> 3x3 회전행렬"""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def yaw_to_rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T, pts):
    """pts (N,3) 에 4x4 변환 적용"""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


# 카메라 "링크"(x 전방, y 왼쪽, z 위) -> 광학 좌표계(z 전방, x 오른쪽, y 아래)
# Gazebo Classic 카메라 플러그인은 링크의 +x 방향을 보고 렌더링한다.
# URDF 에 *_optical_frame 이 없으면 이 회전을 붙여서 써야 한다.
R_LINK_FROM_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])
T_LINK_FROM_OPTICAL = make_T(R_LINK_FROM_OPTICAL, [0, 0, 0])


# ---------------------------------------------------------------------
# 3D 박스 -> 2D 투영
# ---------------------------------------------------------------------
def cuboid_corners(center, size, R=None):
    """물체 중심·크기·회전으로 3D 박스 8꼭짓점 (월드 좌표) 생성"""
    sx, sy, sz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    local = np.array([
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
    ])
    if R is not None:
        local = local @ np.asarray(R).T
    return local + np.asarray(center, dtype=float).reshape(3)


def project_points(K, pts_cam, min_z=0.05):
    """광학 좌표계 점 -> 픽셀. 하나라도 카메라 뒤(또는 min_z 이내)면 None."""
    pts_cam = np.asarray(pts_cam, dtype=float).reshape(-1, 3)
    if np.any(pts_cam[:, 2] <= min_z):
        return None
    u = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1)


def clipped_hull(uv, img_w, img_h):
    """
    투영된 꼭짓점들의 볼록껍질(convex hull)을 이미지 영역으로 잘라낸 다각형.
    반환: (poly (M,2) float, 잘리기 전 면적 대비 남은 비율) 또는 (None, 0)

    3D 박스의 2D 실루엣 = 꼭짓점 투영의 볼록껍질 이므로,
    이 다각형은 segmentation 라벨로도, bbox(min/max) 로도 쓸 수 있다.
    """
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    full_area = float(cv2.contourArea(hull))
    if full_area < 1e-6:
        return None, 0.0
    rect = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32)
    area, inter = cv2.intersectConvexConvex(hull.astype(np.float32), rect)
    if inter is None or area <= 1e-6:
        return None, 0.0
    poly = inter.reshape(-1, 2).astype(np.float64)
    poly[:, 0] = np.clip(poly[:, 0], 0, img_w)
    poly[:, 1] = np.clip(poly[:, 1], 0, img_h)
    return poly, float(area) / full_area


def poly_bbox(poly):
    return (float(poly[:, 0].min()), float(poly[:, 1].min()),
            float(poly[:, 0].max()), float(poly[:, 1].max()))


def yolo_line(class_id, poly, img_w, img_h, fmt='polygon'):
    """
    YOLO 라벨 한 줄.
      fmt='polygon' : class x1 y1 x2 y2 ...  (segmentation 형식)
                      ※ detect 학습에서도 그대로 쓸 수 있다. ultralytics 가
                        다각형의 min/max 로 박스를 자동 계산한다.
      fmt='box'     : class cx cy w h
    """
    if fmt == 'box':
        x1, y1, x2, y2 = poly_bbox(poly)
        return '%d %.6f %.6f %.6f %.6f' % (
            class_id, (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h,
            (x2 - x1) / img_w, (y2 - y1) / img_h)
    coords = []
    for u, v in poly:
        coords.append('%.6f' % min(max(u / img_w, 0.0), 1.0))
        coords.append('%.6f' % min(max(v / img_h, 0.0), 1.0))
    return '%d %s' % (class_id, ' '.join(coords))


def is_front_facing(front, center, R_obj, cam_pos, min_cos=0.2):
    """
    카메라가 물체의 앞면 쪽에 있는지. front: '+x' '-x' '+y' '-y' (모델 좌표계).
    min_cos=0.2 → 앞면 법선에서 약 78도 이내에서 볼 때만 True
    """
    axis = {'x': 0, 'y': 1, 'z': 2}[front[-1]]
    sign = -1.0 if front.startswith('-') else 1.0
    n = sign * np.asarray(R_obj)[:, axis]
    v = np.asarray(cam_pos, dtype=float) - np.asarray(center, dtype=float)
    d = np.linalg.norm(v)
    if d < 1e-9:
        return False
    return float(n @ v) / d > min_cos


# ---------------------------------------------------------------------
# 가림(occlusion) 판정
# ---------------------------------------------------------------------
def occluded_ratio(depth_m, poly, obj_min_depth, tol=0.05):
    """
    다각형 내부 픽셀 중 "물체보다 확실히 앞에 있는" 픽셀의 비율.
    배경은 물체보다 뒤에 있으므로 이 비율에 영향을 주지 않는다.
    앞을 가리는 벽·다른 물체만 비율을 올린다.

    depth_m : (H,W) float 미터, 0/NaN/inf = 무효
    반환    : 0.0~1.0, 유효 깊이 픽셀이 없으면 None
    """
    h, w = depth_m.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    vals = depth_m[mask.astype(bool)]
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return None
    return float(np.count_nonzero(vals < obj_min_depth - tol)) / float(vals.size)


def depth_to_meters(depth, encoding):
    """sensor_msgs/Image 깊이 -> 미터. 16UC1 은 mm(Orbbec), 32FC1 은 m(Gazebo)."""
    d = np.asarray(depth)
    if encoding in ('16UC1', 'mono16') or d.dtype == np.uint16:
        return d.astype(np.float32) / 1000.0
    return d.astype(np.float32)


# ---------------------------------------------------------------------
# 추론 결과 -> 거리 추정 (yolo_detector_node 용)
# ---------------------------------------------------------------------
def backproject(u, v, z, K):
    """픽셀 (u,v) + 깊이 z(광학 z) -> 광학 좌표계 3D 점"""
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.array([x, y, z])


def ray_ground_intersection(u, v, K, T_base_cam, ground_z=0.0):
    """
    픽셀 (u,v) 를 지나는 광선과 바닥 평면(base_link 기준 z=ground_z)의 교점.
    IPM 과 같은 원리지만 사다리꼴 4점 대신 카메라 내부/외부 파라미터를 쓴다.
    반환: base_link 기준 (x, y) 또는 None(광선이 바닥을 안 만남)
    """
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    R = T_base_cam[:3, :3]
    o = T_base_cam[:3, 3]
    d = R @ d_cam
    if abs(d[2]) < 1e-9:
        return None
    s = (ground_z - o[2]) / d[2]
    if s <= 0:
        return None
    p = o + s * d
    return float(p[0]), float(p[1])


def robust_box_depth(depth_m, x1, y1, x2, y2, inner=0.5):
    """bbox 안쪽 중앙 영역 깊이의 중앙값 (m). 배경이 섞이는 가장자리는 버린다."""
    h, w = depth_m.shape[:2]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * inner / 2.0, (y2 - y1) * inner / 2.0
    xa, xb = int(max(cx - hw, 0)), int(min(cx + hw, w - 1)) + 1
    ya, yb = int(max(cy - hh, 0)), int(min(cy + hh, h - 1)) + 1
    if xb <= xa or yb <= ya:
        return None
    vals = depth_m[ya:yb, xa:xb].ravel()
    vals = vals[np.isfinite(vals) & (vals > 0.05)]
    if vals.size < 5:
        return None
    # 가장 가까운 층(물체 면)을 쓰기 위해 하위 30% 분위수 사용
    return float(np.percentile(vals, 30))


def depth_from_size(pixel_h, real_h_m, fy):
    """알려진 실제 높이로 거리 역산 (핀홀). 깊이가 없을 때의 대안."""
    if pixel_h <= 1:
        return None
    return float(fy * real_h_m / pixel_h)


def bar_angle_from_points(pts):
    """
    차단기 바 마스크(또는 윤곽선) 점들 -> 화면상 기울기 [rad].
    0 = 수평(닫힘), pi/2 = 수직(완전 열림). 항상 0~pi/2 로 반환.
    ※ 화면 기준 각도이므로 정면에서 볼 때 실제 각도와 가장 가깝다.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 5:
        return None
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    ang = abs(math.atan2(float(vy), float(vx)))
    if ang > math.pi / 2:
        ang = math.pi - ang
    return ang
