#!/usr/bin/env python3
"""
line_tracing_node v3

영상 분석으로 확인한 실패 원인
  1) heading 이 xm/ym(0.3) 비율만큼 축소돼서 계산됨 -> 회전 끝에서 차량이 이미 45도 틀어져도
     "거의 똑바로"라고 판단하고 계속 우회전 (확 돌아버림)
  2) 곡률 feedforward 가 실제의 1/10 수준 (ym=0.01 가정) -> U턴을 횡오차 P항으로만 돌아야 했음
  3) steer = kp*횡오차 + heading 구조라 횡오차가 크면 조향이 포화 -> 접근각 제한이 없음
  4) 화면 가로 거리로 횡오차를 재서, 차선이 기울어질수록 오차가 과대평가됨

제어 구조 (Stanley + feedforward)
  approach = clip( atan(kp * 횡오차 / v), ±max_approach )   # 중앙선으로 접근하는 각도 제한
  steer    = heading_gain * (헤딩오차 + approach) + atan(L * 전방 곡률) * ff_gain
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


PARAMS = [
    ('camera_topic', '/camera/image_raw'),
    ('cmd_vel_topic', '/cmd_vel'),
    ('lane_detected_topic', '/line_tracing/lane_detected'),
    ('debug_image_topic', '/line_tracing/debug_image'),
    ('roi_image_topic', '/line_tracing/roi_image'),
    ('publish_debug_image', True),

    ('image_width', 640),
    ('image_height', 480),

    ('ipm_src_points', [0.35, 0.62, 0.65, 0.62, 0.95, 0.95, 0.05, 0.95]),
    ('ipm_dst_margin_px', 150),

    # ---- 스케일 ----
    # lane_width_m > 0 : 실제 트랙의 "두 노란선 사이 거리[m]" 를 넣으면
    #                    직선 구간에서 측정한 픽셀 폭으로 xm_per_pix 를 자동 보정한다.
    # lane_width_m = 0 : 아래 xm_per_pix 를 그대로 사용
    ('lane_width_m', 0.40),
    ('xm_per_pix', 0.003),
    # ym_per_pix / xm_per_pix.  IPM 이 정상이면 1.0 (등방)
    ('pixel_aspect', 1.0),

    ('hsv_yellow_lower', [15, 80, 80]),
    ('hsv_yellow_upper', [35, 255, 255]),
    ('morph_kernel_size', 5),
    ('morph_iterations', 1),

    ('histogram_rows_ratio', 0.35),   # 화면 아래쪽 이 비율만 히스토그램에 사용
    ('histogram_smooth_px', 41),
    ('peak_suppress_px', 150),        # 첫 번째 피크 주변은 두 번째 피크 후보에서 제외
    ('sliding_window_count', 12),
    ('sliding_window_margin_px', 60),
    ('sliding_window_min_pixels', 40),
    ('sliding_window_min_pixels_base', 500),
    ('sliding_window_min_total_pixels', 200),
    ('min_lane_span_ratio', 0.25),
    ('near_coverage_min_ratio', 0.55),  # 차선 픽셀이 "차량 근처"(화면 아래쪽)에
                                        # 이 비율 이상까지 있어야 그 차선을 신뢰

    ('near_ratio', 0.10),             # 횡오차/헤딩오차 측정 위치 (아래에서 10%)
    ('lookahead_ratio', 0.35),        # 곡률 feedforward 측정 위치
    ('vehicle_center_offset_px', 0.0),

    ('lane_side', 'right'),           # 기존 YAML 호환용 (사용 안 함)
    ('lane_offset_m', 0.20),          # 차선 폭을 전혀 모를 때만 사용

    ('pid_kp', 1.5),                  # 횡오차 -> 접근각 게인 [1/s]
    ('heading_gain', 2.0),            # 헤딩(+접근각) -> 조향각 게인
    ('max_approach_rad', 0.6),        # 접근각 제한 (약 35도)
    ('curvature_ff_gain', 1.0),
    ('steer_lpf_alpha', 0.7),         # 1.0 = 필터 없음

    ('wheelbase_m', 0.2),
    ('max_steer_rad', 0.5236),

    ('linear_speed_mps', 0.3),
    ('curve_slowdown_gain', 0.5),     # 0 이면 항상 linear_speed_mps 유지
    ('min_speed_ratio', 0.6),
    ('control_rate_hz', 20.0),
    ('frame_timeout_sec', 0.5),

    ('failsafe_hold_sec', 1.0),
    ('failsafe_creep_sec', 3.0),
    ('failsafe_creep_speed_mps', 0.1),
    ('failsafe_stop_after_sec', 5.0),
]


def _poly(fit, y):
    return fit[0] * y ** 2 + fit[1] * y + fit[2]


@dataclass
class LaneResult:
    left_fit: Optional[np.ndarray]
    right_fit: Optional[np.ndarray]
    center_fit: Optional[np.ndarray]
    mode: str
    debug_img: np.ndarray
    lane_width_px: Optional[float]
    y_range: tuple = (0.0, 0.0)   # center_fit 이 실제 픽셀로 뒷받침되는 y 구간


class LaneDetector:

    def __init__(self, params: dict):
        self.p = params
        self.aspect = float(params['pixel_aspect'])

        # 직선에서 측정한 "수직 방향" 차선 폭 [x 픽셀 단위]
        self._lane_width_px = None

        # 좌/우 차선이 마지막으로 보였던 위치 (한쪽만 보일 때 좌/우 구분용)
        self._prev_x = {'left': None, 'right': None}
        self._age = {'left': 0, 'right': 0}

        self._size = None
        self._build_ipm_matrix(params['image_width'], params['image_height'])

    # ---- 스케일 -------------------------------------------------
    @property
    def xm(self):
        if self.p['lane_width_m'] > 0.0 and self._lane_width_px:
            return self.p['lane_width_m'] / self._lane_width_px
        return self.p['xm_per_pix']

    @property
    def ym(self):
        return self.xm * self.aspect

    # ---- IPM ----------------------------------------------------
    def _build_ipm_matrix(self, w, h):
        src = np.float32([[x * w, y * h] for x, y in self.p['ipm_src_points']])
        m = self.p['ipm_dst_margin_px']
        dst = np.float32([[m, 0], [w - m, 0], [w - m, h], [m, h]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self._src_px = src.astype(np.int32)
        self._size = (w, h)

    def apply_roi_and_ipm(self, bgr_img):
        h, w = bgr_img.shape[:2]
        if (w, h) != self._size:
            self._build_ipm_matrix(w, h)
        return cv2.warpPerspective(bgr_img, self.M, (w, h), flags=cv2.INTER_LINEAR)

    def draw_roi(self, bgr_img):
        out = bgr_img.copy()
        cv2.polylines(out, [self._src_px.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
        return out

    # ---- HSV + Morphology --------------------------------------
    def threshold_lane(self, bird_eye_bgr):
        blur = cv2.GaussianBlur(bird_eye_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower = np.array(self.p['hsv_yellow_lower'], dtype=np.uint8)
        upper = np.array(self.p['hsv_yellow_upper'], dtype=np.uint8)
        binary = cv2.inRange(hsv, lower, upper)

        k = self.p['morph_kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,
                                  iterations=self.p['morph_iterations'])
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    # ---- 시작점(피크) 찾기: 화면 전체 폭에서 최대 2개 ------------
    @staticmethod
    def _peak(hist):
        m = float(hist.max())
        if m <= 0.0:
            return 0, 0.0
        thr = 0.98 * m
        i = int(np.argmax(hist))
        lo = hi = i
        # 최대값 주변의 "연속된" plateau 의 중앙을 시작점으로 사용
        while lo > 0 and hist[lo - 1] >= thr:
            lo -= 1
        while hi < len(hist) - 1 and hist[hi + 1] >= thr:
            hi += 1
        return (lo + hi) // 2, m

    def _find_bases(self, binary):
        h, w = binary.shape[:2]
        y0 = int(h * (1.0 - self.p['histogram_rows_ratio']))

        # binary 는 0/255 -> 0/1 픽셀 "개수"로 센다
        hist = (binary[y0:, :] > 0).sum(axis=0).astype(np.float32)
        k = max(1, int(self.p['histogram_smooth_px']))
        hist = np.convolve(hist, np.ones(k, dtype=np.float32), mode='same')

        min_base = self.p['sliding_window_min_pixels_base']
        sup = int(self.p['peak_suppress_px'])

        bases = []
        for _ in range(2):
            x, v = self._peak(hist)
            if v < min_base:
                break
            bases.append(x)
            hist[max(0, x - sup):min(w, x + sup + 1)] = 0.0
        return bases

    # ---- 한쪽 차선 Sliding Window -------------------------------
    def _fit_one_lane(self, binary, base_x, nz_x, nz_y, debug_img):
        h, w = binary.shape[:2]
        n_windows = self.p['sliding_window_count']
        win_h = h // n_windows
        margin = self.p['sliding_window_margin_px']
        min_pix = self.p['sliding_window_min_pixels']

        cur_x = int(base_x)
        shift = 0.0
        inds = []

        for i in range(n_windows):
            y_low = h - (i + 1) * win_h
            y_high = h - i * win_h
            x_low = max(0, cur_x - margin)
            x_high = min(w, cur_x + margin)
            cv2.rectangle(debug_img, (x_low, y_low), (x_high, y_high), (0, 255, 0), 1)

            good = ((nz_y >= y_low) & (nz_y < y_high)
                    & (nz_x >= x_low) & (nz_x < x_high)).nonzero()[0]
            inds.append(good)

            if len(good) > min_pix:
                new_x = int(np.mean(nz_x[good]))
                shift = 0.5 * shift + 0.5 * (new_x - cur_x)
                cur_x = new_x
            else:
                shift *= 0.5
                cur_x = int(np.clip(cur_x + shift, 0, w - 1))

        inds = np.concatenate(inds) if inds else np.array([], dtype=np.int64)
        n_pixels = len(inds)
        if n_pixels < self.p['sliding_window_min_total_pixels']:
            return None, n_pixels, h, 0.0

        x_pts = nz_x[inds]
        y_pts = nz_y[inds]
        y_lo, y_hi = float(y_pts.min()), float(y_pts.max())
        if y_hi - y_lo < self.p['min_lane_span_ratio'] * h:
            return None, n_pixels, y_lo, y_hi

        try:
            fit = np.polyfit(y_pts, x_pts, 2)
        except (np.linalg.LinAlgError, TypeError, ValueError):
            return None, n_pixels, y_lo, y_hi
        return fit, n_pixels, y_lo, y_hi

    # ---- 차선 폭 (수직 방향, x 픽셀 단위) ------------------------
    def _measure_width(self, lf, rf, w, h):
        y = np.linspace(h * 0.5, h - 1, 40)
        dx_px = _poly(rf, y) - _poly(lf, y)
        valid = (dx_px > 30.0) & (dx_px < float(w))
        if not np.any(valid):
            return None, None, 0.0

        # 미터 좌표에서의 중앙선 기울기 dX/dY = (dx/dy) / aspect
        slope = ((lf[0] + rf[0]) * y + (lf[1] + rf[1]) / 2.0) / self.aspect
        width_perp = dx_px / np.sqrt(1.0 + slope ** 2)

        return (float(np.median(width_perp[valid])),
                float(np.median(dx_px[valid])),
                float(np.median(np.abs(slope[valid]))))

    # ---- 한쪽 차선을 법선 방향으로 offset -> 가상 중앙선 ---------
    def _offset_fit(self, fit, offset_m, h):
        xm, ym = self.xm, self.ym
        y = np.linspace(0, h - 1, 30)
        x = _poly(fit, y)
        dxdy = 2.0 * fit[0] * y + fit[1]

        tx = xm * dxdy
        ty = ym
        norm = np.hypot(tx, ty)
        nx = ty / norm
        ny = -tx / norm

        X = x * xm + offset_m * nx
        Y = y * ym + offset_m * ny
        return np.polyfit(Y / ym, X / xm, 2)

    # ---- 한쪽만 보일 때 좌/우 판정: 이전 프레임과의 연속성 -------
    def _assign_side(self, fit, y_near, w):
        x_ref = float(_poly(fit, y_near))
        cands = [(abs(x_ref - px), s) for s, px in self._prev_x.items()
                 if px is not None]
        if cands:
            return min(cands)[1]
        return 'left' if x_ref < w / 2.0 else 'right'

    def _update_tracking(self, lf, rf, y_near):
        for side, fit in (('left', lf), ('right', rf)):
            if fit is not None:
                self._prev_x[side] = float(_poly(fit, y_near))
                self._age[side] = 0
            else:
                self._age[side] += 1
                if self._age[side] > 90:      # 약 3초 이상 못 봤으면 잊는다
                    self._prev_x[side] = None

    # ---- 메인 ---------------------------------------------------
    def detect(self, binary):
        h, w = binary.shape[:2]
        debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        y_near = h * (1.0 - self.p['near_ratio'])

        nz_y, nz_x = binary.nonzero()

        near_required = self.p['near_coverage_min_ratio'] * h

        fits = []
        for base in self._find_bases(binary):
            fit, n, y_lo, y_hi = self._fit_one_lane(binary, base, nz_x, nz_y, debug)
            if fit is None:
                continue
            if y_hi < near_required:
                # 차량 바로 앞 구간에 픽셀이 없다 -> 이 차선의 2차 곡선을
                # 그 지점까지 늘려 그리면 위험하게 어긋난다 (검증됨).
                # 그리기만 하고(디버그용) 제어에는 쓰지 않는다.
                cv2.putText(debug, 'reject:no near px', (15, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
                            cv2.LINE_AA)
                continue
            fits.append((fit, n, y_lo, y_hi))
        fits.sort(key=lambda t: float(_poly(t[0], y_near)))

        left_fit = right_fit = None
        l_pix = r_pix = 0
        l_range = r_range = (0.0, float(h))

        if len(fits) == 2:
            (left_fit, l_pix, *l_range), (right_fit, r_pix, *r_range) = fits
        elif len(fits) == 1:
            fit, n, y_lo, y_hi = fits[0]
            if self._assign_side(fit, y_near, w) == 'left':
                left_fit, l_pix, l_range = fit, n, (y_lo, y_hi)
            else:
                right_fit, r_pix, r_range = fit, n, (y_lo, y_hi)

        # ---- 양쪽이 잡혔을 때: 폭 일관성 검사 + 폭 기억 갱신 ----
        width_px = None
        if left_fit is not None and right_fit is not None:
            w_perp, width_px, tilt = self._measure_width(left_fit, right_fit, w, h)

            reject = w_perp is None
            if (not reject) and self._lane_width_px is not None:
                reject = not (0.6 <= w_perp / self._lane_width_px <= 1.5)

            if reject:
                if l_pix >= r_pix:
                    right_fit = None
                else:
                    left_fit = None
                width_px = None
            elif tilt < 0.5:
                if self._lane_width_px is None:
                    self._lane_width_px = w_perp
                else:
                    self._lane_width_px = 0.85 * self._lane_width_px + 0.15 * w_perp

        self._update_tracking(left_fit, right_fit, y_near)

        plot_y = np.linspace(0, h - 1, h)
        self._draw_fit(debug, left_fit, plot_y, (0, 0, 255), 3)
        self._draw_fit(debug, right_fit, plot_y, (255, 0, 0), 3)

        if self._lane_width_px is not None:
            half_w = 0.5 * self._lane_width_px * self.xm
        elif self.p['lane_width_m'] > 0.0:
            half_w = 0.5 * self.p['lane_width_m']
        else:
            half_w = self.p['lane_offset_m']

        center_fit = None
        mode = 'none'
        y_range = (0.0, float(h))
        if left_fit is not None and right_fit is not None:
            center_fit = (left_fit + right_fit) / 2.0
            mode = 'dual'
            y_lo = max(l_range[0], r_range[0])
            y_hi = min(l_range[1], r_range[1])
            if y_lo > y_hi:      # 두 차선의 데이터 구간이 겹치지 않음 -> 넓게 잡되 보수적으로
                y_lo = min(l_range[0], r_range[0])
                y_hi = max(l_range[1], r_range[1])
            y_range = (y_lo, y_hi)
        elif left_fit is not None:
            center_fit = self._offset_fit(left_fit, +half_w, h)
            mode = 'left_only'
            y_range = l_range
        elif right_fit is not None:
            center_fit = self._offset_fit(right_fit, -half_w, h)
            mode = 'right_only'
            y_range = r_range

        if center_fit is not None:
            self._draw_fit(debug, center_fit, plot_y, (0, 255, 0), 4)
            label = mode if width_px is None else f'{mode} w={width_px:.0f}px'
            cv2.putText(debug, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(debug, (0, int(y_range[0])), (w, int(y_range[0])), (0, 165, 255), 1)
            cv2.line(debug, (0, int(y_range[1])), (w, int(y_range[1])), (0, 165, 255), 1)

        return LaneResult(left_fit, right_fit, center_fit, mode, debug, width_px, y_range)

    @staticmethod
    def _draw_fit(img, fit, y, color, thickness):
        if fit is None:
            return
        h, w = img.shape[:2]
        x = np.clip(_poly(fit, y), -w, 2 * w)
        pts = np.stack([x, y], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, thickness)

    # ---- 픽셀 데이터가 뒷받침하는 구간 밖에서는 2차항을 쓰지 않고
    #      경계에서의 접선(1차)으로만 연장한다. 2차 다항식을 데이터가
    #      없는 곳까지 그대로 늘리면 작은 노이즈가 근접점에서 수십 px
    #      오차로 증폭된다 (실측/시뮬레이션으로 확인됨).
    @staticmethod
    def _bounded_x_slope(fit, y, y_lo, y_hi):
        yc = float(np.clip(y, y_lo, y_hi))
        x_c = _poly(fit, yc)
        slope_px = 2.0 * fit[0] * yc + fit[1]
        x = x_c + slope_px * (y - yc)
        return x, slope_px, (yc == y)   # 마지막 값: 외삽 없이 순수 데이터 구간 안인지

    # ---- 오차 (미터/라디안) -------------------------------------
    #   lateral : 중앙선까지의 "수직" 거리, 목표가 오른쪽이면 음수
    #   heading : 차량 진행방향 대비 경로 방향, 경로가 왼쪽으로 꺾여 있으면 양수
    #   kappa   : 전방 곡률, 왼쪽으로 휘면 양수
    def compute_path_errors(self, center_fit, image_shape, y_range=None):
        h, w = image_shape[:2]
        xm, ym = self.xm, self.ym
        ratio = xm / ym

        if y_range is None:
            y_range = (0.0, float(h))
        y_lo, y_hi = y_range

        y_near = h * (1.0 - self.p['near_ratio'])
        x_near, slope_near_px, _ = self._bounded_x_slope(center_fit, y_near, y_lo, y_hi)
        slope_near = slope_near_px * ratio
        heading = float(np.arctan(slope_near))

        center_px = w / 2.0 + self.p['vehicle_center_offset_px']
        lateral_m = (center_px - x_near) * xm / np.sqrt(1.0 + slope_near ** 2)

        y_look = h * (1.0 - self.p['lookahead_ratio'])
        _, slope_look_px, in_range = self._bounded_x_slope(center_fit, y_look, y_lo, y_hi)
        slope_look = slope_look_px * ratio
        if in_range:
            d2X_dY2 = 2.0 * center_fit[0] * ratio / ym
            kappa_right = d2X_dY2 / (1.0 + slope_look ** 2) ** 1.5
        else:
            # 전방 lookahead 지점이 실제 픽셀 데이터 밖이면 곡률을 모르는
            # 것으로 보고 feedforward 를 0으로 둔다 (틀린 곡률로 미리
            # 꺾는 것보다 안전).
            kappa_right = 0.0

        return float(lateral_m), heading, float(-kappa_right)


class StanleyController:

    def __init__(self, k_cross, k_heading, max_approach, ff_gain, v_ref,
                 wheelbase_m, max_steer_rad, out_alpha):
        self.k_cross = k_cross
        self.k_heading = k_heading
        self.max_approach = max_approach
        self.ff_gain = ff_gain
        self.v_ref = max(v_ref, 0.05)
        self.L = wheelbase_m
        self.max_steer = max_steer_rad
        self.out_alpha = out_alpha
        self.reset()

    def reset(self):
        self._prev_steer = None

    def compute_steering(self, lateral_error_m, heading_error_rad, curvature_left):
        approach = float(np.clip(
            np.arctan(self.k_cross * lateral_error_m / self.v_ref),
            -self.max_approach, self.max_approach))

        ff = self.ff_gain * float(np.arctan(self.L * curvature_left))

        raw = self.k_heading * (heading_error_rad + approach) + ff
        steer = float(np.clip(raw, -self.max_steer, self.max_steer))

        if self._prev_steer is not None:
            steer = (self.out_alpha * steer
                     + (1.0 - self.out_alpha) * self._prev_steer)
        self._prev_steer = steer
        return steer

    def steer_to_angular_velocity(self, steer_rad, linear_v):
        return float(linear_v * np.tan(steer_rad) / self.L)


class LineTracingNode(Node):

    def __init__(self):
        super().__init__('line_tracing_node')

        self._declare_parameters()
        self.p = self._read_parameters()

        self.bridge = CvBridge()
        self.detector = LaneDetector(self.p)

        self.controller = StanleyController(
            k_cross=self.p['pid_kp'],
            k_heading=self.p['heading_gain'],
            max_approach=self.p['max_approach_rad'],
            ff_gain=self.p['curvature_ff_gain'],
            v_ref=self.p['linear_speed_mps'],
            wheelbase_m=self.p['wheelbase_m'],
            max_steer_rad=self.p['max_steer_rad'],
            out_alpha=self.p['steer_lpf_alpha'],
        )

        self._lane_ok = False
        self._lane_cmd = None
        self._last_frame_time = None
        self._lane_lost_since = None
        self._last_valid_cmd = Twist()

        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)

        self.image_sub = self.create_subscription(
            Image, self.p['camera_topic'], self._image_callback, qos)
        self.cmd_pub = self.create_publisher(Twist, self.p['cmd_vel_topic'], 10)
        self.lane_detected_pub = self.create_publisher(
            Bool, self.p['lane_detected_topic'], 10)

        if self.p['publish_debug_image']:
            self.debug_pub = self.create_publisher(
                Image, self.p['debug_image_topic'], 1)
            self.roi_pub = self.create_publisher(
                Image, self.p['roi_image_topic'], 1)
        else:
            self.debug_pub = None
            self.roi_pub = None

        self.control_timer = self.create_timer(
            1.0 / self.p['control_rate_hz'], self._control_loop)

        self.get_logger().info(
            'line_tracing_node v3 시작 | lane_width_m=%.2f (0이면 xm_per_pix 고정) '
            '| pixel_aspect=%.2f' % (self.p['lane_width_m'], self.p['pixel_aspect']))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _declare_parameters(self):
        desc = ParameterDescriptor()
        for name, default in PARAMS:
            self.declare_parameter(name, default, desc)

    def _read_parameters(self):
        values = {name: self.get_parameter(name).value for name, _ in PARAMS}
        f = values['ipm_src_points']
        values['ipm_src_points'] = [(f[0], f[1]), (f[2], f[3]),
                                    (f[4], f[5]), (f[6], f[7])]
        return values

    # ---- 카메라 콜백: 프레임당 1회 인식 + 제어 계산 --------------
    def _image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().warn(f'cv_bridge 변환 실패: {e}')
            return

        self._last_frame_time = self._now()

        bird_eye = self.detector.apply_roi_and_ipm(frame)
        binary = self.detector.threshold_lane(bird_eye)
        res = self.detector.detect(binary)

        lane_detected = res.center_fit is not None
        detected_msg = Bool()
        detected_msg.data = bool(lane_detected)
        self.lane_detected_pub.publish(detected_msg)

        if lane_detected:
            lateral, heading, kappa = self.detector.compute_path_errors(
                res.center_fit, binary.shape, res.y_range)

            steer = self.controller.compute_steering(lateral, heading, kappa)
            v = self._scheduled_speed(steer)
            omega = self.controller.steer_to_angular_velocity(steer, v)

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self._lane_cmd = cmd
            self._lane_ok = True

            self.get_logger().info(
                f'mode={res.mode} | lat={lateral:+.3f} m | '
                f'head={heading:+.3f} rad | kappa={kappa:+.3f} 1/m | '
                f'steer={steer:+.3f} rad | v={v:.2f} | xm={self.detector.xm:.5f}',
                throttle_duration_sec=0.5)

            self._annotate(res.debug_img, lateral, heading, kappa, steer, v)
        else:
            self._lane_ok = False

        if self.debug_pub is not None:
            self._publish_image(self.debug_pub, res.debug_img)
            self._publish_image(self.roi_pub, self.detector.draw_roi(frame))

    def _scheduled_speed(self, steer):
        v0 = self.p['linear_speed_mps']
        ratio = 1.0 - (self.p['curve_slowdown_gain']
                       * abs(steer) / self.p['max_steer_rad'])
        ratio = float(np.clip(ratio, self.p['min_speed_ratio'], 1.0))
        return v0 * ratio

    def _annotate(self, img, lateral, heading, kappa, steer, v):
        h, w = img.shape[:2]
        cx = int(w / 2 + self.p['vehicle_center_offset_px'])
        y_near = int(h * (1.0 - self.p['near_ratio']))
        y_look = int(h * (1.0 - self.p['lookahead_ratio']))
        cv2.line(img, (cx, 0), (cx, h), (255, 255, 255), 1)
        cv2.line(img, (0, y_near), (w, y_near), (0, 255, 255), 1)
        cv2.line(img, (0, y_look), (w, y_look), (255, 0, 255), 1)
        cv2.putText(img, f'lat {lateral:+.3f} head {heading:+.2f} k {kappa:+.2f}',
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f'steer {steer:+.2f} v {v:.2f}',
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # ---- 타이머: 일정 주기 발행 + fail-safe ---------------------
    def _control_loop(self):
        if self._last_frame_time is None:
            return
        now = self._now()

        if now - self._last_frame_time > self.p['frame_timeout_sec']:
            self.cmd_pub.publish(Twist())
            return

        if self._lane_ok and self._lane_cmd is not None:
            self._lane_lost_since = None
            self._last_valid_cmd = self._lane_cmd
            self.cmd_pub.publish(self._lane_cmd)
        else:
            self._handle_lane_loss(now)

    def _handle_lane_loss(self, now):
        if self._lane_lost_since is None:
            self._lane_lost_since = now
            self.controller.reset()

        lost = now - self._lane_lost_since
        last = self._last_valid_cmd

        if lost < self.p['failsafe_hold_sec']:
            cmd = last
        elif lost < self.p['failsafe_creep_sec']:
            v = self.p['failsafe_creep_speed_mps']
            cmd = Twist()
            cmd.linear.x = v
            if abs(last.linear.x) > 1e-3:
                cmd.angular.z = last.angular.z * v / last.linear.x
        else:
            cmd = Twist()

        self.cmd_pub.publish(cmd)

    def _publish_image(self, pub, img):
        try:
            pub.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
        except CvBridgeError as e:
            self.get_logger().warn(f'디버그 이미지 발행 실패: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LineTracingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()