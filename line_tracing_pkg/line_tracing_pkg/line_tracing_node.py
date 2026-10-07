#!/usr/bin/env python3

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

    ('ipm_src_points',
     [0.35, 0.62, 0.65, 0.62, 0.95, 0.95, 0.05, 0.95]),
    ('ipm_dst_margin_px', 150),

    # Scale
    ('lane_width_m', 0.40),
    ('xm_per_pix', 0.003),
    ('pixel_aspect', 1.0),

    # ---- 차선 색 검출 ----
    # threshold_method:
    #   'hsv'          : HSV 마스크만 사용 (기존 방식)
    #   'lab'          : LAB b* 채널 마스크만 사용
    #   'hsv_or_lab'   : 둘 중 하나라도 노랗다고 하면 채택 (합집합, 기본값)
    #                    -> 재현율UP, 회전 중 픽셀이 끊기는 것을 줄이는 데 유리
    #   'hsv_and_lab'  : 둘 다 노랗다고 해야 채택 (교집합)
    #                    -> 노이즈가 많을 때 정밀도UP
    #
    # ★ 기본값을 다시 'hsv'로: 새 트랙(arena_ground)에서 LAB 조합이
    #   벽/연석 같은 밝은 면을 "노란색"으로 잘못 잡는 현상이 실측으로
    #   확인됨. debug_image에서 차선이 아닌 넓은 면이 하얗게 잡히면
    #   그건 LAB(or HSV) 임계값이 아직 이 트랙에 맞지 않다는 뜻이다.
    #   충분히 확인 후에만 'hsv_or_lab'/'lab'로 바꿀 것.
    ('threshold_method', 'hsv'),

    ('hsv_yellow_lower', [15, 80, 80]),
    ('hsv_yellow_upper', [35, 255, 255]),

    # LAB: L(밝기) 0~255, a(초록-빨강), b(파랑-노랑). b값이 클수록 노란색.
    # 아스팔트/그림자는 b가 낮고, 노란 차선은 b가 뚜렷하게 높다.
    ('lab_b_min', 145),
    ('lab_b_max', 255),
    ('lab_l_min', 60),     # 너무 어두운(그림자 속 잡광) 영역 배제
    ('lab_l_max', 255),

    ('morph_kernel_size', 5),
    ('morph_iterations', 1),

    # Sliding window
    ('histogram_rows_ratio', 0.35),
    ('histogram_smooth_px', 41),
    ('peak_suppress_px', 150),

    ('sliding_window_count', 12),
    ('sliding_window_margin_px', 60),
    ('sliding_window_min_pixels', 40),
    ('sliding_window_min_pixels_base', 500),
    ('sliding_window_min_total_pixels', 200),

    ('min_lane_span_ratio', 0.25),

    # 근접 데이터 없는 fit은 신뢰하지 않는다
    ('near_coverage_min_ratio', 0.55),

    # ★ "두 차선" 쌍으로 인정하기 위한 폭 검증.
    # 학습된 차선 폭(_lane_width_px)이 있으면 그 값의 [lo, hi]배 안에
    # 들어야 하고, 아직 학습 전이면 최소 절대폭(lane_width_bootstrap_min_px)
    # 이상이어야 한다. 이게 없으면 벽/연석 같은 큰 오검출 덩어리의
    # 양쪽 가장자리를 "두 차선"으로 잘못 받아들이게 된다 (실측으로 확인됨).
    ('lane_width_ratio_lo', 0.55),
    ('lane_width_ratio_hi', 1.8),
    ('lane_width_bootstrap_min_px', 150),

    # Error evaluation
    ('near_ratio', 0.10),
    ('lookahead_ratio', 0.35),
    ('vehicle_center_offset_px', 0.0),

    ('lane_side', 'right'),
    ('lane_offset_m', 0.20),

    # Stanley
    ('pid_kp', 1.5),
    ('heading_gain', 2.0),
    ('max_approach_rad', 0.6),
    ('curvature_ff_gain', 1.0),
    ('steer_lpf_alpha', 0.7),

    ('wheelbase_m', 0.2),
    ('max_steer_rad', 0.5236),

    # Speed
    ('linear_speed_mps', 0.3),
    ('curve_slowdown_gain', 0.5),
    ('min_speed_ratio', 0.6),

    ('control_rate_hz', 20.0),
    ('frame_timeout_sec', 0.5),

    # Fail-safe
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

    # 실제 픽셀이 중앙경로를 뒷받침하는 범위
    y_range: tuple = (0.0, 0.0)


class LaneDetector:

    def __init__(self, params):
        self.p = params
        self.aspect = float(params['pixel_aspect'])

        self._lane_width_px = None

        self._prev_x = {
            'left': None,
            'right': None,
        }

        self._age = {
            'left': 0,
            'right': 0,
        }

        self._size = None

        self._build_ipm_matrix(
            params['image_width'],
            params['image_height']
        )

    # ---------------------------------------------------------
    # Scale
    # ---------------------------------------------------------

    @property
    def xm(self):

        if (
            self.p['lane_width_m'] > 0.0
            and self._lane_width_px
        ):
            return (
                self.p['lane_width_m']
                / self._lane_width_px
            )

        return self.p['xm_per_pix']

    @property
    def ym(self):
        return self.xm * self.aspect

    # ---------------------------------------------------------
    # IPM
    # ---------------------------------------------------------

    def _build_ipm_matrix(self, w, h):

        src = np.float32([
            [x * w, y * h]
            for x, y in self.p['ipm_src_points']
        ])

        m = self.p['ipm_dst_margin_px']

        dst = np.float32([
            [m, 0],
            [w - m, 0],
            [w - m, h],
            [m, h],
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)

        self._src_px = src.astype(np.int32)

        self._size = (w, h)

    def apply_roi_and_ipm(self, bgr_img):

        h, w = bgr_img.shape[:2]

        if (w, h) != self._size:
            self._build_ipm_matrix(w, h)

        return cv2.warpPerspective(
            bgr_img,
            self.M,
            (w, h),
            flags=cv2.INTER_LINEAR,
        )

    def draw_roi(self, bgr_img):

        out = bgr_img.copy()

        cv2.polylines(
            out,
            [self._src_px.reshape(-1, 1, 2)],
            True,
            (0, 255, 255),
            2,
        )

        return out

    # ---------------------------------------------------------
    # 색 검출: HSV / LAB / 조합
    # ---------------------------------------------------------

    def _hsv_mask(self, blurred_bgr):

        hsv = cv2.cvtColor(
            blurred_bgr,
            cv2.COLOR_BGR2HSV,
        )

        lower = np.array(
            self.p['hsv_yellow_lower'],
            dtype=np.uint8,
        )

        upper = np.array(
            self.p['hsv_yellow_upper'],
            dtype=np.uint8,
        )

        return cv2.inRange(hsv, lower, upper)

    def _lab_mask(self, blurred_bgr):

        lab = cv2.cvtColor(
            blurred_bgr,
            cv2.COLOR_BGR2LAB,
        )

        l_ch, a_ch, b_ch = cv2.split(lab)

        b_ok = cv2.inRange(
            b_ch,
            int(self.p['lab_b_min']),
            int(self.p['lab_b_max']),
        )

        l_ok = cv2.inRange(
            l_ch,
            int(self.p['lab_l_min']),
            int(self.p['lab_l_max']),
        )

        return cv2.bitwise_and(b_ok, l_ok)

    def threshold_lane(self, bird_eye_bgr):

        blur = cv2.GaussianBlur(
            bird_eye_bgr,
            (5, 5),
            0,
        )

        method = self.p['threshold_method']

        if method == 'hsv':
            binary = self._hsv_mask(blur)

        elif method == 'lab':
            binary = self._lab_mask(blur)

        elif method == 'hsv_and_lab':
            binary = cv2.bitwise_and(
                self._hsv_mask(blur),
                self._lab_mask(blur),
            )

        else:
            # 기본값: 'hsv_or_lab' (알 수 없는 값이 들어와도 안전하게 이걸로)
            binary = cv2.bitwise_or(
                self._hsv_mask(blur),
                self._lab_mask(blur),
            )

        k = self.p['morph_kernel_size']

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k),
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self.p['morph_iterations'],
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        return binary

    # ---------------------------------------------------------
    # Histogram peak
    # ---------------------------------------------------------

    @staticmethod
    def _peak(hist):

        m = float(hist.max())

        if m <= 0.0:
            return 0, 0.0

        thr = 0.98 * m

        i = int(np.argmax(hist))

        lo = i
        hi = i

        while (
            lo > 0
            and hist[lo - 1] >= thr
        ):
            lo -= 1

        while (
            hi < len(hist) - 1
            and hist[hi + 1] >= thr
        ):
            hi += 1

        return (lo + hi) // 2, m

    def _find_bases(self, binary):

        h, w = binary.shape[:2]

        y0 = int(
            h * (
                1.0
                - self.p['histogram_rows_ratio']
            )
        )

        hist = (
            binary[y0:, :] > 0
        ).sum(axis=0).astype(np.float32)

        k = max(
            1,
            int(self.p['histogram_smooth_px'])
        )

        hist = np.convolve(
            hist,
            np.ones(k, dtype=np.float32),
            mode='same',
        )

        min_base = (
            self.p['sliding_window_min_pixels_base']
        )

        sup = int(
            self.p['peak_suppress_px']
        )

        bases = []

        for _ in range(2):

            x, v = self._peak(hist)

            if v < min_base:
                break

            bases.append(x)

            hist[
                max(0, x - sup):
                min(w, x + sup + 1)
            ] = 0.0

        return bases

    # ---------------------------------------------------------
    # Sliding window
    # ---------------------------------------------------------

    def _fit_one_lane(
        self,
        binary,
        base_x,
        nz_x,
        nz_y,
        debug_img,
    ):

        h, w = binary.shape[:2]

        n_windows = (
            self.p['sliding_window_count']
        )

        win_h = h // n_windows

        margin = (
            self.p['sliding_window_margin_px']
        )

        min_pix = (
            self.p['sliding_window_min_pixels']
        )

        cur_x = int(base_x)

        shift = 0.0

        inds = []

        for i in range(n_windows):

            y_low = h - (i + 1) * win_h
            y_high = h - i * win_h

            x_low = max(
                0,
                cur_x - margin,
            )

            x_high = min(
                w,
                cur_x + margin,
            )

            cv2.rectangle(
                debug_img,
                (x_low, y_low),
                (x_high, y_high),
                (0, 255, 0),
                1,
            )

            good = (
                (
                    (nz_y >= y_low)
                    & (nz_y < y_high)
                    & (nz_x >= x_low)
                    & (nz_x < x_high)
                )
                .nonzero()[0]
            )

            inds.append(good)

            if len(good) > min_pix:

                new_x = int(
                    np.mean(nz_x[good])
                )

                shift = (
                    0.5 * shift
                    + 0.5 * (new_x - cur_x)
                )

                cur_x = new_x

            else:

                shift *= 0.5

                cur_x = int(
                    np.clip(
                        cur_x + shift,
                        0,
                        w - 1,
                    )
                )

        if inds:
            inds = np.concatenate(inds)
        else:
            inds = np.array(
                [],
                dtype=np.int64,
            )

        n_pixels = len(inds)

        if (
            n_pixels
            < self.p[
                'sliding_window_min_total_pixels'
            ]
        ):
            return None, n_pixels, h, 0.0

        x_pts = nz_x[inds]
        y_pts = nz_y[inds]

        y_lo = float(y_pts.min())
        y_hi = float(y_pts.max())

        if (
            y_hi - y_lo
            < self.p['min_lane_span_ratio'] * h
        ):
            return None, n_pixels, y_lo, y_hi

        try:

            fit = np.polyfit(
                y_pts,
                x_pts,
                2,
            )

        except (
            np.linalg.LinAlgError,
            TypeError,
            ValueError,
        ):

            return None, n_pixels, y_lo, y_hi

        return (
            fit,
            n_pixels,
            y_lo,
            y_hi,
        )

    # ---------------------------------------------------------
    # Lane width
    # ---------------------------------------------------------

    def _measure_width(
        self,
        lf,
        rf,
        w,
        h,
    ):

        y = np.linspace(
            h * 0.5,
            h - 1,
            40,
        )

        dx_px = (
            _poly(rf, y)
            - _poly(lf, y)
        )

        valid = (
            (dx_px > 30.0)
            & (dx_px < float(w))
        )

        if not np.any(valid):
            return None, None, 0.0

        slope = (
            (
                (lf[0] + rf[0]) * y
                + (lf[1] + rf[1]) / 2.0
            )
            / self.aspect
        )

        width_perp = (
            dx_px
            / np.sqrt(
                1.0 + slope ** 2
            )
        )

        return (
            float(
                np.median(
                    width_perp[valid]
                )
            ),
            float(
                np.median(
                    dx_px[valid]
                )
            ),
            float(
                np.median(
                    slope[valid]
                )
            ),
        )

    # ---------------------------------------------------------
    # 한쪽 차선만 있을 때: 실제 픽셀이 있었던 y_lo~y_hi 에서만
    # offset 중앙경로를 생성한다 (그 밖은 외삽하지 않음).
    # ---------------------------------------------------------

    def _offset_fit(
        self,
        fit,
        offset_m,
        h,
        y_range=None,
    ):

        xm = self.xm
        ym = self.ym

        if y_range is None:

            y_lo = 0.0
            y_hi = float(h - 1)

        else:

            y_lo = float(
                np.clip(
                    y_range[0],
                    0.0,
                    h - 1.0,
                )
            )

            y_hi = float(
                np.clip(
                    y_range[1],
                    0.0,
                    h - 1.0,
                )
            )

            if y_lo > y_hi:
                y_lo, y_hi = y_hi, y_lo

        if y_hi - y_lo < 2.0:
            return None

        y = np.linspace(
            y_lo,
            y_hi,
            30,
        )

        x = _poly(
            fit,
            y,
        )

        dxdy = (
            2.0 * fit[0] * y
            + fit[1]
        )

        tx = xm * dxdy
        ty = ym

        norm = np.maximum(
            np.hypot(tx, ty),
            1e-9,
        )

        nx = ty / norm
        ny = -tx / norm

        X = (
            x * xm
            + offset_m * nx
        )

        Y = (
            y * ym
            + offset_m * ny
        )

        y_center = Y / ym
        x_center = X / xm

        try:

            center_fit = np.polyfit(
                y_center,
                x_center,
                2,
            )

        except (
            np.linalg.LinAlgError,
            TypeError,
            ValueError,
        ):

            return None

        center_range = (
            float(np.min(y_center)),
            float(np.max(y_center)),
        )

        return (
            center_fit,
            center_range,
        )

    # ---------------------------------------------------------
    # Single lane side classification
    # ---------------------------------------------------------

    def _classify_single(
        self,
        fit,
        h,
        w,
    ):

        y = h * 0.8

        x = float(
            _poly(
                fit,
                y,
            )
        )

        dl = (
            abs(x - self._prev_x['left'])
            if self._prev_x['left'] is not None
            else 1e9
        )

        dr = (
            abs(x - self._prev_x['right'])
            if self._prev_x['right'] is not None
            else 1e9
        )

        if min(dl, dr) < w * 0.25:

            return (
                'left'
                if dl < dr
                else 'right'
            )

        return (
            'left'
            if x < w / 2
            else 'right'
        )

    # ---------------------------------------------------------
    # Draw polynomial
    # ---------------------------------------------------------

    @staticmethod
    def _draw_fit(
        img,
        fit,
        plot_y,
        color,
        thickness,
    ):

        if fit is None:
            return

        h, w = img.shape[:2]

        xs = np.clip(
            _poly(fit, plot_y),
            0,
            w - 1,
        )

        pts = np.stack(
            [xs, plot_y],
            axis=1,
        ).astype(np.int32)

        cv2.polylines(
            img,
            [pts],
            False,
            color,
            thickness,
        )

    # ---------------------------------------------------------
    # Detect
    # ---------------------------------------------------------

    def detect(self, binary):

        h, w = binary.shape[:2]

        debug = cv2.cvtColor(
            binary,
            cv2.COLOR_GRAY2BGR,
        )

        nz_y, nz_x = binary.nonzero()

        bases = self._find_bases(binary)

        candidates = []

        for base in bases:

            fit, count, y_lo, y_hi = (
                self._fit_one_lane(
                    binary,
                    base,
                    nz_x,
                    nz_y,
                    debug,
                )
            )

            if fit is None:
                continue

            near_required = (
                h
                * self.p[
                    'near_coverage_min_ratio'
                ]
            )

            if y_hi < near_required:
                continue

            candidates.append(
                (
                    fit,
                    count,
                    y_lo,
                    y_hi,
                )
            )

        candidates.sort(
            key=lambda c:
            _poly(c[0], h * 0.8)
        )

        left_fit = None
        right_fit = None

        l_range = None
        r_range = None

        # ---------------------------------------------
        # 두 차선
        # ---------------------------------------------

        if len(candidates) >= 2:

            lf = candidates[0][0]
            rf = candidates[-1][0]

            lf_range = (
                candidates[0][2],
                candidates[0][3],
            )

            rf_range = (
                candidates[-1][2],
                candidates[-1][3],
            )

            width_perp, width_x, _ = (
                self._measure_width(
                    lf,
                    rf,
                    w,
                    h,
                )
            )

            # ★ 학습된 차선 폭과 비교해 그럴듯한 폭인지 검증한다.
            #   (벽/연석 같은 큰 오검출 덩어리의 양쪽 가장자리가
            #   "두 차선"으로 둔갑하는 것을 막는다.)
            if self._lane_width_px is not None:
                width_ok = (
                    width_perp is not None
                    and self.p['lane_width_ratio_lo']
                        * self._lane_width_px
                        <= width_perp
                    <= self.p['lane_width_ratio_hi']
                        * self._lane_width_px
                )
            else:
                width_ok = (
                    width_perp is not None
                    and width_perp
                        >= self.p['lane_width_bootstrap_min_px']
                    and width_perp < w
                )

            if width_ok:

                left_fit = lf
                right_fit = rf

                l_range = lf_range
                r_range = rf_range

                self._prev_x['left'] = float(
                    _poly(
                        lf,
                        h * 0.8,
                    )
                )

                self._prev_x['right'] = float(
                    _poly(
                        rf,
                        h * 0.8,
                    )
                )

                self._age['left'] = 0
                self._age['right'] = 0

                # 거의 직선일 때만 차선 폭 학습
                if (
                    abs(lf[0]) < 5e-4
                    and abs(rf[0]) < 5e-4
                ):

                    if self._lane_width_px is None:

                        self._lane_width_px = (
                            width_perp
                        )

                    else:

                        self._lane_width_px = (
                            0.95
                            * self._lane_width_px
                            + 0.05
                            * width_perp
                        )

            else:

                # 디버그용: 두 후보가 있었지만 폭이 말이 안 돼서 버렸다는 표시.
                # 이게 자주 뜬다면 색 임계값이 벽/바닥 등을 오검출하고
                # 있다는 신호다.
                wtxt = (
                    f'{width_perp:.0f}px'
                    if width_perp is not None
                    else '?'
                )

                cv2.putText(
                    debug,
                    f'reject dual: bad width {wtxt}',
                    (15, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )

        # ---------------------------------------------
        # 한쪽 차선
        # ---------------------------------------------

        if (
            left_fit is None
            and right_fit is None
            and len(candidates) >= 1
        ):

            best = max(
                candidates,
                key=lambda c: c[1],
            )

            fit = best[0]

            one_range = (
                best[2],
                best[3],
            )

            side = self._classify_single(
                fit,
                h,
                w,
            )

            if side == 'left':

                left_fit = fit
                l_range = one_range

                self._prev_x['left'] = float(
                    _poly(
                        fit,
                        h * 0.8,
                    )
                )

                self._age['left'] = 0

            else:

                right_fit = fit
                r_range = one_range

                self._prev_x['right'] = float(
                    _poly(
                        fit,
                        h * 0.8,
                    )
                )

                self._age['right'] = 0

        # Age update
        for side in ('left', 'right'):

            present = (
                left_fit is not None
                if side == 'left'
                else right_fit is not None
            )

            if not present:

                self._age[side] += 1

                if self._age[side] > 30:
                    self._prev_x[side] = None

        plot_y = np.linspace(
            0,
            h - 1,
            h,
        )

        self._draw_fit(
            debug,
            left_fit,
            plot_y,
            (255, 0, 0),
            3,
        )

        self._draw_fit(
            debug,
            right_fit,
            plot_y,
            (0, 0, 255),
            3,
        )

        # ---------------------------------------------
        # Center line
        # ---------------------------------------------

        center_fit = None
        y_range = (0.0, 0.0)
        mode = 'none'

        width_px = self._lane_width_px

        if (
            self.p['lane_width_m'] > 0.0
            and width_px
        ):

            half_w = (
                self.p['lane_width_m']
                / 2.0
            )

        else:

            half_w = (
                self.p['lane_offset_m']
            )

        # ---------------------------------------------
        # 양쪽 차선
        # ---------------------------------------------

        if (
            left_fit is not None
            and right_fit is not None
        ):

            center_fit = (
                left_fit + right_fit
            ) / 2.0

            mode = 'dual'

            if (
                l_range is not None
                and r_range is not None
            ):

                y_range = (
                    max(
                        l_range[0],
                        r_range[0],
                    ),
                    min(
                        l_range[1],
                        r_range[1],
                    ),
                )

        # ---------------------------------------------
        # 왼쪽 차선 하나
        # ---------------------------------------------

        elif left_fit is not None:

            result = self._offset_fit(
                left_fit,
                +half_w,
                h,
                l_range,
            )

            if result is not None:

                center_fit, y_range = result
                mode = 'left_only'

        # ---------------------------------------------
        # 오른쪽 차선 하나
        # ---------------------------------------------

        elif right_fit is not None:

            result = self._offset_fit(
                right_fit,
                -half_w,
                h,
                r_range,
            )

            if result is not None:

                center_fit, y_range = result
                mode = 'right_only'

        # ---------------------------------------------
        # Debug
        # ---------------------------------------------

        if center_fit is not None:

            self._draw_fit(
                debug,
                center_fit,
                plot_y,
                (0, 255, 0),
                4,
            )

            if width_px is None:

                label = mode

            else:

                label = (
                    f'{mode} '
                    f'w={width_px:.0f}px'
                )

            cv2.putText(
                debug,
                label,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return LaneResult(
            left_fit,
            right_fit,
            center_fit,
            mode,
            debug,
            width_px,
            y_range,
        )

    # ---------------------------------------------------------
    # 데이터 범위 밖에서는 2차 다항식 외삽 금지
    # ---------------------------------------------------------

    @staticmethod
    def _bounded_x_slope(
        fit,
        y,
        y_lo,
        y_hi,
    ):

        y_eval = float(
            np.clip(
                y,
                y_lo,
                y_hi,
            )
        )

        x0 = float(
            _poly(
                fit,
                y_eval,
            )
        )

        slope = float(
            2.0
            * fit[0]
            * y_eval
            + fit[1]
        )

        if y_lo <= y <= y_hi:

            return (
                x0,
                slope,
                True,
            )

        x = (
            x0
            + slope
            * (float(y) - y_eval)
        )

        return (
            float(x),
            slope,
            False,
        )

    # ---------------------------------------------------------
    # Path error
    # ---------------------------------------------------------

    def compute_path_errors(
        self,
        center_fit,
        image_shape,
        y_range=None,
    ):

        h, w = image_shape[:2]

        xm = self.xm
        ym = self.ym

        ratio = xm / ym

        if y_range is None:

            y_range = (
                0.0,
                float(h - 1),
            )

        y_lo, y_hi = y_range

        y_lo = float(
            np.clip(
                y_lo,
                0.0,
                h - 1.0,
            )
        )

        y_hi = float(
            np.clip(
                y_hi,
                0.0,
                h - 1.0,
            )
        )

        if y_lo > y_hi:
            y_lo, y_hi = y_hi, y_lo

        y_near = (
            h
            * (
                1.0
                - self.p['near_ratio']
            )
        )

        (
            x_near,
            slope_near_px,
            _
        ) = self._bounded_x_slope(
            center_fit,
            y_near,
            y_lo,
            y_hi,
        )

        slope_near = (
            slope_near_px
            * ratio
        )

        heading = float(
            np.arctan(
                slope_near
            )
        )

        center_px = (
            w / 2.0
            + self.p[
                'vehicle_center_offset_px'
            ]
        )

        lateral_m = (
            (center_px - x_near)
            * xm
            / np.sqrt(
                1.0
                + slope_near ** 2
            )
        )

        y_look = (
            h
            * (
                1.0
                - self.p[
                    'lookahead_ratio'
                ]
            )
        )

        (
            _,
            slope_look_px,
            in_range,
        ) = self._bounded_x_slope(
            center_fit,
            y_look,
            y_lo,
            y_hi,
        )

        slope_look = (
            slope_look_px
            * ratio
        )

        if in_range:

            d2X_dY2 = (
                2.0
                * center_fit[0]
                * ratio
                / ym
            )

            kappa_right = (
                d2X_dY2
                / (
                    1.0
                    + slope_look ** 2
                ) ** 1.5
            )

        else:

            kappa_right = 0.0

        return (
            float(lateral_m),
            heading,
            float(-kappa_right),
        )


class StanleyController:

    def __init__(
        self,
        k_cross,
        k_heading,
        max_approach,
        ff_gain,
        v_ref,
        wheelbase_m,
        max_steer_rad,
        out_alpha,
    ):

        self.k_cross = k_cross
        self.k_heading = k_heading
        self.max_approach = max_approach
        self.ff_gain = ff_gain

        self.v_ref = max(
            v_ref,
            0.05,
        )

        self.L = wheelbase_m
        self.max_steer = max_steer_rad
        self.out_alpha = out_alpha

        self.reset()

    def reset(self):
        self._prev_steer = None

    def compute_steering(
        self,
        lateral_error_m,
        heading_error_rad,
        curvature_left,
    ):

        approach = float(
            np.clip(
                np.arctan(
                    self.k_cross
                    * lateral_error_m
                    / self.v_ref
                ),
                -self.max_approach,
                self.max_approach,
            )
        )

        ff = (
            self.ff_gain
            * float(
                np.arctan(
                    self.L
                    * curvature_left
                )
            )
        )

        raw = (
            self.k_heading
            * (
                heading_error_rad
                + approach
            )
            + ff
        )

        steer = float(
            np.clip(
                raw,
                -self.max_steer,
                self.max_steer,
            )
        )

        if self._prev_steer is not None:

            steer = (
                self.out_alpha
                * steer
                + (
                    1.0
                    - self.out_alpha
                )
                * self._prev_steer
            )

        self._prev_steer = steer

        return steer

    def steer_to_angular_velocity(
        self,
        steer_rad,
        linear_v,
    ):

        return float(
            linear_v
            * np.tan(steer_rad)
            / self.L
        )


class LineTracingNode(Node):

    def __init__(self):

        super().__init__(
            'line_tracing_node'
        )

        self._declare_parameters()

        self.p = (
            self._read_parameters()
        )

        self.bridge = CvBridge()

        self.detector = LaneDetector(
            self.p
        )

        self.controller = (
            StanleyController(
                k_cross=self.p['pid_kp'],
                k_heading=self.p[
                    'heading_gain'
                ],
                max_approach=self.p[
                    'max_approach_rad'
                ],
                ff_gain=self.p[
                    'curvature_ff_gain'
                ],
                v_ref=self.p[
                    'linear_speed_mps'
                ],
                wheelbase_m=self.p[
                    'wheelbase_m'
                ],
                max_steer_rad=self.p[
                    'max_steer_rad'
                ],
                out_alpha=self.p[
                    'steer_lpf_alpha'
                ],
            )
        )

        self._lane_ok = False
        self._lane_cmd = None

        self._last_frame_time = None

        self._lane_lost_since = None

        self._last_valid_cmd = Twist()

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.image_sub = (
            self.create_subscription(
                Image,
                self.p['camera_topic'],
                self._image_callback,
                qos,
            )
        )

        self.cmd_pub = (
            self.create_publisher(
                Twist,
                self.p['cmd_vel_topic'],
                10,
            )
        )

        self.lane_detected_pub = (
            self.create_publisher(
                Bool,
                self.p[
                    'lane_detected_topic'
                ],
                10,
            )
        )

        if self.p['publish_debug_image']:

            self.debug_pub = (
                self.create_publisher(
                    Image,
                    self.p[
                        'debug_image_topic'
                    ],
                    1,
                )
            )

            self.roi_pub = (
                self.create_publisher(
                    Image,
                    self.p[
                        'roi_image_topic'
                    ],
                    1,
                )
            )

        else:

            self.debug_pub = None
            self.roi_pub = None

        self.control_timer = (
            self.create_timer(
                1.0
                / self.p[
                    'control_rate_hz'
                ],
                self._control_loop,
            )
        )

        self.get_logger().info(
            'line_tracing_node 시작 | '
            f"threshold_method={self.p['threshold_method']}"
        )

    def _now(self):

        return (
            self.get_clock()
            .now()
            .nanoseconds
            * 1e-9
        )

    def _declare_parameters(self):

        desc = ParameterDescriptor()

        for name, default in PARAMS:

            self.declare_parameter(
                name,
                default,
                desc,
            )

    def _read_parameters(self):

        values = {
            name:
            self.get_parameter(name).value

            for name, _ in PARAMS
        }

        f = values[
            'ipm_src_points'
        ]

        values[
            'ipm_src_points'
        ] = [
            (f[0], f[1]),
            (f[2], f[3]),
            (f[4], f[5]),
            (f[6], f[7]),
        ]

        return values

    # ---------------------------------------------------------
    # Camera callback
    # ---------------------------------------------------------

    def _image_callback(self, msg):

        try:

            frame = (
                self.bridge
                .imgmsg_to_cv2(
                    msg,
                    desired_encoding='bgr8',
                )
            )

        except CvBridgeError as e:

            self.get_logger().warn(
                f'cv_bridge 변환 실패: {e}'
            )

            return

        self._last_frame_time = (
            self._now()
        )

        bird_eye = (
            self.detector
            .apply_roi_and_ipm(frame)
        )

        binary = (
            self.detector
            .threshold_lane(bird_eye)
        )

        res = (
            self.detector
            .detect(binary)
        )

        lane_detected = (
            res.center_fit is not None
        )

        detected_msg = Bool()

        detected_msg.data = bool(
            lane_detected
        )

        self.lane_detected_pub.publish(
            detected_msg
        )

        if lane_detected:

            (
                lateral,
                heading,
                kappa,
            ) = (
                self.detector
                .compute_path_errors(
                    res.center_fit,
                    binary.shape,
                    res.y_range,
                )
            )

            steer = (
                self.controller
                .compute_steering(
                    lateral,
                    heading,
                    kappa,
                )
            )

            v = self._scheduled_speed(
                steer
            )

            omega = (
                self.controller
                .steer_to_angular_velocity(
                    steer,
                    v,
                )
            )

            cmd = Twist()

            cmd.linear.x = v
            cmd.angular.z = omega

            self._lane_cmd = cmd
            self._lane_ok = True

            self.get_logger().info(
                f'mode={res.mode} | '
                f'lat={lateral:+.3f} m | '
                f'head={heading:+.3f} rad | '
                f'kappa={kappa:+.3f} 1/m | '
                f'steer={steer:+.3f} rad | '
                f'v={v:.2f} | '
                f'xm={self.detector.xm:.5f}',
                throttle_duration_sec=0.5,
            )

            self._annotate(
                res.debug_img,
                lateral,
                heading,
                kappa,
                steer,
                v,
            )

        else:

            self._lane_ok = False

        if self.debug_pub is not None:

            self._publish_image(
                self.debug_pub,
                res.debug_img,
            )

            self._publish_image(
                self.roi_pub,
                self.detector.draw_roi(
                    frame
                ),
            )

    # ---------------------------------------------------------
    # Speed scheduling
    # ---------------------------------------------------------

    def _scheduled_speed(
        self,
        steer,
    ):

        v0 = self.p[
            'linear_speed_mps'
        ]

        ratio = (
            1.0
            - (
                self.p[
                    'curve_slowdown_gain'
                ]
                * abs(steer)
                / self.p[
                    'max_steer_rad'
                ]
            )
        )

        ratio = float(
            np.clip(
                ratio,
                self.p[
                    'min_speed_ratio'
                ],
                1.0,
            )
        )

        return v0 * ratio

    # ---------------------------------------------------------
    # Debug
    # ---------------------------------------------------------

    def _annotate(
        self,
        img,
        lateral,
        heading,
        kappa,
        steer,
        v,
    ):

        h, w = img.shape[:2]

        cx = int(
            w / 2
            + self.p[
                'vehicle_center_offset_px'
            ]
        )

        y_near = int(
            h
            * (
                1.0
                - self.p[
                    'near_ratio'
                ]
            )
        )

        y_look = int(
            h
            * (
                1.0
                - self.p[
                    'lookahead_ratio'
                ]
            )
        )

        cv2.line(
            img,
            (cx, 0),
            (cx, h),
            (255, 255, 255),
            1,
        )

        cv2.line(
            img,
            (0, y_near),
            (w, y_near),
            (0, 255, 255),
            1,
        )

        cv2.line(
            img,
            (0, y_look),
            (w, y_look),
            (255, 0, 255),
            1,
        )

        cv2.putText(
            img,
            (
                f'lat {lateral:+.3f} '
                f'head {heading:+.2f} '
                f'k {kappa:+.2f}'
            ),
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            (
                f'steer {steer:+.2f} '
                f'v {v:.2f}'
            ),
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ---------------------------------------------------------
    # Control loop
    # ---------------------------------------------------------

    def _control_loop(self):

        if self._last_frame_time is None:
            return

        now = self._now()

        if (
            now
            - self._last_frame_time
            > self.p[
                'frame_timeout_sec'
            ]
        ):

            self.cmd_pub.publish(
                Twist()
            )

            return

        if (
            self._lane_ok
            and self._lane_cmd is not None
        ):

            self._lane_lost_since = None

            self._last_valid_cmd = (
                self._lane_cmd
            )

            self.cmd_pub.publish(
                self._lane_cmd
            )

        else:

            self._handle_lane_loss(
                now
            )

    # ---------------------------------------------------------
    # Fail-safe
    # ---------------------------------------------------------

    def _handle_lane_loss(
        self,
        now,
    ):

        if self._lane_lost_since is None:

            self._lane_lost_since = now

            self.controller.reset()

        lost = (
            now
            - self._lane_lost_since
        )

        last = (
            self._last_valid_cmd
        )

        if (
            lost
            < self.p[
                'failsafe_hold_sec'
            ]
        ):

            cmd = last

        elif (
            lost
            < self.p[
                'failsafe_creep_sec'
            ]
        ):

            v = self.p[
                'failsafe_creep_speed_mps'
            ]

            cmd = Twist()

            cmd.linear.x = v

            if abs(last.linear.x) > 1e-3:

                cmd.angular.z = (
                    last.angular.z
                    * v
                    / last.linear.x
                )

        else:

            cmd = Twist()

        self.cmd_pub.publish(
            cmd
        )

    # ---------------------------------------------------------
    # Publish image
    # ---------------------------------------------------------

    def _publish_image(
        self,
        pub,
        img,
    ):

        try:

            pub.publish(
                self.bridge
                .cv2_to_imgmsg(
                    img,
                    encoding='bgr8',
                )
            )

        except CvBridgeError as e:

            self.get_logger().warn(
                f'디버그 이미지 발행 실패: {e}'
            )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = (
        LineTracingNode()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()