#!/usr/bin/env python3
"""
camera_perception_node.py

역할: 카메라 영상 -> 인지 결과 토픽 발행 (오직 인지만 담당)

  입력  /camera/color/image_raw
  출력  /perception/lane_error   (LaneError)
        /perception/color_zone   (ColorZone)

[중요] 이 노드는 /cmd_vel 을 발행하지 않는다.
       주행 명령은 mission_supervisor_node(판단+상위제어)가 단독으로 발행한다.
       차선 유실 시 대응(fail-safe)도 판단 모듈 책임이며, 여기서는
       valid=false 로 알리기만 한다.

[단위 규약] 모든 출력은 미터/라디안. 픽셀 단위는 이 노드 밖으로 나가지 않는다.
       IPM 변환 후 Bird's-eye view 는 실제 평면과 비례하므로,
       BEV 가 담는 실제 크기(BEV_WIDTH_M/BEV_LENGTH_M)를 알면 px->m 환산이 가능하다.

[부호 규약] LaneError.msg 주석 참조. ROS 표준(y 좌측 양수)에 맞춰
       lateral_offset_m, heading_error_rad 모두 "양수 = 좌회전 필요".
"""

import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ar_msgs.msg import LaneError, ColorZone


class CameraPerceptionNode(Node):

    def __init__(self):
        super().__init__('camera_perception_node')
        self._declare_params()
        self._load_params()

        self.bridge = CvBridge()

        # IPM 변환 행렬 (파라미터에서 1회 계산)
        self.M_ipm = cv2.getPerspectiveTransform(self.ipm_src, self.ipm_dst)

        # BEV 픽셀 -> 실제 거리 환산 계수 [m/px]
        self.m_per_px_x = self.bev_width_m / self.bev_w
        self.m_per_px_y = self.bev_length_m / self.bev_h

        # 색상 구역 Hysteresis 상태
        self.current_zone = ColorZone.ZONE_NONE
        self.zone_candidate = ColorZone.ZONE_NONE
        self.zone_streak = 0

        self.pub_lane = self.create_publisher(LaneError, '/perception/lane_error', 10)
        self.pub_zone = self.create_publisher(ColorZone, '/perception/color_zone', 10)
        # 카메라 드라이버/Gazebo 는 best-effort 로 발행하는 경우가 많다.
        # reliable 로 구독하면 QoS 불일치로 영상이 조용히 안 들어온다(에러 없음).
        self.create_subscription(Image, self.image_topic, self.image_cb,
                                 qos_profile_sensor_data)

        if self.publish_debug:
            self.pub_dbg = self.create_publisher(Image, '/perception/lane_debug', 2)

        self.get_logger().info(
            f'camera_perception_node 시작 | 입력={self.image_topic} | '
            f'BEV {self.bev_width_m}m x {self.bev_length_m}m '
            f'({self.m_per_px_x*1000:.2f} mm/px)')

    # ------------------------------------------------------------------
    # 파라미터
    # ------------------------------------------------------------------
    def _declare_params(self):
        p = self.declare_parameter
        p('image_topic', '/camera/color/image_raw')
        p('publish_debug', True)

        # --- IPM: 원본 영상의 사다리꼴 4점 (좌상,우상,좌하,우하) ---
        # 실제 카메라 장착 각도에 맞춰 반드시 재측정해야 한다.
        p('ipm_src', [220.0, 300.0, 420.0, 300.0, 40.0, 480.0, 600.0, 480.0])

        # --- BEV 출력 크기와 그것이 담는 실제 영역 ---
        # 이 두 쌍이 px<->m 환산의 근거. BEV 가 담는 실제 폭/길이를 정확히 넣을 것.
        p('bev_width_px', 400)
        p('bev_length_px', 400)
        p('bev_width_m', 0.80)     # 좌우로 담는 실제 폭 [m]
        p('bev_length_m', 0.80)    # 전방으로 담는 실제 길이 [m]

        # --- 차선 HSV (노란 실선) ---
        p('lane_hsv_lower', [15, 80, 80])
        p('lane_hsv_upper', [35, 255, 255])

        # --- 실제 트랙 차선 폭 [m]. 한쪽만 보일 때 중앙 추정에 사용 ---
        p('lane_width_m', 0.40)

        # --- 곡률을 평가할 전방 거리 [m] ---
        # 0 이면 로봇 현재 위치의 곡률(=지금 돌고 있는 정도).
        # 양수면 그만큼 앞을 내다본 곡률 -> 제어 쪽에서 커브 전에 미리 감속 가능.
        # nav2 velocity regulation 과 맞물리므로 제어 담당과 값을 합의할 것.
        p('curvature_lookahead_m', 0.30)

        # --- Sliding Window ---
        p('n_windows', 9)
        p('window_margin_px', 50)
        p('min_pixels_recenter', 40)
        p('min_pixels_fit', 300)

        # --- 색상 구역(미션 1) ---
        # ROI 를 범퍼 바로 앞으로 극단적으로 좁힘: 원본 영상 하단 비율
        p('zone_roi_top_ratio', 0.88)
        p('zone_roi_x_margin_ratio', 0.30)
        p('red_hsv1_lower', [0, 110, 80]);   p('red_hsv1_upper', [10, 255, 255])
        p('red_hsv2_lower', [170, 110, 80]); p('red_hsv2_upper', [180, 255, 255])
        p('blue_hsv_lower', [100, 110, 60]); p('blue_hsv_upper', [130, 255, 255])
        # Hysteresis: 진입은 높은 문턱, 이탈은 낮은 문턱
        p('zone_enter_ratio', 0.35)
        p('zone_exit_ratio', 0.15)
        p('zone_confirm_frames', 3)

    def _load_params(self):
        g = lambda n: self.get_parameter(n).value
        self.image_topic = g('image_topic')
        self.publish_debug = g('publish_debug')

        s = g('ipm_src')
        self.ipm_src = np.float32([[s[0], s[1]], [s[2], s[3]], [s[4], s[5]], [s[6], s[7]]])

        self.bev_w = int(g('bev_width_px'))
        self.bev_h = int(g('bev_length_px'))
        self.bev_width_m = float(g('bev_width_m'))
        self.bev_length_m = float(g('bev_length_m'))
        # BEV 목적지: 이미지 전체를 채우는 직사각형
        self.ipm_dst = np.float32([[0, 0], [self.bev_w, 0],
                                   [0, self.bev_h], [self.bev_w, self.bev_h]])

        self.lane_lo = np.array(g('lane_hsv_lower'), dtype=np.uint8)
        self.lane_hi = np.array(g('lane_hsv_upper'), dtype=np.uint8)
        self.lane_width_m = float(g('lane_width_m'))
        self.curv_lookahead_m = float(g('curvature_lookahead_m'))

        self.n_windows = int(g('n_windows'))
        self.win_margin = int(g('window_margin_px'))
        self.min_recenter = int(g('min_pixels_recenter'))
        self.min_fit = int(g('min_pixels_fit'))

        self.zone_top = float(g('zone_roi_top_ratio'))
        self.zone_xm = float(g('zone_roi_x_margin_ratio'))
        self.r1lo = np.array(g('red_hsv1_lower'), np.uint8)
        self.r1hi = np.array(g('red_hsv1_upper'), np.uint8)
        self.r2lo = np.array(g('red_hsv2_lower'), np.uint8)
        self.r2hi = np.array(g('red_hsv2_upper'), np.uint8)
        self.blo = np.array(g('blue_hsv_lower'), np.uint8)
        self.bhi = np.array(g('blue_hsv_upper'), np.uint8)
        self.zone_enter = float(g('zone_enter_ratio'))
        self.zone_exit = float(g('zone_exit_ratio'))
        self.zone_confirm = int(g('zone_confirm_frames'))

    # ------------------------------------------------------------------
    # 메인 콜백
    # ------------------------------------------------------------------
    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')
            return

        stamp = msg.header.stamp

        # ---------- 차선 인식 ----------
        lane_msg, dbg = self.detect_lane(frame)
        lane_msg.header.stamp = stamp
        lane_msg.header.frame_id = 'base_link'
        self.pub_lane.publish(lane_msg)

        # ---------- 색상 구역 ----------
        zone_msg = self.detect_color_zone(frame)
        zone_msg.header.stamp = stamp
        zone_msg.header.frame_id = 'base_link'
        self.pub_zone.publish(zone_msg)

        if self.publish_debug and dbg is not None:
            try:
                self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, 'bgr8'))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 차선 인식: ROI -> IPM -> HSV -> 이진화 -> Morphology -> Sliding Window
    # ------------------------------------------------------------------
    def detect_lane(self, frame):
        out = LaneError()
        out.valid = False
        out.single_lane_estimated = False

        # 1) IPM (ROI 마스킹은 IPM 사다리꼴이 사실상 수행하므로 생략 가능)
        bev = cv2.warpPerspective(frame, self.M_ipm, (self.bev_w, self.bev_h))

        # 2) HSV 이진화
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lane_lo, self.lane_hi)

        # 3) Morphology Opening (잡음 제거 -> 선 두께 복원)
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        # 4) Sliding Window
        left_fit, right_fit, n_left, n_right, dbg = self._sliding_window(mask, bev)

        h = self.bev_h
        y_eval = h - 1                       # 로봇에 가장 가까운 지점

        def poly_x(fit, y):
            return fit[0] * y * y + fit[1] * y + fit[2]

        has_l, has_r = left_fit is not None, right_fit is not None
        half_lane_px = (self.lane_width_m / 2.0) / self.m_per_px_x

        if has_l and has_r:
            center_px = (poly_x(left_fit, y_eval) + poly_x(right_fit, y_eval)) / 2.0
            # 중앙선 다항식 = 좌우 평균
            cfit = (left_fit + right_fit) / 2.0
            conf = min(1.0, (n_left + n_right) / (2.0 * self.min_fit * 2))
        elif has_l:
            center_px = poly_x(left_fit, y_eval) + half_lane_px
            cfit = left_fit.copy()
            cfit[2] += half_lane_px
            out.single_lane_estimated = True
            conf = min(0.6, n_left / (self.min_fit * 2.0))
        elif has_r:
            center_px = poly_x(right_fit, y_eval) - half_lane_px
            cfit = right_fit.copy()
            cfit[2] -= half_lane_px
            out.single_lane_estimated = True
            conf = min(0.6, n_right / (self.min_fit * 2.0))
        else:
            return out, dbg          # valid=False 로 반환. 판단 모듈이 fail-safe 처리.

        # ---------- 픽셀 -> 미터 환산 ----------
        # 양수 = 차선 중앙이 로봇보다 왼쪽 (BEV 에서 x 가 작을수록 왼쪽)
        robot_x_px = self.bev_w / 2.0
        out.lateral_offset_m = float((robot_x_px - center_px) * self.m_per_px_x)

        # 다항식 계수를 미터 기준으로 환산
        #   x_px = a*y_px^2 + b*y_px + c,  x_m = x_px*sx,  y_m = y_px*sy
        #   -> a_m = a*sx/sy^2,  b_m = b*sx/sy
        sx, sy = self.m_per_px_x, self.m_per_px_y
        a_m = cfit[0] * sx / (sy * sy)
        b_m = cfit[1] * sx / sy
        y_eval_m = y_eval * sy

        # 접선 기울기 dx/dy (미터 기준) — 로봇 현재 위치에서 평가
        dxdy = 2.0 * a_m * y_eval_m + b_m
        # BEV 에서 전방은 y 감소 방향. dx/dy>0 이면 전방으로 갈수록 x 감소(=왼쪽).
        # 왼쪽으로 꺾임 = 양수 규약이므로 그대로 사용.
        out.heading_error_rad = float(math.atan(dxdy))

        # 곡률은 전방 주시 지점에서 평가 (제어 쪽이 커브 진입 전 미리 감속하도록)
        # BEV 에서 전방은 y 감소 방향이므로 lookahead 만큼 y 를 줄인다.
        #
        # [부호 유도] 전방거리 s 에 대해 y = -s + const 이므로
        #   d2x/ds2 = +2*a_m  -> a_m>0 이면 x 가 오른쪽으로 가속 = 우커브.
        # heading 은 "좌회전 필요 = 양수" 규약이므로 곡률도 맞추려면 부호를 뒤집는다.
        #   => kappa = -(2*a_m)/denom  (좌커브 양수, angular.z 부호와 동일)
        y_curv_m = max(0.0, y_eval_m - self.curv_lookahead_m)
        dxdy_c = 2.0 * a_m * y_curv_m + b_m
        denom = (1.0 + dxdy_c * dxdy_c) ** 1.5
        out.curvature = float(-(2.0 * a_m) / denom) if denom > 1e-9 else 0.0

        out.confidence = float(max(0.0, min(1.0, conf)))
        out.valid = True

        if dbg is not None:
            cv2.putText(dbg, f'off={out.lateral_offset_m:+.3f}m '
                             f'hdg={math.degrees(out.heading_error_rad):+.1f}deg '
                             f'k={out.curvature:+.2f}',
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if out.single_lane_estimated:
                cv2.putText(dbg, 'SINGLE LANE EST.', (6, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        return out, dbg

    def _sliding_window(self, mask, bev_img):
        h, w = mask.shape
        hist = np.sum(mask[h // 2:, :], axis=0)
        mid = w // 2

        left_base = int(np.argmax(hist[:mid])) if hist[:mid].max() > 0 else mid // 2
        right_base = int(np.argmax(hist[mid:]) + mid) if hist[mid:].max() > 0 else mid + mid // 2

        win_h = h // self.n_windows
        nz = mask.nonzero()
        nzy, nzx = np.array(nz[0]), np.array(nz[1])

        lx, rx = left_base, right_base
        l_inds, r_inds = [], []

        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if self.publish_debug else None

        for i in range(self.n_windows):
            y_lo, y_hi = h - (i + 1) * win_h, h - i * win_h
            xl_lo, xl_hi = lx - self.win_margin, lx + self.win_margin
            xr_lo, xr_hi = rx - self.win_margin, rx + self.win_margin

            if dbg is not None:
                cv2.rectangle(dbg, (xl_lo, y_lo), (xl_hi, y_hi), (0, 255, 0), 1)
                cv2.rectangle(dbg, (xr_lo, y_lo), (xr_hi, y_hi), (0, 255, 0), 1)

            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl_lo) & (nzx < xl_hi)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr_lo) & (nzx < xr_hi)).nonzero()[0]
            l_inds.append(gl)
            r_inds.append(gr)

            if len(gl) > self.min_recenter:
                lx = int(np.mean(nzx[gl]))
            if len(gr) > self.min_recenter:
                rx = int(np.mean(nzx[gr]))

        l_inds = np.concatenate(l_inds) if l_inds else np.array([], dtype=int)
        r_inds = np.concatenate(r_inds) if r_inds else np.array([], dtype=int)

        left_fit = right_fit = None
        if len(l_inds) > self.min_fit:
            left_fit = np.polyfit(nzy[l_inds], nzx[l_inds], 2)
        if len(r_inds) > self.min_fit:
            right_fit = np.polyfit(nzy[r_inds], nzx[r_inds], 2)

        if dbg is not None:
            if len(l_inds) > 0:
                dbg[nzy[l_inds], nzx[l_inds]] = (255, 0, 0)
            if len(r_inds) > 0:
                dbg[nzy[r_inds], nzx[r_inds]] = (0, 0, 255)

        return left_fit, right_fit, len(l_inds), len(r_inds), dbg

    # ------------------------------------------------------------------
    # 미션 1: 색상 구역 (Hysteresis 포함)
    # ------------------------------------------------------------------
    def detect_color_zone(self, frame):
        msg = ColorZone()
        h, w = frame.shape[:2]

        # 범퍼 바로 앞 좁은 ROI (원경 색상이 미리 인식되는 것을 원천 차단)
        y0 = int(h * self.zone_top)
        x0 = int(w * self.zone_xm)
        x1 = int(w * (1.0 - self.zone_xm))
        roi = frame[y0:h, x0:x1]
        if roi.size == 0:
            msg.zone = self.current_zone
            return msg

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(cv2.inRange(hsv, self.r1lo, self.r1hi),
                             cv2.inRange(hsv, self.r2lo, self.r2hi))
        blue = cv2.inRange(hsv, self.blo, self.bhi)

        k = np.ones((5, 5), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k)
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, k)

        total = roi.shape[0] * roi.shape[1]
        r_ratio = cv2.countNonZero(red) / total
        b_ratio = cv2.countNonZero(blue) / total
        msg.red_ratio = float(r_ratio)
        msg.blue_ratio = float(b_ratio)

        # --- Hysteresis: 진입 문턱은 높게, 이탈 문턱은 낮게 ---
        if self.current_zone == ColorZone.ZONE_NONE:
            if r_ratio > self.zone_enter and r_ratio > b_ratio:
                raw = ColorZone.ZONE_RED
            elif b_ratio > self.zone_enter and b_ratio > r_ratio:
                raw = ColorZone.ZONE_BLUE
            else:
                raw = ColorZone.ZONE_NONE
        elif self.current_zone == ColorZone.ZONE_RED:
            raw = ColorZone.ZONE_RED if r_ratio > self.zone_exit else ColorZone.ZONE_NONE
        else:
            raw = ColorZone.ZONE_BLUE if b_ratio > self.zone_exit else ColorZone.ZONE_NONE

        # --- 연속 N프레임 확인 후에만 실제 전이 (순간 노이즈 차단) ---
        if raw == self.zone_candidate:
            self.zone_streak += 1
        else:
            self.zone_candidate = raw
            self.zone_streak = 1

        if self.zone_streak >= self.zone_confirm and raw != self.current_zone:
            self.current_zone = raw
            names = {0: 'NONE', 1: 'RED', 2: 'BLUE'}
            self.get_logger().info(f'구역 전이 -> {names[raw]} '
                                   f'(R={r_ratio:.2f} B={b_ratio:.2f})')

        msg.zone = self.current_zone
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = CameraPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
