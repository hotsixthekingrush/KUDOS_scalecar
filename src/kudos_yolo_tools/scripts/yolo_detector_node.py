#!/usr/bin/env python3
"""
yolo_detector_node.py
카메라 영상 -> YOLO 미션 객체 인식 -> 판단 모듈로 전달 (인지만 담당)

  입력  /camera/color/image_raw        (sensor_msgs/Image)
        /camera/color/camera_info      (sensor_msgs/CameraInfo)
        /camera/depth/image_raw        (sensor_msgs/Image, 컬러에 정렬된 깊이)  — 선택
  출력  /perception/mission_objects    (ar_msgs/MissionObjectArray)
        /perception/yolo_debug         (sensor_msgs/Image)  — publish_debug=true 일 때만

[원칙] LaneError 와 동일
  - 픽셀은 노드 밖으로 내보내지 않는다. 위치는 base_link 기준 미터.
  - 검출이 없으면 objects 가 빈 배열인 메시지를 계속 발행한다
    (판단 쪽이 "안 보임"과 "노드 죽음"을 구분할 수 있도록).
  - 연속 프레임 확인(채터링 방지)은 판단 모듈 책임. 여기서는 위치값 흔들림만
    지수이동평균(EMA)으로 줄인다.

[2026 규정 기준 (11/2, 판단 세진님 요청)]
  클래스: PEDESTRIAN_SIGN, SIGN_A, SIGN_B, TRAFFIC_LIGHT(+light_color), PARKING_SIGN,
          LIMO(보이는 것 전부), BUMP_SIGN, SLOW_SIGN, SLOW_GUIDE_LINE
  angle_valid / angle_rad 는 메시지에서 삭제됨 (차단기 없음)

[거리 추정 — 미션마다 다름]
  crosswalk       : 박스 하단 ∩ 바닥 → x_m = 횡단보도의 차에 가까운 끝까지 거리. EMA 없음.
                    화면 맨 아래에 닿으면 position_valid=false (판단이 odom 으로 이어서 계산)
                    미션 진입 신호 아님 — 신호등·보행자 미션의 정지 기준 (11/2)
  slow_guide_line : 박스 하단 중앙 픽셀의 광선 ∩ 바닥평면 (깊이 불필요)
                    → x_m = 유도선 가까운 끝(시작점)까지 거리. 박스가 화면 맨 아래에 닿으면
                      이미 올라탄 것 → position_valid=false
  limo            : 깊이 → 실패 시 박스 하단 ∩ 바닥 (차 바퀴가 바닥에 닿아 있으므로 크기 역산보다 정확)
  그 외 표지판류   : 정렬된 깊이의 박스 중앙부 값 → 실패 시 알려진 실제 높이로 역산
  traffic_light   : 위와 같고, 박스 안 램프 색으로 light_color (light_color.py)

[모델]
  weights 에 .pt(개발 PC) / .engine(Jetson TensorRT) 모두 가능.
  클래스 번호는 model.names 의 "이름"으로 MissionObject 상수에 매핑하므로
  학습 시 클래스 순서가 달라도 잘못된 번호가 나가지 않는다. (모르는 이름은 무시+경고)

[호환] Python 3.8 문법만 사용 (실차 Humble 이어도 문제없음).

실행:
  ros2 run kudos_yolo_tools yolo_detector_node.py --ros-args \
    --params-file src/kudos_yolo_tools/config/yolo_detector.yaml
"""

import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException

import yolo_geometry as geo
import light_color as lc

try:
    from ar_msgs.msg import MissionObject, MissionObjectArray
except ImportError:
    MissionObject = None
    MissionObjectArray = None


# 학습 클래스 이름 -> MissionObject 상수 이름 (2026 규정, ar_msgs feat/decision-2026)
#   학습 데이터(dataset_v2.yaml)는 A/B 를 marker_left/right 이름으로 쓰고 있어 둘 다 받는다.
CLASS_TO_CONST = {
    'pedestrian_sign': 'PEDESTRIAN_SIGN',
    'sign_a': 'SIGN_A', 'marker_left': 'SIGN_A',
    'sign_b': 'SIGN_B', 'marker_right': 'SIGN_B',
    'traffic_light': 'TRAFFIC_LIGHT',
    'parking_sign': 'PARKING_SIGN',
    'limo': 'LIMO',
    'bump_sign': 'BUMP_SIGN',
    'slow_sign': 'SLOW_SIGN',
    'slow_guide_line': 'SLOW_GUIDE_LINE',
    'crosswalk': 'CROSSWALK',            # 11/2 추가 (=9). 신호등·보행자 미션의 정지 기준
}
# 한 화면에 여러 개 나올 수 있는 클래스 (나머지는 가장 확실한 1개만 보냄)
MULTI_INSTANCE = {'limo'}
# 바닥에 그려진 것 → 박스 하단 ∩ 바닥으로 거리
GROUND_CLASSES = {'slow_guide_line', 'crosswalk'}
# 위치 EMA 를 하지 않는 클래스 — 다가가는 중이라 평균을 내면 거리가 늦게 나옴
#   (ema_alpha 0.5 면 약 2 cm 늦음). 판단이 3 cm 안에 세우므로 원값 그대로 보냄
NO_EMA = {'crosswalk'}


def tf_to_T(tf_msg):
    q = tf_msg.transform.rotation
    t = tf_msg.transform.translation
    return geo.make_T(geo.quat_to_rot(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector_node')
        if MissionObject is None:
            raise RuntimeError(
                'ar_msgs 에 MissionObject/MissionObjectArray 가 없습니다. '
                'docs/ar_msgs_proposal 의 메시지를 ar_msgs 에 추가하고 다시 빌드하세요.')

        p = self.declare_parameter
        p('weights', 'best.pt')
        p('image_topic', '/camera/color/image_raw')
        p('camera_info_topic', '/camera/color/camera_info')
        p('depth_topic', '/camera/depth/image_raw')     # '' 이면 깊이 안 씀
        p('base_frame', 'base_link')
        p('camera_frame', '')                           # 비우면 이미지 header.frame_id
        p('camera_frame_is_optical', True)
        p('rate_hz', 15.0)
        p('imgsz', 640)
        p('device', '0')
        p('half', True)
        p('conf_default', 0.5)
        # 클래스별 신뢰도 문턱 — 이름:값 문자열 목록 (예: "marker_left:0.6")
        p('conf_per_class', [''])
        # 깊이 실패 시 크기로 거리 역산할 때 쓰는 실제 높이 (m) — 이름:값  [실측필요]
        p('object_height_m', ['marker_left:0.08', 'marker_right:0.08', 'sign_a:0.08', 'sign_b:0.08',
                              'parking_sign:0.08', 'pedestrian_sign:0.08',
                              'slow_sign:0.08', 'bump_sign:0.08', 'traffic_light:0.205'])
        # 화면 가장자리에 닿아 잘린 표지판은 버림 (잘린 A/B 가 P 처럼 보이는 경우 확인, 10/8). 0 = 끔
        p('edge_reject_px', 4)
        p('edge_reject_classes', ['marker_left', 'marker_right', 'sign_a', 'sign_b', 'parking_sign'])
        p('light_s_min', 90)        # 신호등 색 판정: 켜진 램프로 볼 최소 채도
        p('light_v_min', 150)       #                 최소 밝기
        # 신호등 모양 검사 (11/4, M1 광고 모니터 대응). 화면 속 그림 신호등·원은 크기나 비율이 다름
        #   실제 본체 12 x 20.5 cm → 세로/가로 1.71. 깊이가 있으면 박스 높이로 실제 높이(m)를 계산해 검사
        #   통과한 후보 중 가장 확실한 1개를 보냄 (예전: 무조건 conf 최고 1개)
        p('light_check', True)
        p('light_aspect_min', 1.2)  # 박스 세로/가로
        p('light_aspect_max', 3.0)
        p('light_h_min_m', 0.13)    # 박스 높이 x 깊이 / fy 로 구한 실제 높이 (m)
        p('light_h_max_m', 0.30)
        p('ground_z', 0.0)          # base_link 기준 바닥 높이. ground_frame 을 찾으면 자동으로 덮어씀
        p('ground_frame', 'base_footprint')   # 바닥에 붙은 프레임. '' 이면 ground_z 그대로 사용
        p('max_depth_m', 3.0)       # DaBai 깊이 유효거리 ~3m
        p('ema_alpha', 0.5)         # 1.0 = 필터 없음
        p('ema_reset_sec', 0.5)     # 이 시간 이상 안 보이면 필터 초기화
        p('publish_debug', False)

        g = lambda n: self.get_parameter(n).value
        self.image_topic = g('image_topic')
        self.cam_frame_param = g('camera_frame')
        self.is_optical = bool(g('camera_frame_is_optical'))
        self.base_frame = g('base_frame')
        self.imgsz = int(g('imgsz'))
        self.device = str(g('device'))
        self.half = bool(g('half'))
        self.conf_default = float(g('conf_default'))
        self.conf_per_class = self._parse_kv(g('conf_per_class'))
        self.obj_h = self._parse_kv(g('object_height_m'))
        self.ground_z = float(g('ground_z'))
        self.ground_frame = g('ground_frame')
        self.ground_z_ready = not self.ground_frame
        self.max_depth = float(g('max_depth_m'))
        self.ema_alpha = float(g('ema_alpha'))
        self.ema_reset = float(g('ema_reset_sec'))
        self.publish_debug = bool(g('publish_debug'))
        self.edge_px = int(g('edge_reject_px'))
        self.edge_classes = set(x for x in g('edge_reject_classes') if x)
        self.light_s_min = int(g('light_s_min'))
        self.light_v_min = int(g('light_v_min'))
        self.light_check = bool(g('light_check'))
        self.light_aspect = (float(g('light_aspect_min')), float(g('light_aspect_max')))
        self.light_h = (float(g('light_h_min_m')), float(g('light_h_max_m')))
        self.light_rejected = 0

        # --- 모델 로드 ---
        from ultralytics import YOLO
        weights = g('weights')
        self.model = YOLO(weights, task=None)
        self.names = self.model.names if isinstance(self.model.names, dict) \
            else dict(enumerate(self.model.names))
        self.cls_map = {}
        for i, nm in self.names.items():
            const = CLASS_TO_CONST.get(nm)
            if const is None:
                self.get_logger().warn('모델 클래스 "%s" 는 MissionObject 에 없음 → 무시' % nm)
                continue
            if not hasattr(MissionObject, const):
                self.get_logger().warn('MissionObject 에 %s 상수가 없음 (ar_msgs 옛 버전?) → "%s" 무시'
                                       % (const, nm))
                continue
            self.cls_map[int(i)] = (nm, int(getattr(MissionObject, const)))
        self.min_conf = min([self.conf_default] + list(self.conf_per_class.values()))
        self.has_light = hasattr(MissionObject(), 'light_color')
        if not self.has_light:
            self.get_logger().warn('MissionObject 에 light_color 필드가 없음 → 신호등 색은 안 보냄')
        self.get_logger().info('모델 로드: %s | 클래스 %s' % (weights, self.names))

        # --- 통신 ---
        self.bridge = CvBridge()
        self.K = None
        self.T_base_cam = None
        self.latest_img = None
        self.latest_depth = None
        self.last_stamp = None
        self.ema = {}   # class name -> (x, y, t)
        self.stat_n, self.stat_ms = 0, 0.0
        self.win = []        # 최근 처리 기록 (처리 ms, 지연 s) — 주기 로그용

        # 영상은 큐 1장: 처리 중에 쌓인 옛 프레임을 나중에 하나씩 처리하는 일이 없도록.
        # (rclpy 는 한 번 돌 때 구독당 메시지 1개만 꺼내므로, 큐가 5장이면 타이머와 번갈아
        #  옛 프레임 5장을 차례로 처리 → 지연 수 초 + 결과가 몰려서 나감. 10/1 확인)
        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(CameraInfo, g('camera_info_topic'), self.info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.image_topic, self.image_cb, img_qos)
        if g('depth_topic'):
            self.create_subscription(Image, g('depth_topic'), self.depth_cb, img_qos)

        # 결과도 최신 1개만 유지 (받는 쪽이 늦어도 옛 결과가 줄줄이 가지 않게)
        self.pub = self.create_publisher(MissionObjectArray, '/perception/mission_objects', 1)
        if self.publish_debug:
            self.pub_dbg = self.create_publisher(Image, '/perception/yolo_debug', 2)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 콜백마다 추론하면 큐가 밀려 지연이 쌓인다 → 최신 프레임만 타이머로 처리
        self.create_timer(1.0 / float(g('rate_hz')), self.process)
        self.get_logger().info('yolo_detector_node 시작 | 입력=%s' % self.image_topic)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_kv(items):
        out = {}
        for s in items or []:
            if s and ':' in s:
                k, v = s.split(':', 1)
                out[k.strip()] = float(v)
        return out

    def info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def image_cb(self, msg):
        self.latest_img = msg

    def depth_cb(self, msg):
        self.latest_depth = msg

    def get_T_base_cam(self, cam_frame):
        if self.T_base_cam is not None:
            return self.T_base_cam
        try:
            T = tf_to_T(self.tf_buffer.lookup_transform(
                self.base_frame, cam_frame, rclpy.time.Time()))
        except TransformException as e:
            self.get_logger().warn('TF %s->%s 실패: %s' % (self.base_frame, cam_frame, e),
                                   throttle_duration_sec=5.0)
            return None
        if not self.is_optical:
            T = T @ geo.T_LINK_FROM_OPTICAL
        self.T_base_cam = T
        return T

    def update_ground_z(self):
        """base_link 기준 바닥 높이를 TF 로 1회 계산 (LIMO 는 base_link 가 바닥보다 위에 있음)"""
        if self.ground_z_ready:
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.ground_frame, rclpy.time.Time())
            self.ground_z = float(tf.transform.translation.z)
            self.get_logger().info('바닥 높이(%s 기준 %s) = %.3f m'
                                   % (self.base_frame, self.ground_frame, self.ground_z))
        except TransformException:
            self.get_logger().warn('TF %s->%s 없음 → ground_z=%.3f 사용'
                                   % (self.base_frame, self.ground_frame, self.ground_z),
                                   throttle_duration_sec=10.0)
            return
        self.ground_z_ready = True

    # ------------------------------------------------------------------
    def process(self):
        msg = self.latest_img
        if msg is None or msg.header.stamp == self.last_stamp:
            return
        self.last_stamp = msg.header.stamp
        t_start = time.time()
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]
        if self.K is None:
            # 검출 결과는 계속 보낸다(빈 배열 포함). 위치만 position_valid=false
            self.get_logger().warn('CameraInfo 미수신 → 위치 없이 발행', throttle_duration_sec=5.0)
            T_base_cam = None
        else:
            T_base_cam = self.get_T_base_cam(self.cam_frame_param or msg.header.frame_id)
            self.update_ground_z()

        depth = None
        if self.latest_depth is not None:
            d = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
            depth = geo.depth_to_meters(d, self.latest_depth.encoding)
            if depth.shape[:2] != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        t0 = time.time()
        kw = dict(imgsz=self.imgsz, conf=self.min_conf, device=self.device, verbose=False)
        if self.half and not str(self.get_parameter('weights').value).endswith('.engine'):
            kw['half'] = True    # .engine 은 export 때 이미 FP16 으로 빌드됨
        res = self.model.predict(frame, **kw)[0]
        self.stat_ms += (time.time() - t0) * 1000.0
        self.stat_n += 1

        rejected = []
        dets = self.collect(res, w, h, depth, rejected)
        out = MissionObjectArray()
        out.header.stamp = msg.header.stamp          # 원본 영상 촬영 시각 그대로 (판단 쪽 지연 보정용)
        out.header.frame_id = self.base_frame
        now = time.time()
        dbg = frame.copy() if self.publish_debug else None

        for d in dets:
            obj = MissionObject()
            obj.class_id = d['const']
            obj.confidence = float(d['conf'])
            obj.position_source = MissionObject.POS_NONE
            xy, src = self.estimate_position(d, depth, T_base_cam, h)
            if xy is not None:
                if d['name'] not in MULTI_INSTANCE and d['name'] not in NO_EMA:
                    xy = self.smooth(d['name'], xy, now)
                obj.position_valid = True
                obj.x_m, obj.y_m = float(xy[0]), float(xy[1])
                obj.position_source = src
            color = None
            if d['name'] == 'traffic_light':
                color, why = lc.classify(frame, d['box'], self.light_s_min, self.light_v_min)
                if self.has_light:
                    obj.light_color = int(color)
            out.objects.append(obj)

            if dbg is not None:
                x1, y1, x2, y2 = [int(v) for v in d['box']]
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt = '%s %.2f' % (d['name'], d['conf'])
                if obj.position_valid:
                    txt += ' x=%.2f y=%.2f' % (obj.x_m, obj.y_m)
                if color is not None:
                    txt += ' ' + lc.NAMES[color]
                cv2.putText(dbg, txt, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if dbg is not None:
            for d, why in rejected:   # 모양 검사에서 버린 신호등 후보 (회색)
                x1, y1, x2, y2 = [int(v) for v in d['box']]
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (150, 150, 150), 1)
                cv2.putText(dbg, 'X light %.2f %s' % (d['conf'], why), (x1, min(y2 + 14, h - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        self.pub.publish(out)
        if dbg is not None:
            m = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
            m.header = msg.header
            self.pub_dbg.publish(m)
        self.log_stats(msg, t_start)

    def log_stats(self, msg, t_start):
        """20장마다: 노드 처리 시간(벽시계 ms)과 발행 시점의 영상 지연(age, 노드 시계 기준 s)"""
        proc_ms = (time.time() - t_start) * 1000.0
        age = (self.get_clock().now() - rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9
        self.win.append((proc_ms, age))
        if len(self.win) >= 20:
            p = [w[0] for w in self.win]
            a = [w[1] for w in self.win]
            self.get_logger().info('최근 %d장 | 처리 평균 %.1f ms (추론만 누적 %.1f ms) | 지연 age 평균 %.2f s, 최대 %.2f s'
                                   % (len(self.win), sum(p) / len(p), self.stat_ms / self.stat_n,
                                      sum(a) / len(a), max(a)))
            self.win = []

    def collect(self, res, img_w, img_h, depth=None, rejected=None):
        """YOLO 결과 -> 클래스별 문턱 적용, 가장자리에 잘린 표지판 제거,
        단일 인스턴스 클래스는 최고 신뢰도 1개만 (LIMO 는 전부).
        신호등은 모양 검사(light_shape_ok)를 통과한 후보 중 최고 신뢰도 1개."""
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        e = self.edge_px
        best = {}
        multi = []
        lights = []
        for i in range(len(cls)):
            if int(cls[i]) not in self.cls_map:
                continue
            name, const = self.cls_map[int(cls[i])]
            if conf[i] < self.conf_per_class.get(name, self.conf_default):
                continue
            x1, y1, x2, y2 = xyxy[i]
            if e > 0 and name in self.edge_classes and (
                    x1 <= e or y1 <= e or x2 >= img_w - e or y2 >= img_h - e):
                continue
            d = {'name': name, 'const': const, 'conf': float(conf[i]), 'box': xyxy[i]}
            if name in MULTI_INSTANCE:
                multi.append(d)
            elif name == 'traffic_light' and self.light_check:
                lights.append(d)
            elif name not in best or d['conf'] > best[name]['conf']:
                best[name] = d
        for d in sorted(lights, key=lambda x: -x['conf']):
            ok, why = self.light_shape_ok(d['box'], depth, img_w, img_h)
            if ok:
                best['traffic_light'] = d
                break
            self.light_rejected += 1
            if rejected is not None:
                rejected.append((d, why))
            self.get_logger().info('신호등 후보 버림 (conf %.2f, %s) — 광고 화면 등. 누적 %d'
                                   % (d['conf'], why, self.light_rejected), throttle_duration_sec=2.0)
        return list(best.values()) + multi

    def light_shape_ok(self, box, depth, img_w, img_h):
        """신호등 박스가 실제 신호등 모양·크기인지. 반환 (통과 여부, 이유 문자열)"""
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        # 화면 가장자리에 잘린 박스는 비율·크기를 믿을 수 없음 → 검사 없이 통과
        if x1 <= 2 or y1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2:
            return True, 'edge'
        aspect = bh / bw
        if not (self.light_aspect[0] <= aspect <= self.light_aspect[1]):
            return False, 'ratio %.2f' % aspect
        if depth is not None and self.K is not None:
            z = geo.robust_box_depth(depth, x1, y1, x2, y2)
            if z is not None and z <= self.max_depth:
                h_m = bh * z / float(self.K[1, 1])
                if not (self.light_h[0] <= h_m <= self.light_h[1]):
                    return False, 'h %.2fm' % h_m
        return True, ''

    def estimate_position(self, d, depth, T_base_cam, img_h):
        """반환: ((x, y) base_link, position_source) 또는 (None, POS_NONE)"""
        if T_base_cam is None:
            return None, MissionObject.POS_NONE
        x1, y1, x2, y2 = [float(v) for v in d['box']]
        u = (x1 + x2) / 2.0

        if d['name'] in GROUND_CLASSES:
            # 박스 하단 = 유도선 가까운 끝(시작점). 화면 맨 아래에 닿았으면 이미 올라탄 중
            if y2 >= img_h - 2:
                return None, MissionObject.POS_NONE
            xy = geo.ray_ground_intersection(u, y2, self.K, T_base_cam, self.ground_z)
            return (xy, MissionObject.POS_GROUND) if xy else (None, MissionObject.POS_NONE)

        v = (y1 + y2) / 2.0
        z, src = None, MissionObject.POS_NONE
        if depth is not None:
            z = geo.robust_box_depth(depth, x1, y1, x2, y2)
            if z is not None and z <= self.max_depth:
                src = MissionObject.POS_DEPTH
            else:
                z = None
        if z is None and d['name'] == 'limo' and y2 < img_h - 2:
            # 차는 바닥에 서 있음 → 박스 하단(바퀴) ∩ 바닥. 크기 역산보다 y 가 정확
            xy = geo.ray_ground_intersection(u, y2, self.K, T_base_cam, self.ground_z)
            if xy:
                return xy, MissionObject.POS_GROUND
        if z is None and d['name'] in self.obj_h:
            z = geo.depth_from_size(y2 - y1, self.obj_h[d['name']], self.K[1, 1])
            src = MissionObject.POS_SIZE
        if z is None:
            return None, MissionObject.POS_NONE
        p_cam = geo.backproject(u, v, z, self.K)
        p_base = geo.transform_points(T_base_cam, p_cam)[0]
        return (float(p_base[0]), float(p_base[1])), src

    def smooth(self, name, xy, now):
        prev = self.ema.get(name)
        if prev is None or now - prev[2] > self.ema_reset:
            sx, sy = xy
        else:
            a = self.ema_alpha
            sx = a * xy[0] + (1 - a) * prev[0]
            sy = a * xy[1] + (1 - a) * prev[1]
        self.ema[name] = (sx, sy, now)
        return sx, sy


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
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
