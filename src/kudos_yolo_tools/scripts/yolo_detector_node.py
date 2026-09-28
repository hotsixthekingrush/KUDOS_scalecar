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

[거리 추정 — 미션마다 다름]
  crosswalk     : 박스 하단 중앙 픽셀의 광선 ∩ 바닥평면  (IPM 과 같은 원리, 깊이 불필요)
                  → x_m = 횡단보도 "앞쪽 끝"까지 거리
  그 외 표지판류 : 정렬된 깊이의 박스 중앙부 값  → 실패 시 알려진 실제 높이로 역산
  gate_bar      : 위와 동일 + seg 모델이면 마스크로 화면상 기울기(angle_rad)

[모델]
  weights 에 .pt(개발 PC) / .engine(Jetson TensorRT) 모두 가능.
  클래스 번호는 model.names 의 "이름"으로 MissionObject 상수에 매핑하므로
  학습 시 클래스 순서가 달라도 잘못된 번호가 나가지 않는다. (모르는 이름은 무시+경고)

[Foxy 호환] Python 3.8 문법만 사용.

실행:
  ros2 run kudos_yolo_tools yolo_detector_node.py --ros-args \
    --params-file src/kudos_yolo_tools/config/yolo_detector.yaml
"""

import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException

import yolo_geometry as geo

try:
    from ar_msgs.msg import MissionObject, MissionObjectArray
except ImportError:
    MissionObject = None
    MissionObjectArray = None


# 학습 클래스 이름 -> MissionObject 상수 이름
CLASS_TO_CONST = {
    'crosswalk': 'CROSSWALK',
    'marker_left': 'MARKER_LEFT',
    'marker_right': 'MARKER_RIGHT',
    'gate_bar': 'GATE_BAR',
    'parking_sign': 'PARKING_SIGN',
    'dynamic_car': 'DYNAMIC_CAR',
}
# 한 화면에 여러 개 나올 수 있는 클래스 (나머지는 가장 확실한 1개만 보냄)
MULTI_INSTANCE = {'dynamic_car'}


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
        p('object_height_m', ['marker_left:0.16', 'marker_right:0.16',
                              'parking_sign:0.16', 'dynamic_car:0.20'])
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
            self.cls_map[int(i)] = (nm, int(getattr(MissionObject, const)))
        self.min_conf = min([self.conf_default] + list(self.conf_per_class.values()))
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

        qos = qos_profile_sensor_data
        self.create_subscription(CameraInfo, g('camera_info_topic'), self.info_cb, qos)
        self.create_subscription(Image, self.image_topic, self.image_cb, qos)
        if g('depth_topic'):
            self.create_subscription(Image, g('depth_topic'), self.depth_cb, qos)

        self.pub = self.create_publisher(MissionObjectArray, '/perception/mission_objects', 10)
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
        if self.K is None:
            self.get_logger().warn('CameraInfo 미수신', throttle_duration_sec=5.0)
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]
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
        if self.stat_n % 100 == 0:
            self.get_logger().info('추론 평균 %.1f ms' % (self.stat_ms / self.stat_n))

        dets = self.collect(res)
        out = MissionObjectArray()
        out.header = msg.header
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
                if d['name'] not in MULTI_INSTANCE:
                    xy = self.smooth(d['name'], xy, now)
                obj.position_valid = True
                obj.x_m, obj.y_m = float(xy[0]), float(xy[1])
                obj.position_source = src
            if d.get('angle') is not None:
                obj.angle_valid = True
                obj.angle_rad = float(d['angle'])
            out.objects.append(obj)

            if dbg is not None:
                x1, y1, x2, y2 = [int(v) for v in d['box']]
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt = '%s %.2f' % (d['name'], d['conf'])
                if obj.position_valid:
                    txt += ' x=%.2f y=%.2f' % (obj.x_m, obj.y_m)
                if obj.angle_valid:
                    txt += ' %.0fdeg' % np.degrees(obj.angle_rad)
                cv2.putText(dbg, txt, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        self.pub.publish(out)
        if dbg is not None:
            m = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
            m.header = msg.header
            self.pub_dbg.publish(m)

    def collect(self, res):
        """YOLO 결과 -> 클래스별 문턱 적용, 단일 인스턴스 클래스는 최고 신뢰도 1개만"""
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        masks = res.masks.xy if getattr(res, 'masks', None) is not None else None

        best = {}
        multi = []
        for i in range(len(cls)):
            if int(cls[i]) not in self.cls_map:
                continue
            name, const = self.cls_map[int(cls[i])]
            if conf[i] < self.conf_per_class.get(name, self.conf_default):
                continue
            d = {'name': name, 'const': const, 'conf': float(conf[i]), 'box': xyxy[i]}
            if name == 'gate_bar' and masks is not None and len(masks[i]) >= 5:
                d['angle'] = geo.bar_angle_from_points(masks[i])
            if name in MULTI_INSTANCE:
                multi.append(d)
            elif name not in best or d['conf'] > best[name]['conf']:
                best[name] = d
        return list(best.values()) + multi

    def estimate_position(self, d, depth, T_base_cam, img_h):
        """반환: ((x, y) base_link, position_source) 또는 (None, POS_NONE)"""
        if T_base_cam is None:
            return None, MissionObject.POS_NONE
        x1, y1, x2, y2 = [float(v) for v in d['box']]
        u = (x1 + x2) / 2.0

        if d['name'] == 'crosswalk':
            # 박스 하단 = 횡단보도의 앞쪽 끝. 화면 맨 아래에 닿았으면 이미 밟고 있는 중
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
