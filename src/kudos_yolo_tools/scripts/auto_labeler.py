#!/usr/bin/env python3
"""
auto_labeler.py
Gazebo 시뮬레이션에서 카메라 이미지와 YOLO 라벨을 자동으로 동시 생성하는 노드.

[원리]
Gazebo는 모든 물체의 3D 위치·크기를 이미 알고 있다.
따라서 물체의 3D 바운딩박스 8개 꼭짓점을 카메라 내부 파라미터(intrinsics)로
2D 이미지 평면에 투영하면, 사람이 손으로 그리는 것보다 정확한
Bounding Box 좌표를 얻을 수 있다.

[출력]
dataset/images/train/frame_000001.jpg
dataset/labels/train/frame_000001.txt   <- YOLO 포맷 (class cx cy w h, 모두 0~1 정규화)

[사전 준비]
1) 월드 파일에 gazebo_ros_state 플러그인 추가 (물체 위치를 토픽으로 받기 위해)
   <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
     <ros><namespace>/gazebo</namespace></ros>
     <update_rate>30.0</update_rate>
   </plugin>

2) 실행
   ros2 run <pkg> auto_labeler.py --ros-args \
      -p image_topic:=/camera/color/image_raw \
      -p camera_info_topic:=/camera/color/camera_info \
      -p output_dir:=./dataset
"""

import os
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException


# =====================================================================
# 탐지 대상 정의
#   - gazebo_name : 월드 파일에 정의된 모델 이름 (동적 물체용)
#   - world_pose  : 고정 위치 (노면에 그려진 횡단보도처럼 3D 모델이 아닌 것)
#   - size        : (길이x, 폭y, 높이z) 단위 m — 3D 바운딩박스 크기
# =====================================================================
CLASS_NAMES = [
    'crosswalk',      # 0
    'marker_left',    # 1
    'marker_right',   # 2
    'gate_bar',       # 3
    'parking_sign',   # 4
    'dynamic_car',    # 5
]

TARGETS = [
    # 노면에 그려진 횡단보도 — 3D 모델이 없으므로 월드 좌표를 직접 지정
    # (autorace_2026.world 기준 상단 우측 횡단보도 위치)
    {'class_id': 0, 'world_pose': (2.10, 3.15, 0.005), 'size': (0.45, 0.60, 0.01)},

    # Gazebo 모델로 존재하는 것들 — 이름으로 위치를 조회
    {'class_id': 1, 'gazebo_name': 'sign_lane_change', 'size': (0.02, 0.16, 0.16), 'z_offset': 0.30},
    {'class_id': 4, 'gazebo_name': 'sign_parking',     'size': (0.02, 0.16, 0.16), 'z_offset': 0.30},
    {'class_id': 3, 'gazebo_name': 'gate_bar',         'size': (0.04, 1.00, 0.05), 'z_offset': 0.17},
    # 동적 차량은 움직이므로 매 프레임 위치를 다시 조회
    {'class_id': 5, 'gazebo_name': 'dynamic_car_1',    'size': (0.32, 0.22, 0.25), 'z_offset': 0.12},
    {'class_id': 5, 'gazebo_name': 'dynamic_car_2',    'size': (0.32, 0.22, 0.25), 'z_offset': 0.12},
]

# 최소 박스 크기 (픽셀) — 너무 작게 보이면 학습에 방해되므로 제외
MIN_BOX_PX = 12
# 이미지 밖으로 이 비율 이상 벗어나면 제외
MAX_CLIP_RATIO = 0.6


def quat_to_rot(x, y, z, w):
    """쿼터니언 -> 3x3 회전행렬"""
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


def bbox_corners_3d(center, size, R=None):
    """물체 중심과 크기로부터 3D 바운딩박스 8개 꼭짓점 생성 (월드 좌표계)"""
    cx, cy, cz = center
    sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
    local = np.array([
        [-sx,-sy,-sz], [ sx,-sy,-sz], [ sx, sy,-sz], [-sx, sy,-sz],
        [-sx,-sy, sz], [ sx,-sy, sz], [ sx, sy, sz], [-sx, sy, sz],
    ])
    if R is not None:
        local = local @ R.T
    return local + np.array([cx, cy, cz])


class AutoLabeler(Node):
    def __init__(self):
        super().__init__('auto_labeler')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('output_dir', './dataset')
        self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('save_every_n', 3)       # N프레임마다 1장 저장 (중복 줄이기)
        self.declare_parameter('draw_debug', True)      # 확인용 시각화 이미지도 같이 저장

        self.image_topic = self.get_parameter('image_topic').value
        self.info_topic = self.get_parameter('camera_info_topic').value
        self.out_dir = self.get_parameter('output_dir').value
        self.cam_frame = self.get_parameter('camera_optical_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        self.save_every_n = self.get_parameter('save_every_n').value
        self.draw_debug = self.get_parameter('draw_debug').value

        # 출력 폴더 준비
        for sub in ['images/train', 'labels/train', 'debug']:
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

        self.bridge = CvBridge()
        self.K = None                  # 카메라 내부 파라미터
        self.model_states = {}         # Gazebo 물체 위치 캐시
        self.frame_idx = 0
        self.saved_count = 0

        # 카메라 영상·CameraInfo 는 best-effort 로 발행되는 경우가 많다.
        # reliable 로 구독하면 조용히 아무것도 안 들어오므로 sensor_data QoS 사용.
        # (best-effort 구독은 reliable 발행자와도 호환되므로 ModelStates 에도 안전)
        self.create_subscription(CameraInfo, self.info_topic, self.info_cb, qos_profile_sensor_data)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.states_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile_sensor_data)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(f'auto_labeler 시작 | 저장 경로: {self.out_dir}')

    def info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f'카메라 내부 파라미터 수신: fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} '
                f'cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}')

    def states_cb(self, msg: ModelStates):
        for name, pose in zip(msg.name, msg.pose):
            self.model_states[name] = pose

    def get_camera_transform(self):
        """월드 -> 카메라 광학 프레임 변환 (R, t) 반환"""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.cam_frame, self.world_frame, rclpy.time.Time())
        except TransformException:
            return None, None
        q = tf.transform.rotation
        t = tf.transform.translation
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        return R, np.array([t.x, t.y, t.z])

    def project_to_image(self, pts_world, R, t, img_w, img_h):
        """월드 좌표 3D 점들 -> 2D 픽셀 좌표. 카메라 뒤쪽 점이 있으면 None."""
        pts_cam = (pts_world @ R.T) + t          # 카메라 광학 좌표계 (Z가 전방)
        if np.any(pts_cam[:, 2] <= 0.05):        # 카메라 뒤 또는 너무 가까움
            return None
        u = self.K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + self.K[1, 2]
        return np.stack([u, v], axis=1)

    def compute_bbox(self, target, R, t, img_w, img_h):
        """하나의 대상에 대해 2D 바운딩박스 계산. 반환: (class_id, x1,y1,x2,y2) 또는 None"""
        # --- 물체의 월드 좌표 결정 ---
        if 'world_pose' in target:
            center = target['world_pose']
            rot = None
        else:
            name = target.get('gazebo_name')
            pose = self.model_states.get(name)
            if pose is None:
                return None
            z_off = target.get('z_offset', 0.0)
            center = (pose.position.x, pose.position.y, pose.position.z + z_off)
            q = pose.orientation
            rot = quat_to_rot(q.x, q.y, q.z, q.w)

        corners = bbox_corners_3d(center, target['size'], rot)
        uv = self.project_to_image(corners, R, t, img_w, img_h)
        if uv is None:
            return None

        x1, y1 = uv[:, 0].min(), uv[:, 1].min()
        x2, y2 = uv[:, 0].max(), uv[:, 1].max()

        # 화면 밖으로 너무 많이 벗어난 경우 제외
        full_area = max((x2 - x1) * (y2 - y1), 1e-6)
        cx1, cy1 = max(x1, 0), max(y1, 0)
        cx2, cy2 = min(x2, img_w), min(y2, img_h)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        clipped_area = (cx2 - cx1) * (cy2 - cy1)
        if clipped_area / full_area < (1 - MAX_CLIP_RATIO):
            return None
        if (cx2 - cx1) < MIN_BOX_PX or (cy2 - cy1) < MIN_BOX_PX:
            return None

        return (target['class_id'], cx1, cy1, cx2, cy2)

    def image_cb(self, msg: Image):
        self.frame_idx += 1
        if self.frame_idx % self.save_every_n != 0:
            return
        if self.K is None:
            # CameraInfo 가 안 오면 라벨이 한 장도 저장되지 않는다. 조용히 넘어가지 않도록 경고.
            self.get_logger().warn(
                f'CameraInfo 미수신 — {self.info_topic} 토픽 이름과 QoS 를 확인하세요. '
                '이 상태에선 아무것도 저장되지 않습니다.', throttle_duration_sec=5.0)
            return

        R, t = self.get_camera_transform()
        if R is None:
            self.get_logger().warn('TF 조회 실패 — world_frame / camera_optical_frame 파라미터를 확인하세요',
                                   throttle_duration_sec=5.0)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')
            return

        h, w = frame.shape[:2]
        boxes = []
        for tgt in TARGETS:
            r = self.compute_bbox(tgt, R, t, w, h)
            if r is not None:
                boxes.append(r)

        # 대상이 하나도 안 보이는 프레임도 10%는 저장 (negative sample)
        if not boxes and (self.saved_count % 10 != 0):
            return

        name = f'frame_{self.saved_count:06d}'
        cv2.imwrite(os.path.join(self.out_dir, 'images/train', name + '.jpg'), frame)

        # --- YOLO 포맷 라벨 저장 (class cx cy w h, 0~1 정규화) ---
        lines = []
        for cid, x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f'{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
        with open(os.path.join(self.out_dir, 'labels/train', name + '.txt'), 'w') as f:
            f.write('\n'.join(lines))

        # --- 확인용 시각화 ---
        if self.draw_debug and boxes:
            dbg = frame.copy()
            for cid, x1, y1, x2, y2 in boxes:
                cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(dbg, CLASS_NAMES[cid], (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(self.out_dir, 'debug', name + '.jpg'), dbg)

        self.saved_count += 1
        if self.saved_count % 50 == 0:
            self.get_logger().info(f'{self.saved_count}장 저장 완료')


def main(args=None):
    rclpy.init(args=args)
    node = AutoLabeler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f'종료. 총 {node.saved_count}장 저장됨.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
