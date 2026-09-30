#!/usr/bin/env python3
"""
auto_labeler.py
Gazebo 시뮬레이션에서 카메라 이미지와 YOLO 라벨을 자동으로 동시 생성하는 노드.

[원리]
Gazebo 는 모든 물체의 3D 위치·크기를 알고 있다. 물체의 3D 박스 8꼭짓점을
카메라 내부 파라미터로 이미지에 투영하고, 그 볼록껍질을 라벨로 저장한다.

[v2 변경점 — 9/28 리뷰 반영]
  1) 카메라 위치를 odom TF 대신 Gazebo 정답 위치(/gazebo/model_states 의 로봇 모델)로 계산.
     기존 방식(world_frame=odom)은 로봇이 원점이 아닌 곳에서 스폰되거나 odom 이
     조금만 틀어져도 박스가 전부 어긋났다. (Gazebo 물체 좌표는 world 기준이므로)
  2) 대상 목록을 config/targets.yaml 로 분리. marker_right 대상이 빠져 있던 문제 수정.
  3) 깊이 영상으로 가림(occlusion) 검사 — 벽/터널/다른 물체 뒤에 숨은 물체는 라벨 제외.
  4) 파일명에 세션 접두어 추가 — 재실행 시 기존 데이터를 덮어쓰던 문제 수정.
     (split_dataset.py 가 이 접두어로 세션/시간 단위 분할을 한다)
  5) 라벨을 다각형(segmentation 형식)으로 저장 — detect 학습에도 그대로 쓰이고,
     차단기 각도용 seg 모델 학습에도 쓸 수 있다. (label_format:=box 로 기존 형식 가능)

[출력]
  dataset/images/all/<세션>_000001.jpg
  dataset/labels/all/<세션>_000001.txt
  dataset/debug/<세션>_000001.jpg        <- 라벨을 그린 확인용 이미지
  → 수집 후 split_dataset.py 로 train/val 을 나눈다.

[사전 준비] 월드 파일 <world> 안에 추가
  <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
    <ros><namespace>/gazebo</namespace></ros>
    <update_rate>30.0</update_rate>
  </plugin>

[실행]
  ros2 run kudos_yolo_tools auto_labeler.py --ros-args -p output_dir:=$HOME/yolo_data/dataset

  기본값은 팀 git 의 `ros2 launch limo_car ackermann_gazebo.launch.py` 기준:
    토픽 /rgb/image_raw, /rgb/camera_info, /depth_camera/depth/image_raw (9/29 실측), 로봇 엔티티 mbot,
    카메라 frame depth_link (광학 좌표계, limo_car/gazebo/sensor.xacro 에서 rpy -90,0,-90)
    ※ 토픽 이름은 `ros2 topic list | grep -E "rgb|depth"` 로 꼭 한 번 확인할 것
"""

import os
import time
import random

import numpy as np
import cv2
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException

import yolo_geometry as geo


# 너무 작게 보이는 물체는 학습에 방해 — 제외 (픽셀)
MIN_BOX_PX = 12
# 화면 밖으로 잘려나간 비율이 이보다 크면 제외
MAX_CLIP_RATIO = 0.6


def default_targets_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('kudos_yolo_tools'),
                            'config', 'targets.yaml')
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'config', 'targets.yaml')


def load_targets(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    classes = list(cfg['classes'])
    targets = []
    for t in cfg['targets']:
        if t['class'] not in classes:
            raise ValueError('targets.yaml: 알 수 없는 class "%s"' % t['class'])
        item = {
            'class_id': classes.index(t['class']),
            'size': [float(v) for v in t['size']],
            'offset': [float(v) for v in t.get('offset', [0, 0, 0])],
            # 앞면 방향 (모델 좌표계 축). 지정하면 그 쪽에서 볼 때만 라벨을 붙인다.
            # 표지판 뒷면은 그림이 좌우 반전되어 보이므로(좌회전 -> 우회전처럼) 반드시 필요
            'front': t.get('front', ''),
        }
        if 'world_pose' in t:
            item['world_pose'] = [float(v) for v in t['world_pose']]
        else:
            item['gazebo_name'] = t['gazebo_name']
        targets.append(item)
    return classes, targets


def tf_to_T(tf_msg):
    q = tf_msg.transform.rotation
    t = tf_msg.transform.translation
    return geo.make_T(geo.quat_to_rot(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])


def pose_to_T(pose):
    q = pose.orientation
    p = pose.position
    return geo.make_T(geo.quat_to_rot(q.x, q.y, q.z, q.w), [p.x, p.y, p.z])


class AutoLabeler(Node):
    def __init__(self):
        super().__init__('auto_labeler')

        p = self.declare_parameter
        p('image_topic', '/rgb/image_raw')
        p('camera_info_topic', '/rgb/camera_info')
        p('depth_topic', '/depth_camera/depth/image_raw')          # '' 이면 가림 검사 안 함
        p('output_dir', './dataset')
        p('session', '')                 # 비우면 시각으로 자동 생성 (예: 0928_201530)
        p('targets_file', '')            # 비우면 패키지 config/targets.yaml

        # --- 카메라 위치 계산 방식 ---
        #   gazebo : 로봇 정답 위치(model_states) x 로봇->카메라 정적 TF   ← 권장
        #   tf     : world_frame -> 카메라 TF 를 직접 조회 (world TF 가 있을 때만)
        p('camera_pose_source', 'gazebo')
        p('robot_model_name', 'mbot')           # ackermann_gazebo.launch.py 의 -entity
        p('robot_root_frame', 'base_footprint')  # 로봇 URDF 의 최상위 링크
        p('world_frame', 'world')               # camera_pose_source=tf 일 때만 사용
        # 이미지 header 의 frame(depth_camera_frame_optical)은 URDF/TF 에 없다.
        # 같은 위치·같은 방향(광학 좌표계)인 depth_link 를 대신 쓴다. (limo_car/gazebo/sensor.xacro)
        p('camera_frame', 'depth_link')         # '' 이면 이미지 header.frame_id 사용
        # 이미지 frame 이 *_optical_frame 이 아니라 카메라 "링크"(x 전방)라면 false
        p('camera_frame_is_optical', True)

        p('save_every_n', 1)            # N프레임마다 1장 (Gazebo 카메라 10Hz → 5장/초)
        p('negative_ratio', 0.1)        # 대상이 하나도 없는 프레임 저장 확률
        p('max_occluded_ratio', 0.5)    # 이보다 많이 가려지면 라벨 제외
        p('occlusion_tol_m', 0.05)
        p('label_format', 'polygon')    # polygon | box
        p('draw_debug', True)
        # 장면이 바뀌는 순간(순간이동·마커 교체)에는 이미지와 물체 위치가 어긋날 수 있다.
        # viewpoint_sampler / domain_randomizer 가 이 토픽으로 저장을 잠시 멈춘다.
        p('enable_topic', '/auto_labeler/enable')

        g = lambda n: self.get_parameter(n).value
        self.image_topic = g('image_topic')
        self.info_topic = g('camera_info_topic')
        self.depth_topic = g('depth_topic')
        self.out_dir = os.path.expanduser(g('output_dir'))
        self.session = g('session') or time.strftime('%m%d_%H%M%S')
        tfile = g('targets_file') or default_targets_file()
        self.pose_source = g('camera_pose_source')
        self.robot_name = g('robot_model_name')
        self.root_frame = g('robot_root_frame')
        self.world_frame = g('world_frame')
        self.cam_frame_param = g('camera_frame')
        self.is_optical = bool(g('camera_frame_is_optical'))
        self.save_every_n = int(g('save_every_n'))
        self.neg_ratio = float(g('negative_ratio'))
        self.max_occ = float(g('max_occluded_ratio'))
        self.occ_tol = float(g('occlusion_tol_m'))
        self.label_fmt = g('label_format')
        self.draw_debug = bool(g('draw_debug'))
        self.enabled = True
        self.enable_stamp_ns = 0   # 저장 재개 시각(시뮬 시간). 이보다 먼저 찍힌 사진은 버림

        self.class_names, self.targets = load_targets(tfile)
        self.get_logger().info('대상 %d개 로드: %s' % (len(self.targets), tfile))

        for sub in ('images/all', 'labels/all', 'debug'):
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

        self.bridge = CvBridge()
        self.K = None
        self.model_states = {}
        self.depth = None
        self.T_root_cam = None      # 정적이므로 한 번만 조회
        self.frame_idx = 0
        self.saved = 0
        self.class_counts = [0] * len(self.class_names)
        self.occ_dropped = 0
        self._warned_models = False

        qos = qos_profile_sensor_data
        self.create_subscription(CameraInfo, self.info_topic, self.info_cb, qos)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.states_cb, qos)
        self.create_subscription(Image, self.image_topic, self.image_cb, qos)
        if g('enable_topic'):
            self.create_subscription(Bool, g('enable_topic'), self.enable_cb, 10)
        if self.depth_topic:
            self.create_subscription(Image, self.depth_topic, self.depth_cb, qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info('auto_labeler 시작 | 세션=%s | 저장=%s | 카메라위치=%s'
                               % (self.session, self.out_dir, self.pose_source))

    # ------------------------------------------------------------------
    def info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)
            self.get_logger().info('CameraInfo 수신 fx=%.1f fy=%.1f cx=%.1f cy=%.1f'
                                   % (self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]))

    def states_cb(self, msg):
        self.model_states = dict(zip(msg.name, msg.pose))

    def depth_cb(self, msg):
        try:
            d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth = geo.depth_to_meters(d, msg.encoding)
        except Exception as e:
            self.get_logger().warn('깊이 변환 실패: %s' % e, throttle_duration_sec=10.0)

    # ------------------------------------------------------------------
    def camera_T_cam_world(self, cam_frame):
        """월드 점 -> 카메라 광학 좌표로 보내는 변환 T_cam_world. 실패 시 None."""
        if self.pose_source == 'tf':
            try:
                T_world_cam = tf_to_T(self.tf_buffer.lookup_transform(
                    self.world_frame, cam_frame, rclpy.time.Time()))
            except TransformException as e:
                self.get_logger().warn('TF %s->%s 실패: %s' % (self.world_frame, cam_frame, e),
                                       throttle_duration_sec=5.0)
                return None
        else:
            robot = self.model_states.get(self.robot_name)
            if robot is None:
                if self.model_states and not self._warned_models:
                    self._warned_models = True
                    self.get_logger().error(
                        'model_states 에 "%s" 없음. robot_model_name 을 다음 중 하나로: %s'
                        % (self.robot_name, ', '.join(sorted(self.model_states.keys()))))
                return None
            if self.T_root_cam is None:
                try:
                    self.T_root_cam = tf_to_T(self.tf_buffer.lookup_transform(
                        self.root_frame, cam_frame, rclpy.time.Time()))
                    if not self.is_optical:
                        self.T_root_cam = self.T_root_cam @ geo.T_LINK_FROM_OPTICAL
                    self.get_logger().info('로봇(%s)->카메라(%s) 정적 TF 확보'
                                           % (self.root_frame, cam_frame))
                except TransformException as e:
                    self.get_logger().warn(
                        'TF %s->%s 실패: %s  (ros2 run tf2_tools view_frames 로 이름 확인)'
                        % (self.root_frame, cam_frame, e), throttle_duration_sec=5.0)
                    return None
            T_world_cam = pose_to_T(robot) @ self.T_root_cam
        return geo.inv_T(T_world_cam)

    def target_corners_world(self, tgt):
        if 'world_pose' in tgt:
            x, y, z, yaw = tgt['world_pose']
            R = geo.yaw_to_rot(yaw)
            origin = np.array([x, y, z])
        else:
            pose = self.model_states.get(tgt['gazebo_name'])
            if pose is None:
                return None
            T = pose_to_T(pose)
            R = T[:3, :3]
            origin = T[:3, 3]
        center = origin + R @ np.array(tgt['offset'])
        return geo.cuboid_corners(center, tgt['size'], R), center, R

    # ------------------------------------------------------------------
    def enable_cb(self, msg):
        on = bool(msg.data)
        if on and not self.enabled:
            # use_sim_time 이면 시뮬 시간. Gazebo 가 느려도 "재개 이후에 찍힌 사진"만 받는다.
            self.enable_stamp_ns = self.get_clock().now().nanoseconds
        self.enabled = on

    def image_cb(self, msg):
        if not self.enabled:
            return
        stamp_ns = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        if stamp_ns < self.enable_stamp_ns:
            return   # 장면을 바꾸기 전에 찍혀서 늦게 도착한 사진
        self.frame_idx += 1
        if self.frame_idx % self.save_every_n != 0:
            return
        if self.K is None:
            self.get_logger().warn('CameraInfo 미수신 — %s 확인. 저장 안 됨.' % self.info_topic,
                                   throttle_duration_sec=5.0)
            return

        cam_frame = self.cam_frame_param or msg.header.frame_id
        T_cam_world = self.camera_T_cam_world(cam_frame)
        if T_cam_world is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error('이미지 변환 실패: %s' % e)
            return
        h, w = frame.shape[:2]
        cam_pos = geo.inv_T(T_cam_world)[:3, 3]   # 카메라 월드 위치

        depth = None
        if self.depth_topic and self.depth is None:
            self.get_logger().warn(
                '깊이 영상 미수신(%s) — 가림 검사가 꺼진 채 저장 중. '
                '`ros2 topic list | grep depth` 로 실제 토픽 이름 확인' % self.depth_topic,
                throttle_duration_sec=5.0)
        if self.depth is not None:
            depth = self.depth
            if depth.shape[:2] != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        labels = []   # (class_id, poly)
        for tgt in self.targets:
            got = self.target_corners_world(tgt)
            if got is None:
                continue
            corners, center, R_obj = got
            if tgt['front'] and not geo.is_front_facing(tgt['front'], center, R_obj, cam_pos):
                continue
            pts_cam = geo.transform_points(T_cam_world, corners)
            uv = geo.project_points(self.K, pts_cam)
            if uv is None:
                continue
            poly, keep = geo.clipped_hull(uv, w, h)
            if poly is None or keep < (1.0 - MAX_CLIP_RATIO):
                continue
            x1, y1, x2, y2 = geo.poly_bbox(poly)
            if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
                continue
            if depth is not None:
                occ = geo.occluded_ratio(depth, poly, float(pts_cam[:, 2].min()), self.occ_tol)
                if occ is not None and occ > self.max_occ:
                    self.occ_dropped += 1
                    continue
            labels.append((tgt['class_id'], poly))

        if not labels and random.random() > self.neg_ratio:
            return

        stem = '%s_%06d' % (self.session, self.saved)
        cv2.imwrite(os.path.join(self.out_dir, 'images/all', stem + '.jpg'), frame)
        with open(os.path.join(self.out_dir, 'labels/all', stem + '.txt'), 'w') as f:
            f.write('\n'.join(geo.yolo_line(c, pl, w, h, self.label_fmt) for c, pl in labels))

        if self.draw_debug and labels:
            dbg = frame.copy()
            for c, pl in labels:
                cv2.polylines(dbg, [np.round(pl).astype(np.int32)], True, (0, 255, 0), 2)
                x1, y1, _, _ = geo.poly_bbox(pl)
                cv2.putText(dbg, self.class_names[c], (int(x1), max(int(y1) - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(self.out_dir, 'debug', stem + '.jpg'), dbg)

        for c, _ in labels:
            self.class_counts[c] += 1
        self.saved += 1
        if self.saved % 50 == 0:
            self.get_logger().info('%d장 저장 | 클래스별 %s | 가림 제외 %d'
                                   % (self.saved, self.count_str(), self.occ_dropped))

    def count_str(self):
        return ', '.join('%s=%d' % (n, c) for n, c in zip(self.class_names, self.class_counts))


def main(args=None):
    rclpy.init(args=args)
    node = AutoLabeler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('종료. 총 %d장 | %s' % (node.saved, node.count_str()))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
