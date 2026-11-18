import math
import json
import numpy as np

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from scipy.spatial import cKDTree

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from ar_msgs.msg import Obstacle, ObstacleArray, WallArray, WallSegment, ParkingGap
from ar_perception.point_accumulation import DEFAULTS, OdomHistory, PointAccumulator


from ar_perception.wall_filter import WALL_DEFAULTS, WallFilter


class LidarPerception(Node):

    def __init__(self):
        super().__init__('lidar_perception')

        # =========================================================
        # Parameters
        # =========================================================

        # LiDAR mounting position
        self.declare_parameter('lidar_offset_x', 0.0)
        self.declare_parameter('lidar_offset_y', 0.0)

        # Optional static-LIMO refinement; baseline LINE/CIRCLE remains the fallback.
        self.declare_parameter('use_point_accumulation', True)
        self.declare_parameter('point_accumulation_debug', True)
        for name, value in DEFAULTS.items():
            self.declare_parameter(name, value)
        self.accumulation_odom = OdomHistory()
        self.point_accumulator = PointAccumulator()
        self.accumulation_epoch = 0

        self.declare_parameter('use_wall_filter', False)
        for name, value in WALL_DEFAULTS.items():
            self.declare_parameter(name, value)
        self.wall_filter = WallFilter()

        # Self filter
        self.declare_parameter('self_x_min', -0.25)
        self.declare_parameter('self_x_max', 0.25)
        self.declare_parameter('self_y_min', -0.20)
        self.declare_parameter('self_y_max', 0.20)

        # ROI
        self.declare_parameter('roi_x_min', 0.1)
        self.declare_parameter('roi_x_max', 3.0)
        self.declare_parameter('roi_y_min', -1.0)
        self.declare_parameter('roi_y_max', 1.0)

        # Euclidean clustering
        self.declare_parameter('cluster_tolerance', 0.07)
        self.declare_parameter('cluster_min_points', 3)

        #cone candidate
        self.declare_parameter('cone_radius_min', 0.035)
        self.declare_parameter('cone_radius_max', 0.065)
        self.declare_parameter('cone_residual_max', 0.015)
        self.declare_parameter('cone_min_points', 4)

        self.declare_parameter('cone_radius_margin', 0.02)

        self.declare_parameter('cone_event_min_total', 4)
        self.declare_parameter('cone_event_min_left', 2)
        self.declare_parameter('cone_event_min_right', 2)


        self.declare_parameter('cone_near_distance', 1.0)
        self.declare_parameter('cone_far_distance', 2.0)

        self.declare_parameter('cone_min_points_near', 4)
        self.declare_parameter('cone_min_points_far', 3)

        self.declare_parameter('cone_side_deadband', 0.05)

        self.declare_parameter('parking_x_min', 0.0)
        self.declare_parameter('parking_x_max', 2.0)
        self.declare_parameter('parking_y_min', 0.20)
        self.declare_parameter('parking_y_max', 1.20)

        self.declare_parameter('parking_wall_min_length', 0.15)
        self.declare_parameter('parking_wall_residual_max', 0.03)
        self.declare_parameter('parking_wall_min_points', 5)

        self.declare_parameter('parking_boundary_merge_distance', 0.25)
        self.declare_parameter('parking_boundary_alpha', 0.2)
        self.declare_parameter('parking_gap_min_length', 0.60)
        self.declare_parameter('parking_gap_confirm_frames', 3)
        self.declare_parameter('parking_gap_match_distance', 0.20)
        self.declare_parameter('parking_gap_keep_behind_distance', 1.0)
        self.declare_parameter('parking_boundary_stale_frames', 3)
        self.declare_parameter('parking_max_boundaries', 10)

        # =========================================================
        # Subscriber
        # =========================================================

        self.create_subscription(
            LaserScan,
            '/scan',
            self.on_scan,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # =========================================================
        # Publishers
        # =========================================================

        self.pub_obstacles = self.create_publisher(
            ObstacleArray,
            '/perception/obstacles',
            10
        )

        self.pub_event = self.create_publisher(
            String,
            '/perception/event/lidar',
            10
        )

        self.pub_walls = self.create_publisher(
            WallArray,
            '/perception/parking_walls',
            10
        )

        self.pub_parking_gap = self.create_publisher(
            ParkingGap,
            '/perception/parking_gap',
            10
        )

        # Baseline-vs-refined centers and rejection reasons; existing markers unchanged.
        self.pub_accumulation_debug = self.create_publisher(
            String, '/perception/debug/limo_accumulation', 10)

        self.pub_wall_filter_debug = self.create_publisher(
            String, '/perception/debug/wall_filter', 10)

        # ROI debug
        self.pub_debug = self.create_publisher(
            Marker,
            '/perception/debug/lidar_roi',
            10
        )

        # Cluster + circle debug
        self.pub_clusters = self.create_publisher(
            MarkerArray,
            '/perception/debug/lidar_clusters',
            10
        )

        self.get_logger().info(
            'LiDAR perception node started.'
        )

        # Odometry는 주차 경계 추적을 위해 유지한다.
        self.latest_odom = None #가장 최근 자차 위치 저장
        self.parking_boundaries = []
        self.parking_frame_id = 0
        self.previous_parking_gap = None
        self.parking_gap_confirm_count = 0
        self.confirmed_parking_gap = None
        self.parking_gap_id_counter = 0
        self.active_parking_gap_id = 0

    def refine_limo_centers(self, scan, candidates, obstacle_array):
        if not self.get_parameter('use_point_accumulation').value:
            self.point_accumulator.clear()
            return
        config = {k: self.get_parameter(k).value for k in DEFAULTS}
        self.point_accumulator.cfg.update(config)
        if self.accumulation_epoch != self.accumulation_odom.epoch:
            self.point_accumulator.clear()
            self.accumulation_epoch = self.accumulation_odom.epoch
        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        pose = self.accumulation_odom.at(stamp, config['accumulation_odom_max_age_sec'])
        sensor = (self.get_parameter('lidar_offset_x').value,
                  self.get_parameter('lidar_offset_y').value)
        replacements, diagnostics = self.point_accumulator.process(stamp, pose, sensor, candidates)
        for index, candidate in replacements.items():
            obstacle_array.obstacles[index].x = candidate['x']
            obstacle_array.obstacles[index].y = candidate['y']
        if self.get_parameter('point_accumulation_debug').value:
            msg = String()
            msg.data = json.dumps({'stamp': stamp, 'frame_id': 'base_link',
                                   'candidates': diagnostics}, allow_nan=False)
            self.pub_accumulation_debug.publish(msg)

    def odom_callback(self, msg):
        self.latest_odom = msg
        q = msg.pose.pose.orientation
        roll = math.atan2(2 * (q.w*q.x + q.y*q.z), 1 - 2 * (q.x*q.x + q.y*q.y))
        pitch = math.asin(max(-1., min(1., 2 * (q.w*q.y - q.z*q.x))))
        if not all(math.isfinite(v) for v in (q.x, q.y, q.z, q.w)) or max(abs(roll), abs(pitch)) > .15:
            self.accumulation_odom.samples.clear()
            self.accumulation_odom.epoch += 1
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.accumulation_odom.add(stamp, self.get_odom_pose(),
                                   (msg.header.frame_id, msg.child_frame_id))

    # /odom의 quaternion을 yaw각도로 변환
    def get_odom_pose(self):
        if self.latest_odom is None:
            return None

        pose = self.latest_odom.pose.pose

        px = pose.position.x
        py = pose.position.y

        q = pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        yaw = math.atan2(siny_cosp, cosy_cosp)

        return px, py, yaw

    #base_link를 odem으로 좌표변환
    def base_to_odom(self, x_base, y_base):
        odom_pose = self.get_odom_pose()

        if odom_pose is None:
            return None

        px, py, yaw = odom_pose

        c = math.cos(yaw)
        s = math.sin(yaw)

        x_odom = px + c * x_base - s * y_base
        y_odom = py + s * x_base + c * y_base

        return x_odom, y_odom

    def odom_to_base(self, x_odom, y_odom):
        odom_pose = self.get_odom_pose()

        if odom_pose is None:
            return None

        px, py, yaw = odom_pose

        dx = x_odom - px
        dy = y_odom - py

        c = math.cos(yaw)
        s = math.sin(yaw)

        x_base = c * dx + s * dy
        y_base = -s * dx + c * dy

        return x_base, y_base
    
    # =============================================================
    # Main callback
    # =============================================================

    def on_scan(self, scan):
        self.parking_frame_id += 1

        # ---------------------------------------------------------
        # 1. LaserScan -> base_link XY
        # ---------------------------------------------------------
        points = self.to_base_link(scan)

        # ---------------------------------------------------------
        # 2. Remove robot's own body
        # ---------------------------------------------------------        
        filtered_points = self.remove_self_points(points)

# ---------------------------------------------------------
# Parking perception
# 주차용 ROI는 왼쪽 영역만 사용한다.
# 다른 미션 ROI와 독립적으로 항상 동작한다.
# 판단 파트가 parking state일 때만 이 결과를 사용한다.
# ---------------------------------------------------------
        parking_walls = self.detect_parking_walls(filtered_points)

        wall_array = WallArray()
        wall_array.header = scan.header
        wall_array.header.frame_id = 'base_link'
        wall_array.walls = parking_walls

        self.pub_walls.publish(wall_array)
        self.update_parking_boundaries(parking_walls)
        self.evaluate_parking_gaps(scan.header.stamp)

# ---------------------------------------------------------
# General perception ROI
# 일반 장애물 및 라바콘 인지용
# 차량 전방 좌우 전체 영역을 사용한다.
# ---------------------------------------------------------



        # ---------------------------------------------------------
        # 3. ROI filtering
        # ---------------------------------------------------------
        roi_points = self.apply_roi(filtered_points)

        # ---------------------------------------------------------
        # 4. Euclidean clustering
        # ---------------------------------------------------------
        clusters = self.euclidean_clustering(roi_points)

        # ---------------------------------------------------------
        # 5. RViz debug
        # ---------------------------------------------------------
        self.publish_debug_marker(
            scan,
            roi_points
        )

        self.publish_cluster_markers(
            scan,
            clusters
        )

        # ---------------------------------------------------------
        # 6. Circle fitting result log
        # ---------------------------------------------------------
        for i, cluster in enumerate(clusters):

            circle = self.fit_circle(cluster)

            if circle is None:
                self.get_logger().info(
                    f'Cluster {i}: '
                    f'points={len(cluster)} | '
                    f'circle fit failed'
                )
                continue

            center_x, center_y, radius, residual = circle

            # 이 cluster가 cone 후보인지 판별
            is_cone = self.is_cone_candidate(cluster, circle)

            distance = math.hypot(
            center_x,
            center_y
            )

            required_points = self.get_cone_min_points(
            distance
        )

            self.get_logger().info(
                f'Cluster {i}: '
                f'center=({center_x:.2f}, {center_y:.2f}) | '
                f'distance={distance:.2f} m | '
                f'radius={radius:.3f} m | '
                f'residual={residual:.4f} m | '
                f'points={len(cluster)} | '
                f'cone={is_cone}'
                f'required={required_points} | '
            )

        # ---------------------------------------------------------
        # 7. ObstacleArray publish
        # ---------------------------------------------------------
        
        obstacle_array = ObstacleArray()

        obstacle_array.header.stamp = scan.header.stamp
        obstacle_array.header.frame_id = 'base_link'

        cone_margin = self.get_parameter(
            'cone_radius_margin'
        ).value

        wall_enabled = self.get_parameter('use_wall_filter').value
        if wall_enabled:
            self.wall_filter.cfg.update({k: self.get_parameter(k).value for k in WALL_DEFAULTS})
            stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
            pose = self.accumulation_odom.at(stamp, self.wall_filter.cfg['wall_odom_max_age_sec'])
            self.wall_filter.update(stamp, pose, filtered_points, self.accumulation_odom.epoch)
        else:
            self.wall_filter.clear()
        wall_diagnostics = []
        limo_candidates = []
        # M7: 한 클러스터에서 Obstacle 최대 하나 발행.
        # 라바콘은 기존 값을 유지하고, LIMO 후보만 차체 중심으로 변환한다.
        for cluster in clusters:
            circle = self.fit_circle(cluster)
            is_cone = self.is_cone_candidate(cluster, circle)
            limo = None if is_cone else self.estimate_limo_candidate(cluster, circle)

            # 그 외 클러스터는 차량/라바콘으로 검증되지 않았으므로 발행하지 않는다.
            if not is_cone and limo is None:
                continue

            if limo is not None and wall_enabled:
                rejected, reason = self.wall_filter.reject(cluster)
                wall_diagnostics.append(dict(x=float(limo['x']), y=float(limo['y']),
                                             method=limo['method'], rejected=rejected, reason=reason))
                if rejected:
                    continue

            obstacle = Obstacle()
            if limo is not None:
                obstacle.x = float(limo['x'])
                obstacle.y = float(limo['y'])
                obstacle.radius = 0.16
                limo_candidates.append((len(obstacle_array.obstacles), cluster, limo))
            else:
                center_x, center_y, radius, residual = circle
                obstacle.x = float(center_x)
                obstacle.y = float(center_y)
                obstacle.radius = float(radius + cone_margin if is_cone else radius)

            obstacle.point_count = len(cluster)
            obstacle_array.obstacles.append(obstacle)

        # Refinement never creates/deletes detections or changes cone outputs.
        self.refine_limo_centers(scan, limo_candidates, obstacle_array)

        # 정적 장애물 인터페이스 유지: 속도 필드만 0으로 채운다.
        for obstacle in obstacle_array.obstacles:
            obstacle.vx = 0.0
            obstacle.vy = 0.0
            obstacle.is_dynamic = False

        self.pub_obstacles.publish(obstacle_array)
        if wall_enabled:
            debug = String()
            debug.data = json.dumps(dict(stamp=stamp, frame_id='base_link',
                walls=[dict(endpoints=w.tolist(), source=source) for w, source in self.wall_filter.walls],
                candidates=wall_diagnostics), allow_nan=False)
            self.pub_wall_filter_debug.publish(debug)

        # ---------------------------------------------------------
        # 8. LiDAR event publish
        # ---------------------------------------------------------
        cone_detected, cone_total, cone_left, cone_right = \
            self.detect_cone_event(clusters)

        event = String()
        event.data = 'cone' if cone_detected else 'none'
        self.pub_event.publish(event)

        self.get_logger().info(
            f'Cone candidates: total={cone_total} | '
            f'left={cone_left} | '
            f'right={cone_right} | '
            f'event={event.data}'
        )        

        # ---------------------------------------------------------
        # 9. Summary log
        # ---------------------------------------------------------
        cluster_sizes = [
            len(cluster)
            for cluster in clusters
        ]

        self.get_logger().info(
            f'Valid points: {len(points)} | '
            f'After self filter: {len(filtered_points)} | '
            f'ROI points: {len(roi_points)} | '
            f'Clusters: {len(clusters)} | '
            f'Sizes: {cluster_sizes}'
        )

    # =============================================================
    # LaserScan -> XY
    # =============================================================

    def to_base_link(self, scan):

        points = []

        offset_x = self.get_parameter(
            'lidar_offset_x'
        ).value

        offset_y = self.get_parameter(
            'lidar_offset_y'
        ).value

        for i, r in enumerate(scan.ranges):

            # inf / nan 제거
            if not math.isfinite(r):
                continue

            # 센서 측정 범위 밖 제거
            if (
                r < scan.range_min
                or r > scan.range_max
            ):
                continue

            angle = (
                scan.angle_min
                + i * scan.angle_increment
            )

            # LiDAR frame 기준 XY
            x_laser = r * math.cos(angle)
            y_laser = r * math.sin(angle)

            # base_link 기준으로 이동
            x_base = x_laser + offset_x
            y_base = y_laser + offset_y

            points.append(
                (x_base, y_base)
            )

        return points

    # =============================================================
    # Self filter
    # =============================================================

    def remove_self_points(self, points):

        x_min = self.get_parameter(
            'self_x_min'
        ).value

        x_max = self.get_parameter(
            'self_x_max'
        ).value

        y_min = self.get_parameter(
            'self_y_min'
        ).value

        y_max = self.get_parameter(
            'self_y_max'
        ).value

        filtered_points = []

        for x, y in points:

            inside_robot = (
                x_min <= x <= x_max
                and y_min <= y <= y_max
            )

            if not inside_robot:
                filtered_points.append(
                    (x, y)
                )

        return filtered_points

    # =============================================================
    # ROI
    # =============================================================

    def apply_roi(self, points):

        x_min = self.get_parameter(
            'roi_x_min'
        ).value

        x_max = self.get_parameter(
            'roi_x_max'
        ).value

        y_min = self.get_parameter(
            'roi_y_min'
        ).value

        y_max = self.get_parameter(
            'roi_y_max'
        ).value

        roi_points = []

        for x, y in points:

            if (
                x_min < x < x_max
                and y_min < y < y_max
            ):
                roi_points.append(
                    (x, y)
                )

        return roi_points

    # =============================================================
    # Euclidean clustering
    # =============================================================

    def euclidean_clustering(self, points):

        if not points:
            return []

        tolerance = self.get_parameter(
            'cluster_tolerance'
        ).value

        min_points = self.get_parameter(
            'cluster_min_points'
        ).value

        # 주변 점 검색용 KD-tree
        tree = cKDTree(points)

        visited = [False] * len(points)

        clusters = []

        for start_idx in range(len(points)):

            if visited[start_idx]:
                continue

            queue = [start_idx]
            visited[start_idx] = True

            cluster_indices = []

            while queue:

                current_idx = queue.pop()

                cluster_indices.append(
                    current_idx
                )

                neighbors = tree.query_ball_point(
                    points[current_idx],
                    tolerance
                )

                for neighbor_idx in neighbors:

                    if not visited[neighbor_idx]:

                        visited[neighbor_idx] = True

                        queue.append(
                            neighbor_idx
                        )

            # 너무 작은 cluster는 noise로 제거
            if len(cluster_indices) >= min_points:

                cluster = [
                    points[i]
                    for i in cluster_indices
                ]

                clusters.append(
                    cluster
                )

        return clusters

    # =============================================================
    # Simple cluster center
    # =============================================================

    def get_cluster_center(self, cluster):

        if not cluster:
            return 0.0, 0.0

        sum_x = 0.0
        sum_y = 0.0

        for x, y in cluster:
            sum_x += x
            sum_y += y

        center_x = sum_x / len(cluster)
        center_y = sum_y / len(cluster)

        return center_x, center_y

    # =============================================================
    # Circle fitting
    # =============================================================

    def fit_circle(self, cluster):

        # 원을 정의하려면 최소 3점 필요
        if len(cluster) < 3:
            return None

        x = np.array(
            [p[0] for p in cluster],
            dtype=float
        )

        y = np.array(
            [p[1] for p in cluster],
            dtype=float
        )

        # 원 방정식:
        #
        # (x-cx)^2 + (y-cy)^2 = r^2
        #
        # 선형식으로 정리
        A = np.column_stack((
            2.0 * x,
            2.0 * y,
            np.ones_like(x)
        ))

        b = x**2 + y**2

        try:

            solution, _, _, _ = np.linalg.lstsq(
                A,
                b,
                rcond=None
            )

            center_x = solution[0]
            center_y = solution[1]
            c = solution[2]

            radius_squared = (
                c
                + center_x**2
                + center_y**2
            )

            if radius_squared <= 0.0:
                return None

            radius = math.sqrt(
                radius_squared
            )

            # 각 점이 원에서 얼마나 벗어났는지
            distances = np.sqrt(
                (x - center_x)**2
                + (y - center_y)**2
            )

            residual = np.mean(
                np.abs(
                    distances - radius
                )
            )

            return (
                float(center_x),
                float(center_y),
                float(radius),
                float(residual)
            )

        except np.linalg.LinAlgError:

            return None
    def fit_line(self, cluster):

        if len(cluster) < 2:
            return None

        pts = np.array(
            cluster,
            dtype=float
        )

        center = np.mean(
            pts,
            axis=0
        )

        centered = pts - center

        # PCA
        covariance = np.cov(
            centered.T
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        # 가장 분산이 큰 방향 = 선 방향
        direction = eigenvectors[
            :,
            np.argmax(eigenvalues)
        ]

        dx = direction[0]
        dy = direction[1]

        # 각 점을 선 방향에 투영
        projections = (
            centered @ direction
        )

        length = (
            np.max(projections)
            - np.min(projections)
        )

        # 선에 수직인 방향
        normal = np.array([
            -dy,
            dx
        ])

        distances = np.abs(
            centered @ normal
        )

        residual = np.mean(
            distances
        )

        angle = math.atan2(
            dy,
            dx
        )

        angle = abs(angle)

        # 선 방향은 +방향/-방향 구분이 없으므로
        # 0~90도로 정규화
        if angle > math.pi / 2:
            angle = math.pi - angle

        angle_deg = math.degrees(
            angle
        )

        return (
            float(center[0]),
            float(center[1]),
            float(length),
            float(angle_deg),
            float(residual)
        )

    
    # =============================================================
    # ROI RViz debug
    # =============================================================

    def publish_debug_marker(
        self,
        scan,
        points
    ):

        marker = Marker()

        marker.header.stamp = scan.header.stamp
        marker.header.frame_id = 'base_link'

        marker.ns = 'lidar_roi'
        marker.id = 0

        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.04
        marker.scale.y = 0.04

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for x, y in points:

            point = Point()

            point.x = float(x)
            point.y = float(y)
            point.z = 0.0

            marker.points.append(
                point
            )

        self.pub_debug.publish(
            marker
        )

    # =============================================================
    # Cluster + circle RViz debug
    # =============================================================

    def publish_cluster_markers(
        self,
        scan,
        clusters
    ):

        marker_array = MarkerArray()

        # 이전 frame marker 삭제
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL

        marker_array.markers.append(
            delete_marker
        )

        colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ]

        for cluster_id, cluster in enumerate(clusters):

            r, g, b = colors[
                cluster_id % len(colors)
            ]

            # -----------------------------------------------------
            # Circle fitting 먼저 계산
            # -----------------------------------------------------
            circle = self.fit_circle(
                cluster
            )

            if circle is not None:

                center_x, center_y, radius, residual = circle
                is_cone = self.is_cone_candidate(cluster, circle)

            else:
                is_cone = False

                center_x, center_y = self.get_cluster_center(
                    cluster
                )

                radius = None
                residual = None

            # -----------------------------------------------------
            # Cluster points
            # -----------------------------------------------------
            point_marker = Marker()

            point_marker.header.stamp = scan.header.stamp
            point_marker.header.frame_id = 'base_link'

            point_marker.ns = 'lidar_cluster_points'
            point_marker.id = cluster_id

            point_marker.type = Marker.POINTS
            point_marker.action = Marker.ADD

            point_marker.scale.x = 0.05
            point_marker.scale.y = 0.05

            point_marker.color.r = r
            point_marker.color.g = g
            point_marker.color.b = b
            point_marker.color.a = 1.0

            for x, y in cluster:

                p = Point()

                p.x = float(x)
                p.y = float(y)
                p.z = 0.0

                point_marker.points.append(
                    p
                )

            marker_array.markers.append(
                point_marker
            )

            # M7: 실제 발행하는 LIMO 후보 중심을 RViz에 표시한다.
            limo = self.estimate_limo_candidate(cluster, circle)
            if limo is not None:
                label = Marker()
                label.header.stamp = scan.header.stamp
                label.header.frame_id = 'base_link'
                label.ns = 'lidar_limo_candidates'
                label.id = 3000 + cluster_id
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = float(limo['x'])
                label.pose.position.y = float(limo['y'])
                label.pose.position.z = 0.35
                label.pose.orientation.w = 1.0
                label.scale.z = 0.12
                label.color.r = 0.0
                label.color.g = 0.8
                label.color.b = 1.0
                label.color.a = 1.0
                label.text = (
                    f"LIMO? {limo['method']} "
                    f"L={limo['length']:.2f} e={limo['residual']:.3f}"
                )
                marker_array.markers.append(label)

                center_marker = Marker()
                center_marker.header.stamp = scan.header.stamp
                center_marker.header.frame_id = 'base_link'
                center_marker.ns = 'lidar_limo_centers'
                center_marker.id = 4000 + cluster_id
                center_marker.type = Marker.SPHERE
                center_marker.action = Marker.ADD
                center_marker.pose.position.x = float(limo['x'])
                center_marker.pose.position.y = float(limo['y'])
                center_marker.pose.position.z = 0.06
                center_marker.pose.orientation.w = 1.0
                center_marker.scale.x = 0.055
                center_marker.scale.y = 0.055
                center_marker.scale.z = 0.055
                center_marker.color.r = 0.0
                center_marker.color.g = 0.8
                center_marker.color.b = 1.0
                center_marker.color.a = 1.0
                marker_array.markers.append(center_marker)

            # -----------------------------------------------------
            # Cluster text
            # -----------------------------------------------------
            text_marker = Marker()

            text_marker.header.stamp = scan.header.stamp
            text_marker.header.frame_id = 'base_link'

            text_marker.ns = 'lidar_cluster_text'
            text_marker.id = 1000 + cluster_id

            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(center_x)
            text_marker.pose.position.y = float(center_y)
            text_marker.pose.position.z = 0.15

            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.15

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            if radius is not None:

                text_marker.text = (
                    f'{cluster_id} {"CONE" if is_cone else "OTHER"}\n'
                    f'{len(cluster)} pts\n'
                    f'r={radius:.3f}m e={residual:.3f}m'
                )

            else:

                text_marker.text = (
                    f'{cluster_id}\n'
                    f'{len(cluster)} pts\n'
                    f'fit fail'
                )

            marker_array.markers.append(
                text_marker
            )

            # -----------------------------------------------------
            # Fitted circle
            # -----------------------------------------------------
            if radius is not None:

                circle_marker = Marker()

                circle_marker.header.stamp = scan.header.stamp
                circle_marker.header.frame_id = 'base_link'

                circle_marker.ns = 'lidar_cluster_circles'
                circle_marker.id = 2000 + cluster_id

                circle_marker.type = Marker.LINE_STRIP
                circle_marker.action = Marker.ADD

                circle_marker.scale.x = 0.015

                circle_marker.color.r = 0.0 if is_cone else 0.6
                circle_marker.color.g = 1.0 if is_cone else 0.6
                circle_marker.color.b = 0.0 if is_cone else 0.6
                circle_marker.color.a = 1.0

                circle_segments = 40

                for j in range(
                    circle_segments + 1
                ):

                    theta = (
                        2.0
                        * math.pi
                        * j
                        / circle_segments
                    )

                    p = Point()

                    p.x = (
                        center_x
                        + radius
                        * math.cos(theta)
                    )

                    p.y = (
                        center_y
                        + radius
                        * math.sin(theta)
                    )

                    p.z = 0.02

                    circle_marker.points.append(
                        p
                    )

                marker_array.markers.append(
                    circle_marker
                )

        self.pub_clusters.publish(
            marker_array
        )


    def estimate_limo_candidate(self, cluster, circle=None):
        """M7 초기 가설: 선형 단일면 또는 비선형 2면 관측의 차체 중심 추정.

        반환값은 확정 차량 ID가 아닌 후보이다. base_link의 원점이 주행 LIMO에
        가까운 위치라는 전제로, 표면 법선의 부호는 원점에서 멀어지는 쪽을 택한다.
        """
        if len(cluster) < 8:
            return None
        if circle is None:
            circle = self.fit_circle(cluster)
        if self.is_cone_candidate(cluster, circle):
            return None

        line = self.fit_line(cluster)
        if line is None:
            return None
        _, _, length, _, line_residual = line

        pts = np.asarray(cluster, dtype=float)
        midpoint = np.mean(pts, axis=0)

        # 단일면: PCA 선분의 두 끝점 사이 중간 위치를 구한다.
        if 0.15 <= length <= 0.40 and line_residual < 0.015:
            centered = pts - np.mean(pts, axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            tangent = vh[0]
            projections = centered @ tangent
            midpoint = np.mean(pts, axis=0) + (
                0.5 * (float(projections.min()) + float(projections.max()))
            ) * tangent
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
            # 실제 LiDAR 센서 위치를 기준으로 표면 법선 방향 결정
            sensor = np.array([
                self.get_parameter('lidar_offset_x').value,
                self.get_parameter('lidar_offset_y').value
            ], dtype=float)

            if float(np.dot(midpoint - sensor, normal)) < 0.0:
                normal = -normal
            offset = 0.16 if length <= 0.27 else 0.11
            center = midpoint + offset * normal
            return {
                'x': float(center[0]), 'y': float(center[1]),
                'method': 'LINE', 'length': float(length),
                'residual': float(line_residual),
            }

        # 2면 관측: 기존 원 피팅을 그대로 사용하되, 차량 후보에만 적용.
        # 직선이 아닌지, 실제 x/y 방향으로 퍼진 점들이 있는지 검사한다.
        if circle is None:
            return None
        cx, cy, radius, circle_residual = circle
        spread_x = float(np.ptp(pts[:, 0]))
        spread_y = float(np.ptp(pts[:, 1]))
        if (
            0.10 <= radius <= 0.40
            and circle_residual <= 0.04
            and 0.06 <= spread_x <= 0.45
            and 0.06 <= spread_y <= 0.45
            and line_residual >= 0.015
        ):
            return {
                'x': float(cx), 'y': float(cy),
                'method': 'CIRCLE', 'length': float(length),
                'residual': float(circle_residual),
            }
        return None

    def is_cone_candidate(self, cluster, circle):
        if circle is None:
            return False

        center_x, center_y, radius, residual = circle

        distance = math.hypot(
            center_x,
            center_y
        )

        min_points = self.get_cone_min_points(
            distance
        )

        radius_min = self.get_parameter('cone_radius_min').value
        radius_max = self.get_parameter('cone_radius_max').value
        residual_max = self.get_parameter('cone_residual_max').value

        return (
            radius_min <= radius <= radius_max
            and residual <= residual_max
            and len(cluster) >= min_points
        )

    def get_cone_min_points(self, distance):
        near_distance = self.get_parameter(
            'cone_near_distance'
        ).value

        far_distance = self.get_parameter(
            'cone_far_distance'
        ).value

        near_points = int(
            self.get_parameter(
                'cone_min_points_near'
            ).value
        )

        far_points = int(
            self.get_parameter(
                'cone_min_points_far'
            ).value
        )

        if distance <= near_distance:
            return near_points

        if distance >= far_distance:
            return far_points

    # 중간 거리에서는 보수적으로 가까운 기준 사용
        return near_points
    


    def detect_cone_event(self, clusters):
        """
        좌/우에 여러 cone 후보가 존재하는지 확인한다.
        """

        min_total = self.get_parameter(
            'cone_event_min_total'
        ).value

        min_left = self.get_parameter(
            'cone_event_min_left'
        ).value

        min_right = self.get_parameter(
            'cone_event_min_right'
        ).value

        left_count = 0
        right_count = 0
        total_count = 0

        for cluster in clusters:

            circle = self.fit_circle(cluster)

            if circle is None:
                continue

            if not self.is_cone_candidate(
                cluster,
                circle
            ):
                continue

            center_x, center_y, radius, residual = circle


            
            if center_x <= 0.0:
                continue

            self.get_logger().info(
                f'Cone side check: '
                f'x={center_x:.2f} | '
                f'y={center_y:.2f} | '
                f'side={"LEFT" if center_y > 0.0 else "RIGHT"}'
            )
            
            total_count += 1

            side_deadband = float(
                self.get_parameter('cone_side_deadband').value
            )

            if center_y > side_deadband:
                left_count += 1
                side = 'LEFT'

            elif center_y < -side_deadband:
                right_count += 1
                side = 'RIGHT'

            else:
                side = 'CENTER'

            self.get_logger().info(
                f'Cone side check: '
                f'x={center_x:.2f} | '
                f'y={center_y:.2f} | '
                f'side={side}'
            )



        detected = (
            total_count >= min_total
            and left_count >= min_left
            and right_count >= min_right
        )

        return (
            detected,
            total_count,
            left_count,
            right_count
        )




    def apply_parking_roi(self, points):
        x_min = self.get_parameter('parking_x_min').value
        x_max = self.get_parameter('parking_x_max').value
        y_min = self.get_parameter('parking_y_min').value
        y_max = self.get_parameter('parking_y_max').value

        return [
            (x, y)
            for x, y in points
            if x_min <= x <= x_max
            and y_min <= y <= y_max
        ]

    def fit_wall_segment(self, cluster):
        if len(cluster) < 2:
            return None

        pts = np.array(cluster, dtype=float)

        center = np.mean(pts, axis=0)
        centered = pts - center

        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        direction = eigenvectors[:, np.argmax(eigenvalues)]
        dx, dy = direction

        projections = centered @ direction

        t_min = np.min(projections)
        t_max = np.max(projections)

        p1 = center + t_min * direction
        p2 = center + t_max * direction

        length = t_max - t_min

        normal = np.array([-dy, dx])
        residual = np.mean(np.abs(centered @ normal))

        theta = math.atan2(dy, dx)

        return (
            float(p1[0]), float(p1[1]),
            float(p2[0]), float(p2[1]),
            float(theta),
            float(length),
            float(residual)
        )


    def extract_parking_lines(self, points):
        """
        Parking ROI 안에서 RANSAC을 반복하여
        서로 연결된 여러 벽을 각각의 직선으로 분리한다.
        """

        if len(points) < 2:
            return []

        remaining = np.array(points, dtype=float)

        max_lines = 4
        max_iterations = 100

        distance_threshold = 0.025
        min_inliers = 8

        lines = []

    # 매 frame에서 결과가 과하게 랜덤하게 바뀌는 것을 방지
        rng = np.random.default_rng(0)

        for _ in range(max_lines):

            if len(remaining) < min_inliers:
                break

            best_indices = None
            best_count = 0

            for _ in range(max_iterations):

                indices = rng.choice(
                    len(remaining),
                    size=2,
                    replace=False
                )

                p1 = remaining[indices[0]]
                p2 = remaining[indices[1]]

                direction = p2 - p1
                norm = np.linalg.norm(direction)

                if norm < 1e-6:
                    continue

                direction = direction / norm

            # 직선의 법선벡터
                normal = np.array([
                    -direction[1],
                    direction[0]
                ])

                distances = np.abs(
                    (remaining - p1) @ normal
                )

                inlier_indices = np.where(
                    distances <= distance_threshold
                )[0]

                if len(inlier_indices) > best_count:
                    best_count = len(inlier_indices)
                    best_indices = inlier_indices

            if (
                best_indices is None
                or len(best_indices) < min_inliers
            ):
                break

            line_points = remaining[best_indices]

        # 기존 함수로 최종 선분 fitting
            segment = self.fit_wall_segment(line_points)

            if segment is not None:
                lines.append(
                    (line_points, segment)
                )

        # 찾은 직선의 점들을 제거
            mask = np.ones(
                len(remaining),
                dtype=bool
            )

            mask[best_indices] = False
            remaining = remaining[mask]

        return lines

    def update_parking_boundaries(self, walls):
        """
        ROLE_SIDE 벽을 odom 좌표 기준으로 저장하고,
        같은 경계가 반복 검출되면 EMA smoothing으로 안정화한다.
        """
        merge_distance = float(
            self.get_parameter('parking_boundary_merge_distance').value
        )
        alpha = float(
            self.get_parameter('parking_boundary_alpha').value
        )
        alpha = min(max(alpha, 0.0), 1.0)
        max_boundaries = int(
            self.get_parameter('parking_max_boundaries').value
        )

        if max_boundaries <= 0:
            max_boundaries = 1

        for wall in walls:
            if wall.role != WallSegment.ROLE_SIDE:
                continue

            center_x = (wall.x1 + wall.x2) / 2.0
            center_y = (wall.y1 + wall.y2) / 2.0

            odom_pos = self.base_to_odom(center_x, center_y)
            if odom_pos is None:
                continue

            x_odom, y_odom = odom_pos
            matched_idx = self.find_matching_parking_boundary(x_odom, y_odom)

            if matched_idx is not None:
                old_x = self.parking_boundaries[matched_idx]['x']
                old_y = self.parking_boundaries[matched_idx]['y']

                new_x = (1.0 - alpha) * old_x + alpha * x_odom
                new_y = (1.0 - alpha) * old_y + alpha * y_odom

                self.parking_boundaries[matched_idx]['x'] = new_x
                self.parking_boundaries[matched_idx]['y'] = new_y
                self.parking_boundaries[matched_idx]['last_seen_frame'] = self.parking_frame_id

                self.get_logger().info(
                    f'Parking boundary MATCH: '
                    f'odom=({x_odom:.2f}, {y_odom:.2f}) '
                    f'matched={matched_idx}'
                )
                continue

            self.parking_boundaries.append({
                'x': x_odom,
                'y': y_odom,
                'last_seen_frame': self.parking_frame_id,
            })

            self.get_logger().info(
                f'Parking boundary NEW: '
                f'odom=({x_odom:.2f}, {y_odom:.2f}) '
                f'total_boundaries={len(self.parking_boundaries)}'
            )

            if len(self.parking_boundaries) > max_boundaries:
                self.parking_boundaries = self.parking_boundaries[-max_boundaries:]
                self.get_logger().info(
                    f'Parking boundaries trimmed to {max_boundaries}'
                )

    def find_matching_parking_boundary(self, x, y):
        """
        odom 기준으로 가장 가까운 저장된 경계벽과의 거리 비교.
        """
        if not self.parking_boundaries:
            return None

        merge_distance = float(
            self.get_parameter('parking_boundary_merge_distance').value
        )

        best_idx = None
        best_dist = float('inf')

        for idx, boundary in enumerate(self.parking_boundaries):
            dist = math.hypot(
                x - boundary['x'],
                y - boundary['y']
            )

            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is not None and best_dist <= merge_distance:
            return best_idx

        return None

    def select_parking_gap(self, gaps):
        """
        후보 gap 중 현재 차량 앞쪽의 가장 가까운 하나를 선택한다.
        """
        if not gaps:
            return None

        odom_pose = self.get_odom_pose()
        if odom_pose is None:
            return None

        px, py, yaw = odom_pose
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)

        best_gap = None
        best_score = float('inf')

        for gap in gaps:
            x1 = gap['x1']
            y1 = gap['y1']
            x2 = gap['x2']
            y2 = gap['y2']

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            dx = center_x - px
            dy = center_y - py
            ahead_distance = dx * forward_x + dy * forward_y

            if ahead_distance <= 0.0:
                continue

            score = ahead_distance
            if score < best_score:
                best_score = score
                best_gap = gap

        if best_gap is not None:
            return best_gap

        return None

    def update_parking_gap_confirmation(self, gap, scan_stamp):
        """
        연속 frame confirm count를 관리하고,
        확정된 gap을 publish한다.
        """
        confirm_frames = int(
            self.get_parameter('parking_gap_confirm_frames').value
        )

        if confirm_frames <= 0:
            confirm_frames = 1

        if gap is None:
            self.previous_parking_gap = None
            self.parking_gap_confirm_count = 0
            self.confirmed_parking_gap = None
            self.active_parking_gap_id = 0
            self.publish_parking_gap(scan_stamp, False, False, 0)
            return None

        center_x = gap['center_x']
        center_y = gap['center_y']
        length = gap['length']

        match_distance = float(
            self.get_parameter('parking_gap_match_distance').value
        )
        same_gap = False
        if self.previous_parking_gap is not None:
            distance = math.hypot(
                center_x - self.previous_parking_gap['center_x'],
                center_y - self.previous_parking_gap['center_y']
            )
            same_gap = distance <= match_distance

        if same_gap:
            self.parking_gap_confirm_count += 1
        else:
            self.parking_gap_confirm_count = 1
            self.confirmed_parking_gap = None
            self.parking_gap_id_counter += 1
            self.active_parking_gap_id = self.parking_gap_id_counter

        self.previous_parking_gap = {
            'center_x': center_x,
            'center_y': center_y,
        }

        if self.confirmed_parking_gap is None:
            self.get_logger().info(
                f'Parking gap tracking: '
                f'length={length:.2f} m | '
                f'confirm={self.parking_gap_confirm_count}/{confirm_frames}'
            )

        if self.confirmed_parking_gap is None and self.parking_gap_confirm_count >= confirm_frames:
            self.confirmed_parking_gap = {
                'center_x': center_x,
                'center_y': center_y,
                'length': length,
            }
            self.parking_gap_confirm_count = confirm_frames
            self.get_logger().info(
                f'Parking gap CONFIRMED: '
                f'center=({center_x:.2f}, {center_y:.2f}) | '
                f'length={length:.2f} m'
            )
            self.publish_parking_gap(
                scan_stamp, True, True, self.active_parking_gap_id,
                center_x, center_y, length
            )
            return self.confirmed_parking_gap

        if self.confirmed_parking_gap is not None:
            self.publish_parking_gap(
                scan_stamp, True, True, self.active_parking_gap_id,
                self.confirmed_parking_gap['center_x'],
                self.confirmed_parking_gap['center_y'],
                self.confirmed_parking_gap['length']
            )
            return self.confirmed_parking_gap

        self.publish_parking_gap(
            scan_stamp, True, False, self.active_parking_gap_id,
            center_x, center_y, length
        )
        return None

    def publish_parking_gap(self, scan_stamp, valid, confirmed,gap_id, center_x_odom=0.0,
        center_y_odom=0.0,
        length=0.0,
        depth=0.0
    ):
        msg = ParkingGap()

        msg.header.stamp = scan_stamp
        msg.header.frame_id = 'base_link'

        msg.valid = bool(valid)
        msg.confirmed = bool(confirmed)
        msg.gap_id = int(gap_id)

        if valid:
            base_pos = self.odom_to_base(
                center_x_odom,
                center_y_odom
            )

            if base_pos is None:
                msg.valid = False
                msg.confirmed = False
                msg.center_x = 0.0
                msg.center_y = 0.0

            else:
                center_x_base, center_y_base = base_pos

                msg.center_x = float(center_x_base)
                msg.center_y = float(center_y_base)

        else:
            msg.center_x = 0.0
            msg.center_y = 0.0

        if not msg.valid:
            msg.confirmed = False
            msg.gap_id = 0
        msg.length = float(length) if msg.valid else 0.0
        msg.depth = 0.0

        self.pub_parking_gap.publish(msg)




    def evaluate_parking_gaps(self, scan_stamp):
        """
        parking boundary를 주행방향 기준으로 정렬하고,
        인접한 pair만 projected gap을 계산한 뒤
        confirm count와 최종 gap을 관리한다.
        """
        if self.confirmed_parking_gap is not None:
            gap = self.confirmed_parking_gap
            base_pos = self.odom_to_base(gap['center_x'], gap['center_y'])
            keep_behind = float(
                self.get_parameter('parking_gap_keep_behind_distance').value
            )
            if base_pos is None or base_pos[0] >= -keep_behind:
                self.publish_parking_gap(
                    scan_stamp, True, True, self.active_parking_gap_id,
                    gap['center_x'], gap['center_y'], gap['length']
                )
                return []
            self.previous_parking_gap = None
            self.parking_gap_confirm_count = 0
            self.confirmed_parking_gap = None
            self.active_parking_gap_id = 0

        stale_frames = int(
            self.get_parameter('parking_boundary_stale_frames').value
        )
        active_boundaries = [
            boundary for boundary in self.parking_boundaries
            if self.parking_frame_id - boundary['last_seen_frame'] <= stale_frames
        ]
        # No new ROLE_SIDE observation: stored boundaries cannot confirm a gap.
        if len(active_boundaries) < 2 or not any(
            boundary['last_seen_frame'] == self.parking_frame_id
            for boundary in active_boundaries
        ):
            self.update_parking_gap_confirmation(None, scan_stamp)
            return []

        odom_pose = self.get_odom_pose()
        if odom_pose is None:
            forward_x, forward_y = 1.0, 0.0
        else:
            _, _, yaw = odom_pose
            forward_x = math.cos(yaw)
            forward_y = math.sin(yaw)

        ordered = sorted(
            active_boundaries,
            key=lambda b: (
                b['x'] * forward_x + b['y'] * forward_y
            )
        )

        min_gap = float(
            self.get_parameter('parking_gap_min_length').value
        )

        candidate_gaps = []

        for i in range(len(ordered) - 1):
            boundary_0 = ordered[i]
            boundary_1 = ordered[i + 1]

            # An unrelated observed wall must not confirm an unobserved pair.
            if (
                boundary_0['last_seen_frame'] != self.parking_frame_id
                and boundary_1['last_seen_frame'] != self.parking_frame_id
            ):
                continue

            dx = boundary_1['x'] - boundary_0['x']
            dy = boundary_1['y'] - boundary_0['y']

            euclidean_length = math.hypot(dx, dy)
            projected_length = abs(
                dx * forward_x + dy * forward_y
            )

            if projected_length < min_gap:
                continue

            center_x = (boundary_0['x'] + boundary_1['x']) / 2.0
            center_y = (boundary_0['y'] + boundary_1['y']) / 2.0

            candidate = {
                'x1': boundary_0['x'],
                'y1': boundary_0['y'],
                'x2': boundary_1['x'],
                'y2': boundary_1['y'],
                'center_x': center_x,
                'center_y': center_y,
                'length': projected_length,
                'euclidean_length': euclidean_length,
            }

            candidate_gaps.append(candidate)

            self.get_logger().info(
                f'Parking gap: '
                f'boundary_{i}=({boundary_0["x"]:.2f}, {boundary_0["y"]:.2f}) | '
                f'boundary_{i + 1}=({boundary_1["x"]:.2f}, {boundary_1["y"]:.2f}) | '
                f'projected_length={projected_length:.2f} m | '
                f'euclidean_length={euclidean_length:.2f} m | '
                f'candidate=True'
            )

        selected_gap = self.select_parking_gap(candidate_gaps)
        confirmed_gap = self.update_parking_gap_confirmation(selected_gap, scan_stamp)

        if confirmed_gap is not None:
            self.confirmed_parking_gap = confirmed_gap

        return candidate_gaps

    def classify_wall_role(self, theta):
        """
        벽 선분의 방향각 theta를 이용해
        주차 벽의 역할을 분류한다.

        base_link 기준:
        +x = 차량 전방
        +y = 차량 왼쪽
        """

        angle_deg = abs(math.degrees(theta)) % 180.0

    # 직선은 방향이 반대여도 같은 선이므로
    # 0~90도 범위로 정규화
        if angle_deg > 90.0:
            angle_deg = 180.0 - angle_deg

        if angle_deg <= 20.0:
            return WallSegment.ROLE_BACK

        if angle_deg >= 70.0:
            return WallSegment.ROLE_SIDE

        return WallSegment.ROLE_UNKNOWN

    def detect_parking_walls(self, points):

        parking_points = self.apply_parking_roi(points)

        self.get_logger().info(
            f'Parking ROI: input={len(points)} | '
            f'Parking_points={len(parking_points)}'
        )

        if len(parking_points) < 2:
            return []

        line_results = self.extract_parking_lines(
            parking_points
        )

        min_length = self.get_parameter(
            'parking_wall_min_length'
        ).value

        residual_max = self.get_parameter(
            'parking_wall_residual_max'
        ).value

        min_points = self.get_parameter(
            'parking_wall_min_points'
        ).value

        walls = []

        for i, result in enumerate(line_results):

            line_points, segment = result

            (
                x1, y1,
                x2, y2,
                theta,
                length,
                residual
            ) = segment

            candidate = (
                len(line_points) >= min_points
                and length >= min_length
                and residual <= residual_max
            )

            role = self.classify_wall_role(theta)

            self.get_logger().info(
                f'Parking wall {i}: '
                f'length={length:.2f} | '
                f'theta={math.degrees(theta):.1f} deg | '
                f'residual={residual:.4f} | '
                f'points={len(line_points)} | '
                f'role={role} | '
                f'candidate={candidate}'
            )

            if not candidate:
                continue

            wall = WallSegment()

            wall.x1 = x1
            wall.y1 = y1
            wall.x2 = x2
            wall.y2 = y2

            wall.theta = theta
            wall.length = length

            wall.role = role

            walls.append(wall)

        return walls


def main(args=None):

    rclpy.init(
        args=args
    )

    node = LidarPerception()

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
