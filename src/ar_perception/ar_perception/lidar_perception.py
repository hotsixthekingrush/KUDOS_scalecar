import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from scipy.spatial import cKDTree

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from ar_msgs.msg import Obstacle, ObstacleArray, WallArray


class LidarPerception(Node):

    def __init__(self):
        super().__init__('lidar_perception')

        # =========================================================
        # Parameters
        # =========================================================

        # LiDAR mounting position
        self.declare_parameter('lidar_offset_x', 0.0)
        self.declare_parameter('lidar_offset_y', 0.0)

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

        # =========================================================
        # Subscriber
        # =========================================================

        self.create_subscription(
            LaserScan,
            '/scan',
            self.on_scan,
            qos_profile_sensor_data
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

        self.pub_gate = self.create_publisher(
            Float32,
            '/perception/gate_distance',
            10
        )

        self.pub_walls = self.create_publisher(
            WallArray,
            '/perception/parking_walls',
            10
        )

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

    # =============================================================
    # Main callback
    # =============================================================

    def on_scan(self, scan):

        # ---------------------------------------------------------
        # 1. LaserScan -> base_link XY
        # ---------------------------------------------------------
        points = self.to_base_link(scan)

        # ---------------------------------------------------------
        # 2. Remove robot's own body
        # ---------------------------------------------------------
        filtered_points = self.remove_self_points(points)

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

            self.get_logger().info(
                f'Cluster {i}: '
                f'center=({center_x:.2f}, {center_y:.2f}) | '
                f'radius={radius:.3f} m | '
                f'residual={residual:.4f} m | '
                f'points={len(cluster)}'
                f'cone={is_cone}'
            )

        # ---------------------------------------------------------
        # 7. ObstacleArray publish
        # ---------------------------------------------------------
        obstacle_array = ObstacleArray()

        obstacle_array.header.stamp = scan.header.stamp
        obstacle_array.header.frame_id = 'base_link'

        # 아직 실제 Obstacle 생성 전
        obstacle_array.obstacles = []

        cone_margin = self.get_parameter(
            'cone_radius_margin'
        ).value

        for cluster in clusters:

            circle = self.fit_circle(cluster)

        # 원 피팅이 실패한 cluster는
        # 지금 단계에서는 Obstacle로 만들지 않는다.
            if circle is None:
                continue

            center_x, center_y, radius, residual = circle

            is_cone = self.is_cone_candidate(
                cluster,
                circle
            )

            obstacle = Obstacle()

            obstacle.x = float(center_x)
            obstacle.y = float(center_y)

    # cone이면 safety margin 추가
            if is_cone:
                obstacle.radius = float(
                    radius + cone_margin
                )
            else:
                obstacle.radius = float(radius)

            obstacle.point_count = len(cluster)


    # TODO:
    # Replace with centroid tracking velocity
    # after dynamic-object tracking is implemented.
    # 동적 장애물 tracking은 아직 구현 전
            obstacle.vx = 0.0
            obstacle.vy = 0.0
            obstacle.is_dynamic = False

            obstacle_array.obstacles.append(
                obstacle
            )

        self.pub_obstacles.publish(
            obstacle_array
        )

        # ---------------------------------------------------------
        # 8. LiDAR event publish
        # ---------------------------------------------------------
        cone_detected, cone_total, cone_left, cone_right = \
            self.detect_cone_event(clusters)
        
        event = String()

        if cone_detected:
            event.data= 'cone'
        else:
            event.data = 'none'

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

            else:

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
                    f'{cluster_id}\n'
                    f'{len(cluster)} pts\n'
                    f'r={radius:.2f}m'
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

                circle_marker.color.r = r
                circle_marker.color.g = g
                circle_marker.color.b = b
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


    def is_cone_candidate(self, cluster, circle):
        if circle is None:
            return False

        center_x, center_y, radius, residual = circle

        radius_min = self.get_parameter('cone_radius_min').value
        radius_max = self.get_parameter('cone_radius_max').value
        residual_max = self.get_parameter('cone_residual_max').value
        min_points = self.get_parameter('cone_min_points').value

        return (
            radius_min <= radius <= radius_max
            and residual <= residual_max
            and len(cluster) >= min_points
        )

    def detect_cone_event(self, clusters):
        """
        좌/우에 여러 cone 후보가 존재하는지 확인한다.

        base_link 기준:
        +y = 왼쪽
        -y = 오른쪽
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

                if not self.is_cone_candidate(cluster, circle):
                    continue

                center_x, center_y, radius, residual = circle

        # ROI 안에 있으므로 기본적으로 전방 물체지만
        # 혹시 모르니 x > 0인 물체만 사용
                if center_x <= 0.0:
                    continue

                total_count += 1

                if center_y > 0.0:
                    left_count += 1

                elif center_y < 0.0:
                    right_count += 1
        
        detected = (
                total_count >= min_total
                and left_count >= min_left
                and right_count >= min_right
        )

        return (detected, total_count, left_count, right_count)

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
