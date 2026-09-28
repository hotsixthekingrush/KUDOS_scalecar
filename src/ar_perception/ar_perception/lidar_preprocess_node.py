import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker


class LidarPreprocessNode(Node):

    def __init__(self):
        super().__init__('lidar_preprocess_node')

        # /scan 구독
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # RViz 디버그용 Marker 발행
        self.marker_pub = self.create_publisher(
            Marker,
            '/perception/debug/lidar_roi',
            10
        )

        self.get_logger().info(
            'LiDAR preprocess node started.'
        )

    def scan_callback(self, msg):

        roi_points = []

        # ==============================
        # 1. LaserScan → (x, y)
        # ==============================
        for i, r in enumerate(msg.ranges):

            # inf, nan 제거
            if not math.isfinite(r):
                continue

            # 센서 유효거리 밖 제거
            if r < msg.range_min or r > msg.range_max:
                continue

            # i번째 레이저의 각도
            theta = (
                msg.angle_min
                + i * msg.angle_increment
            )

            # 극좌표 → 직교좌표
            x = r * math.cos(theta)
            y = r * math.sin(theta)

            # ==============================
            # 2. ROI
            # ==============================
            #
            # 현재는 테스트용:
            #
            # 전방 0.1 ~ 3.0 m
            # 좌우 ±1.0 m
            #
            if (
                0.1 < x < 3.0
                and -1.0 < y < 1.0
            ):
                roi_points.append((x, y))

        # ==============================
        # 3. RViz Marker 생성
        # ==============================
        marker = Marker()

        # 현재 /scan의 좌표계 그대로 사용
        marker.header.frame_id = msg.header.frame_id
        marker.header.stamp = msg.header.stamp

        marker.ns = 'lidar_roi'
        marker.id = 0

        # 여러 점을 한 번에 표시
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # 점 크기
        marker.scale.x = 0.04
        marker.scale.y = 0.04

        # 색
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # ==============================
        # 4. ROI 점들을 Marker에 추가
        # ==============================
        from geometry_msgs.msg import Point

        for x, y in roi_points:

            point = Point()

            point.x = x
            point.y = y
            point.z = 0.0

            marker.points.append(point)

        # ==============================
        # 5. RViz용 토픽으로 발행
        # ==============================
        self.marker_pub.publish(marker)

        self.get_logger().info(
            f'ROI points: {len(roi_points)}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = LidarPreprocessNode()

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

