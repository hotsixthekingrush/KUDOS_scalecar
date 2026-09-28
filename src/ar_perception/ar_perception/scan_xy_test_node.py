import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanXYTestNode(Node):

    def __init__(self):
        super().__init__('scan_xy_test_node')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.get_logger().info('Scan XY ROI test node started.')

    def scan_callback(self, msg):

        points = []

        # 1. LaserScan 전체를 (x, y) 좌표로 변환
        for i, r in enumerate(msg.ranges):

            # inf, nan 제거
            if not math.isfinite(r):
                continue

            # 센서 유효 측정 범위 밖 제거
            if r < msg.range_min or r > msg.range_max:
                continue

            theta = msg.angle_min + i * msg.angle_increment

            x = r * math.cos(theta)
            y = r * math.sin(theta)

            points.append((x, y))

        # 2. 로봇 앞쪽의 넓은 ROI만 선택
        roi_points = []

        for x, y in points:

            # 앞쪽 0.1 ~ 3.0 m
            # 좌우 ±1.0 m
            if 0.1 < x < 3.0 and abs(y) < 1.0:
                roi_points.append((x, y))

        self.get_logger().info(
            f'Valid points: {len(points)} | '
            f'ROI points: {len(roi_points)}'
        )

        # 3. ROI 안에서 가장 가까운 점 확인
        if roi_points:

            nearest = min(
                roi_points,
                key=lambda p: math.hypot(p[0], p[1])
            )

            self.get_logger().info(
                f'Nearest ROI point: '
                f'x={nearest[0]:.2f} m, '
                f'y={nearest[1]:.2f} m'
            )


def main(args=None):

    rclpy.init(args=args)

    node = ScanXYTestNode()

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