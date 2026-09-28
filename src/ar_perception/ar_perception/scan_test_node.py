import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanTestNode(Node):

    def __init__(self):
        super().__init__('scan_test_node')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.get_logger().info('Scan test node started.')

    def scan_callback(self, msg):

        front_index = int(
            round((0.0 - msg.angle_min) / msg.angle_increment)
        )

        if front_index < 0 or front_index >= len(msg.ranges):
            self.get_logger().warn('Front angle is outside scan range.')
            return

        front_distance = msg.ranges[front_index]

        if math.isfinite(front_distance):
            self.get_logger().info(
                f'Front distance: {front_distance:.2f} m'
            )
        else:
            self.get_logger().info(
                'Front distance: no obstacle detected'
            )


def main(args=None):
    rclpy.init(args=args)

    node = ScanTestNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    