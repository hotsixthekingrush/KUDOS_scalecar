#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class LabMeasureNode(Node):

    def __init__(self):
        super().__init__('lab_measure_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/rgb/image_raw',
            self.image_callback,
            10
        )

        self.frame_count = 0

        self.get_logger().info(
            'LAB measurement started: /rgb/image_raw'
        )

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        self.frame_count += 1

        # 너무 자주 출력하지 않도록 약 30프레임마다 측정
        if self.frame_count % 30 != 0:
            return

        # 기존 HSV 노란색 검출을 샘플 선택에만 사용
        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        lower_yellow = np.array(
            [15, 80, 80],
            dtype=np.uint8
        )

        upper_yellow = np.array(
            [35, 255, 255],
            dtype=np.uint8
        )

        yellow_mask = cv2.inRange(
            hsv,
            lower_yellow,
            upper_yellow
        )

        # 실제 같은 픽셀들을 LAB으로 변환
        lab = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2LAB
        )

        pixels = lab[
            yellow_mask > 0
        ]

        if len(pixels) < 50:
            self.get_logger().warn(
                f'Yellow pixels too few: {len(pixels)}'
            )
            return

        L = pixels[:, 0]
        A = pixels[:, 1]
        B = pixels[:, 2]

        def stats(channel):
            return (
                int(np.percentile(channel, 5)),
                int(np.percentile(channel, 50)),
                int(np.percentile(channel, 95))
            )

        l5, l50, l95 = stats(L)
        a5, a50, a95 = stats(A)
        b5, b50, b95 = stats(B)

        self.get_logger().info(
            '\n'
            f'Yellow pixels = {len(pixels)}\n'
            f'L : p05={l5}, p50={l50}, p95={l95}\n'
            f'A : p05={a5}, p50={a50}, p95={a95}\n'
            f'B : p05={b5}, p50={b50}, p95={b95}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = LabMeasureNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
