#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState


class BarrierController(Node):

    def __init__(self):
        super().__init__('barrier_controller')

        # 움직일 Gazebo model 이름
        self.model_name = 'moving_barrier_arm'

        # make_arena.py의 create_barrier() 값과 맞춰야 함
        self.hinge_x = 3.55
        self.hinge_y = -3.12
        self.hinge_z = 0.48

        self.arm_length = 1.25

        # 0 deg = 내려가 있음
        # 88 deg = 거의 수직으로 올라감
        self.closed_angle = 0.0
        self.open_angle = math.radians(88.0)

        self.angle = self.closed_angle

        # 회전 속도
        self.angular_speed = math.radians(35.0)

        # 닫힘/열림 유지시간
        self.closed_wait = 3.0
        self.open_wait = 3.0

        self.state = 'WAIT_CLOSED'
        self.state_start_time = self.get_clock().now()

        self.client = self.create_client(
            SetEntityState,
            '/gazebo/set_entity_state'
        )

        self.get_logger().info(
            'Waiting for /gazebo/set_entity_state ...'
        )

        self.client.wait_for_service()

        self.get_logger().info(
            'Barrier controller started.'
        )

        self.dt = 0.02

        self.timer = self.create_timer(
            self.dt,
            self.update
        )

    def quaternion_from_roll(self, roll):

        qx = math.sin(roll / 2.0)
        qy = 0.0
        qz = 0.0
        qw = math.cos(roll / 2.0)

        return qx, qy, qz, qw

    def elapsed(self):

        now = self.get_clock().now()

        return (
            now - self.state_start_time
        ).nanoseconds / 1e9

    def change_state(self, new_state):

        self.state = new_state
        self.state_start_time = self.get_clock().now()

        self.get_logger().info(
            f'Barrier state: {new_state}'
        )

    def send_pose(self):

        half = self.arm_length / 2.0

        # hinge 기준 회전
        center_x = self.hinge_x

        center_y = (
            self.hinge_y
            - half * math.cos(self.angle)
        )

        center_z = (
            self.hinge_z
            + half * math.sin(self.angle)
        )

        qx, qy, qz, qw = self.quaternion_from_roll(
            self.angle
        )

        request = SetEntityState.Request()

        request.state = EntityState()

        request.state.name = self.model_name

        request.state.pose.position.x = center_x
        request.state.pose.position.y = center_y
        request.state.pose.position.z = center_z

        request.state.pose.orientation.x = qx
        request.state.pose.orientation.y = qy
        request.state.pose.orientation.z = qz
        request.state.pose.orientation.w = qw

        request.state.reference_frame = 'world'

        self.client.call_async(request)

    def update(self):

        if self.state == 'WAIT_CLOSED':

            self.angle = self.closed_angle

            if self.elapsed() >= self.closed_wait:
                self.change_state('OPENING')

        elif self.state == 'OPENING':

            self.angle += self.angular_speed * self.dt

            if self.angle >= self.open_angle:

                self.angle = self.open_angle
                self.change_state('WAIT_OPEN')

        elif self.state == 'WAIT_OPEN':

            self.angle = self.open_angle

            if self.elapsed() >= self.open_wait:
                self.change_state('CLOSING')

        elif self.state == 'CLOSING':

            self.angle -= self.angular_speed * self.dt

            if self.angle <= self.closed_angle:

                self.angle = self.closed_angle
                self.change_state('WAIT_CLOSED')

        self.send_pose()


def main(args=None):

    rclpy.init(args=args)

    node = BarrierController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
