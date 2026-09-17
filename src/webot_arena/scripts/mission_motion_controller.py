#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from rcl_interfaces.msg import SetParametersResult


# ============================================================
# USER TUNING / 사용자 수정 영역
# ============================================================
# 단위:
#   obstacle_speed      -> m/s
#   obstacle_radius     -> m
#   barrier_speed_deg   -> deg/s
#   barrier_wait_*      -> s
#
# 실행 중에도 아래처럼 동적장애물 속도/반경을 바꿀 수 있다.
#   ros2 param set /mission_motion_controller obstacle_speed 0.15
#   ros2 param set /mission_motion_controller obstacle_radius 0.68
DEFAULT_OBSTACLE_SPEED = 0.20
DEFAULT_OBSTACLE_RADIUS = 0.68

DEFAULT_BARRIER_SPEED_DEG = 30.0
DEFAULT_BARRIER_WAIT_CLOSED = 3.0
DEFAULT_BARRIER_WAIT_OPEN = 3.0


class MissionMotionController(Node):

    def __init__(self):
        super().__init__('mission_motion_controller')

        # =====================================================
        # ROS parameters
        # =====================================================

        self.declare_parameter('obstacle_speed', DEFAULT_OBSTACLE_SPEED)
        self.declare_parameter('obstacle_radius', DEFAULT_OBSTACLE_RADIUS)

        self.declare_parameter('barrier_speed_deg', DEFAULT_BARRIER_SPEED_DEG)
        self.declare_parameter('barrier_wait_closed', DEFAULT_BARRIER_WAIT_CLOSED)
        self.declare_parameter('barrier_wait_open', DEFAULT_BARRIER_WAIT_OPEN)

        self.obstacle_speed = float(
            self.get_parameter('obstacle_speed').value
        )

        self.obstacle_radius = float(
            self.get_parameter('obstacle_radius').value
        )

        # 실행 중 ros2 param set으로 속도/반경 변경 가능
        self.add_on_set_parameters_callback(
            self.parameter_callback
        )

        self.barrier_speed = math.radians(
            float(self.get_parameter('barrier_speed_deg').value)
        )

        self.barrier_wait_closed = float(
            self.get_parameter('barrier_wait_closed').value
        )

        self.barrier_wait_open = float(
            self.get_parameter('barrier_wait_open').value
        )

        # =====================================================
        # Gazebo state service
        # =====================================================

        self.client = self.create_client(
            SetEntityState,
            '/gazebo/set_entity_state'
        )

        self.get_logger().info(
            'Waiting for /gazebo/set_entity_state ...'
        )

        while not self.client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Still waiting for Gazebo state service...'
            )

        # pending service calls
        self.pending = {}

        # =====================================================
        # Barrier
        # =====================================================

        self.barrier_name = 'moving_barrier_arm'

        self.barrier_hinge_x = 3.55
        self.barrier_hinge_y = -3.12
        self.barrier_hinge_z = 0.48

        self.barrier_length = 1.25

        self.barrier_closed_angle = 0.0
        self.barrier_open_angle = math.radians(88.0)

        self.barrier_angle = self.barrier_closed_angle

        self.barrier_state = 'WAIT_CLOSED'
        self.barrier_state_time = self.get_clock().now()

        # =====================================================
        # Roundabout
        # =====================================================

        self.roundabout_cx = -2.15
        self.roundabout_cy = 0.20

        self.obstacles = [
            {
                'name': 'roundabout_dynamic_obstacle_1',
                'angle': 0.0
            },
            {
                'name': 'roundabout_dynamic_obstacle_2',
                'angle': math.pi
            },
        ]

        self.obstacle_z = 0.0

        # 20 Hz
        # 50 Hz보다 Gazebo service 부하를 줄여 버벅임 감소
        self.dt = 0.05

        self.timer = self.create_timer(
            self.dt,
            self.update
        )

        self.get_logger().info(
            f'Mission controller started | '
            f'obstacle speed = {self.obstacle_speed:.2f} m/s'
        )

    def parameter_callback(self, params):

        for param in params:

            if param.name == 'obstacle_speed':

                speed = float(param.value)

                if speed < 0.0 or speed > 1.0:
                    return SetParametersResult(
                        successful=False,
                        reason='obstacle_speed must be 0.0 ~ 1.0 m/s'
                    )

                self.obstacle_speed = speed

                self.get_logger().info(
                    f'Obstacle speed changed -> {speed:.2f} m/s'
                )

            elif param.name == 'obstacle_radius':

                radius = float(param.value)

                if radius <= 0.1:
                    return SetParametersResult(
                        successful=False,
                        reason='obstacle_radius is too small'
                    )

                self.obstacle_radius = radius

                self.get_logger().info(
                    f'Obstacle radius changed -> {radius:.2f} m'
                )

        return SetParametersResult(successful=True)


    # =========================================================
    # Gazebo service
    # =========================================================

    def send_state(
        self,
        name,
        x,
        y,
        z,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=1.0
    ):

        old_future = self.pending.get(name)

        # 이전 요청이 아직 끝나지 않았다면 새 요청을 보내지 않음
        if old_future is not None and not old_future.done():
            return

        request = SetEntityState.Request()

        state = EntityState()

        state.name = name

        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)

        state.pose.orientation.x = float(qx)
        state.pose.orientation.y = float(qy)
        state.pose.orientation.z = float(qz)
        state.pose.orientation.w = float(qw)

        state.reference_frame = 'world'

        request.state = state

        self.pending[name] = self.client.call_async(
            request
        )

    # =========================================================
    # Barrier
    # =========================================================

    def update_barrier(self):

        now = self.get_clock().now()

        elapsed = (
            now - self.barrier_state_time
        ).nanoseconds / 1e9

        if self.barrier_state == 'WAIT_CLOSED':

            self.barrier_angle = (
                self.barrier_closed_angle
            )

            if elapsed >= self.barrier_wait_closed:

                self.barrier_state = 'OPENING'
                self.barrier_state_time = now

                self.get_logger().info(
                    'Barrier: OPENING'
                )

        elif self.barrier_state == 'OPENING':

            self.barrier_angle += (
                self.barrier_speed * self.dt
            )

            if (
                self.barrier_angle
                >= self.barrier_open_angle
            ):

                self.barrier_angle = (
                    self.barrier_open_angle
                )

                self.barrier_state = 'WAIT_OPEN'
                self.barrier_state_time = now

                self.get_logger().info(
                    'Barrier: WAIT_OPEN'
                )

        elif self.barrier_state == 'WAIT_OPEN':

            if elapsed >= self.barrier_wait_open:

                self.barrier_state = 'CLOSING'
                self.barrier_state_time = now

                self.get_logger().info(
                    'Barrier: CLOSING'
                )

        elif self.barrier_state == 'CLOSING':

            self.barrier_angle -= (
                self.barrier_speed * self.dt
            )

            if (
                self.barrier_angle
                <= self.barrier_closed_angle
            ):

                self.barrier_angle = (
                    self.barrier_closed_angle
                )

                self.barrier_state = 'WAIT_CLOSED'
                self.barrier_state_time = now

                self.get_logger().info(
                    'Barrier: WAIT_CLOSED'
                )

        half = self.barrier_length / 2.0

        # =====================================================
        # 중요:
        # 도로는 hinge_y=-3.12에서 -Y 방향에 있음.
        #
        # 따라서 닫힌 상태에서 arm 중심이
        # hinge보다 작은 Y값으로 들어가야 함.
        # =====================================================

        y = (
            self.barrier_hinge_y
            - half * math.cos(self.barrier_angle)
        )

        z = (
            self.barrier_hinge_z
            + half * math.sin(self.barrier_angle)
        )

        # -Y에 놓인 봉이 위로 올라가도록
        # X축 음의 방향으로 회전
        roll = -self.barrier_angle

        qx = math.sin(roll / 2.0)
        qw = math.cos(roll / 2.0)

        self.send_state(
            self.barrier_name,
            self.barrier_hinge_x,
            y,
            z,
            qx=qx,
            qw=qw
        )

    # =========================================================
    # Dynamic obstacles
    # =========================================================

    def update_dynamic_obstacles(self):

        # v = r * omega
        omega = (
            self.obstacle_speed
            / self.obstacle_radius
        )

        for obstacle in self.obstacles:

            obstacle['angle'] += (
                omega * self.dt
            )

            if obstacle['angle'] >= 2.0 * math.pi:
                obstacle['angle'] -= 2.0 * math.pi

            theta = obstacle['angle']

            x = (
                self.roundabout_cx
                + self.obstacle_radius
                * math.cos(theta)
            )

            y = (
                self.roundabout_cy
                + self.obstacle_radius
                * math.sin(theta)
            )

            # 차량 앞방향을 원의 접선 방향으로
            yaw = theta + math.pi / 2.0

            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)

            self.send_state(
                obstacle['name'],
                x,
                y,
                self.obstacle_z,
                qz=qz,
                qw=qw
            )

    # =========================================================

    def update(self):

        self.update_barrier()
        self.update_dynamic_obstacles()


def main():

    rclpy.init()

    node = MissionMotionController()

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

