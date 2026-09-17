from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    pkg_gazebo = get_package_share_directory("gazebo_ros")
    pkg_limo = get_package_share_directory("limo_description")
    pkg_arena = get_package_share_directory("webot_arena")

    world = os.path.join(
        pkg_arena,
        "worlds",
        "limo_competition.world"
    )

    xacro_file = os.path.join(
        pkg_limo,
        "urdf",
        "limo_pro_sim.xacro"
    )

    robot_description = {
        "robot_description": Command(["xacro ", xacro_file])
    }

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": world
        }.items()
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="screen"
    )

    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "limo_pro",
            "-x", "-4.0",
            "-y", "3.0",
            "-z", "0.15"
        ],
        output="screen"
    )

    return LaunchDescription([
        gazebo,
        rsp,
        spawn
    ])
