import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('line_tracing_pkg')
    default_config = os.path.join(pkg_share, 'config', 'line_tracing_params.yaml')

    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='line_tracing_node 파라미터 yaml 경로 '
                     '(조원 가제보 환경/실차용으로 복제해서 값만 바꿔 사용)',
    )

    line_tracing_node = Node(
        package='line_tracing_pkg',
        executable='line_tracing_node',
        name='line_tracing_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    return LaunchDescription([
        config_arg,
        line_tracing_node,
    ])
