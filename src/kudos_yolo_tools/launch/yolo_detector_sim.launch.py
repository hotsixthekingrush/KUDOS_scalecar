"""
Gazebo 에서 YOLO 추론 노드 실행 (판단팀 연동 테스트용)

  ros2 launch kudos_yolo_tools yolo_detector_sim.launch.py weights:=/절대경로/best.pt
  rqt_image_view /perception/yolo_debug
  ros2 topic echo /perception/mission_objects
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('kudos_yolo_tools'),
                       'config', 'yolo_detector_sim.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('weights', description='best.pt 절대경로'),
        Node(package='kudos_yolo_tools', executable='yolo_detector_node.py',
             name='yolo_detector_node', output='screen',
             parameters=[cfg, {'weights': LaunchConfiguration('weights'),
                               'use_sim_time': True}]),
    ])
