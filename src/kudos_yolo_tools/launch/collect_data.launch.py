"""
YOLO 학습 데이터 수집 — Gazebo 를 먼저 띄운 뒤 실행.

  터미널 1: ros2 launch limo_car ackermann_gazebo.launch.py
  터미널 2: ros2 launch kudos_yolo_tools collect_data.launch.py output_dir:=$HOME/yolo_data/dataset
  터미널 3: ros2 run teleop_twist_keyboard teleop_twist_keyboard   (로봇 몰기)

함께 뜨는 것
  - mission_motion_controller (webot_arena) : 회전교차로 차량·차단기 움직임
  - domain_randomizer                        : 좌/우 마커 맞바꾸기, 주차 표지판 흔들기
  - auto_labeler                              : 이미지 + 라벨 저장
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    out = LaunchConfiguration('output_dir')
    motion = LaunchConfiguration('motion_controller')
    return LaunchDescription([
        DeclareLaunchArgument('output_dir', default_value='./yolo_dataset'),
        DeclareLaunchArgument('motion_controller', default_value='true',
                              description='이미 따로 띄웠으면 false'),
        Node(package='webot_arena', executable='mission_motion_controller.py',
             name='mission_motion_controller', output='screen',
             condition=IfCondition(motion)),
        Node(package='kudos_yolo_tools', executable='domain_randomizer.py',
             name='domain_randomizer', output='screen',
             parameters=[{'interval_sec': 2.0}]),
        Node(package='kudos_yolo_tools', executable='auto_labeler.py',
             name='auto_labeler', output='screen',
             parameters=[{'output_dir': out}]),
    ])
