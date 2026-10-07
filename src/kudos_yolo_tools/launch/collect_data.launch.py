"""
YOLO 학습 데이터 수집 — Gazebo 를 먼저 띄운 뒤 실행.

  터미널 1: ros2 launch limo_car ackermann_gazebo.launch.py
  터미널 2: ros2 launch kudos_yolo_tools collect_data.launch.py mode:=auto output_dir:=$HOME/yolo_data/dataset

mode:=auto  (권장) 로봇을 미션 물체 앞에 자동 순간이동시키며 수집. teleop 불필요.
            num_poses:=600 이면 약 15~20분, 클래스마다 수백 장.
mode:=drive 사람이 teleop 으로 운전하며 수집 (터미널 3: ros2 run teleop_twist_keyboard teleop_twist_keyboard)

함께 뜨는 것
  - mission_motion_controller (webot_arena) : 회전교차로 차량·차단기 움직임
  - auto_labeler                              : 이미지 + 라벨 저장
  - auto 모드: viewpoint_sampler  /  drive 모드: domain_randomizer
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    out = LaunchConfiguration('output_dir')
    return LaunchDescription([
        DeclareLaunchArgument('output_dir', default_value='./yolo_dataset'),
        DeclareLaunchArgument('mode', default_value='auto', description='auto | drive'),
        DeclareLaunchArgument('num_poses', default_value='600'),
        DeclareLaunchArgument('motion_controller', default_value='true',
                              description='이미 따로 띄웠으면 false'),

        Node(package='webot_arena', executable='mission_motion_controller.py',
             name='mission_motion_controller', output='screen',
             condition=IfCondition(LaunchConfiguration('motion_controller'))),

        Node(package='kudos_yolo_tools', executable='auto_labeler.py',
             name='auto_labeler', output='screen',
             parameters=[{'output_dir': out, 'use_sim_time': True}]),

        Node(package='kudos_yolo_tools', executable='viewpoint_sampler.py',
             name='viewpoint_sampler', output='screen',
             parameters=[{'num_poses': ParameterValue(LaunchConfiguration('num_poses'), value_type=int),
                          'use_sim_time': True}],
             condition=LaunchConfigurationEquals('mode', 'auto')),

        Node(package='kudos_yolo_tools', executable='domain_randomizer.py',
             name='domain_randomizer', output='screen',
             parameters=[{'interval_sec': 2.0, 'use_sim_time': True}],
             condition=LaunchConfigurationEquals('mode', 'drive')),
    ])
