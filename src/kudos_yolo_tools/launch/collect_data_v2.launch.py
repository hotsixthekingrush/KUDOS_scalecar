"""
YOLO 학습 데이터 수집 v2 — 새 맵(KUDOS_Autorace2026_Gazebo) 용. Gazebo 를 먼저 띄운 뒤 실행.

  터미널 1: source ~/26v2/KUDOS_Autorace2026_Gazebo/install/setup.bash
            ros2 launch kudos_arena competition.launch.py branch_sign:=A     (A 또는 B)
  터미널 2: source ~/26contest/install/setup.bash
            source ~/26v2/KUDOS_Autorace2026_Gazebo/install/setup.bash
            ros2 launch kudos_yolo_tools collect_data_v2.launch.py output_dir:=$HOME/yolo_data/v2/sim_m2m4

branch_sign:=auto (기본)
  /tmp/kudos_arena_*/track_2026.world 중 가장 최근 파일에서 갈림길 표지판 그림(Kudos/Sign_A|B)을
  읽어 targets_v2_A.yaml / targets_v2_B.yaml 을 자동으로 고른다. 직접 정하려면 branch_sign:=A|B.
  저장 파일 이름 앞에 A_ / B_ 가 붙어서 나중에 어느 쪽 실행분인지 확인할 수 있다.

mode:=auto  (권장) 로봇을 표지판 앞에 자동 순간이동시키며 수집 (viewpoint_sampler)
mode:=drive 사람이 teleop 으로 운전하며 수집 (domain_randomizer)

v1(collect_data.launch.py)과 다른 점
  - 옛 맵의 mission_motion_controller(회전교차로·차단기)를 띄우지 않음
  - 표지판 맞바꾸기(swap)·흔들기(jitter) 끔: 새 맵에는 맞바꿀 짝이 없고,
    옛 흔들기 목록이 parking_sign_panel 판만 기둥에서 떼어 움직이기 때문
  - targets / viewpoints 를 v2 파일로 지정

필요 조건: 월드에 gazebo_ros_state 플러그인 (/gazebo/model_states, /gazebo/set_entity_state)
          → kudos_arena/scripts/make_arena.py 의 build_world() 에 추가해야 함
"""
import glob
import os
import re
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def detect_branch_sign(pattern):
    """가장 최근 world 파일에서 갈림길 표지판 그림(A/B)을 읽는다."""
    worlds = glob.glob(os.path.expanduser(pattern))
    if not worlds:
        raise RuntimeError('world 파일을 못 찾음 (%s). Gazebo 를 먼저 띄우거나 branch_sign:=A|B 로 직접 지정'
                           % pattern)
    path = max(worlds, key=os.path.getmtime)
    with open(path) as f:
        text = f.read()
    block = re.search(r'<model name="branch_sign_panel">(.*?)</model>', text, re.S)
    if block is None:
        raise RuntimeError('%s 에 branch_sign_panel 모델이 없음' % path)
    sign = re.search(r'Kudos/Sign_([AB])\b', block.group(1))
    if sign is None:
        raise RuntimeError('%s 의 branch_sign_panel 에서 Kudos/Sign_A|B 를 못 찾음' % path)
    return sign.group(1), path


def setup(context):
    get = lambda name: LaunchConfiguration(name).perform(context).strip()
    actions = []

    sign = get('branch_sign').upper()
    if sign == 'AUTO':
        sign, path = detect_branch_sign(get('world_glob'))
        actions.append(LogInfo(msg='[collect_v2] 갈림길 표지판 = %s  (%s 에서 읽음)' % (sign, path)))
    if sign not in ('A', 'B'):
        raise RuntimeError('branch_sign 은 auto, A, B 중 하나 (받은 값: %s)' % sign)

    share = get_package_share_directory('kudos_yolo_tools')
    targets = os.path.join(share, 'config', 'targets_v2_%s.yaml' % sign)
    viewpoints = os.path.join(share, 'config', 'viewpoints_v2.yaml')
    out = os.path.expanduser(get('output_dir'))
    session = '%s_%s' % (sign, time.strftime('%m%d_%H%M%S'))
    mode = get('mode')
    robot = get('robot')          # Gazebo 안 로봇 이름 (model_states 에 나오는 이름)
    actions.append(LogInfo(msg='[collect_v2] targets=%s | 저장=%s | 세션=%s' % (targets, out, session)))

    actions.append(Node(
        package='kudos_yolo_tools', executable='auto_labeler.py', name='auto_labeler', output='screen',
        parameters=[{'output_dir': out, 'targets_file': targets, 'session': session,
                     'robot_model_name': robot, 'use_sim_time': True}]))

    if mode == 'auto':
        actions.append(Node(
            package='kudos_yolo_tools', executable='viewpoint_sampler.py', name='viewpoint_sampler',
            output='screen',
            parameters=[{'num_poses': int(get('num_poses')), 'viewpoints_file': viewpoints,
                         'robot_model_name': robot, 'swap_markers': False, 'jitter_objects': False, 'use_sim_time': True}]))
    elif mode == 'drive':
        actions.append(Node(
            package='kudos_yolo_tools', executable='domain_randomizer.py', name='domain_randomizer',
            output='screen',
            parameters=[{'interval_sec': 2.0, 'swap_markers': False, 'jitter_objects': False,
                         'use_sim_time': True}]))
    else:
        raise RuntimeError('mode 는 auto 또는 drive (받은 값: %s)' % mode)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('output_dir', default_value='~/yolo_data/v2/sim_m2m4'),
        DeclareLaunchArgument('mode', default_value='auto', description='auto | drive'),
        DeclareLaunchArgument('num_poses', default_value='600'),
        DeclareLaunchArgument('branch_sign', default_value='auto', description='auto | A | B'),
        DeclareLaunchArgument('robot', default_value='limo',
                              description='Gazebo 로봇 이름 (competition.launch.py 의 -entity limo)'),
        DeclareLaunchArgument('world_glob', default_value='/tmp/kudos_arena_*/track_2026.world'),
        OpaqueFunction(function=setup),
    ])
