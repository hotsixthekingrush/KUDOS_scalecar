cmake_minimum_required(VERSION 3.8)
project(kudos_yolo_tools)

find_package(ament_cmake REQUIRED)

# ROS2 노드 (ros2 run 으로 실행)
#   auto_labeler.py      : Gazebo 자동 라벨링
#   domain_randomizer.py : 물체 배치 무작위화
#   yolo_detector_node.py: 실시간 추론 -> /perception/mission_objects
#   yolo_geometry.py     : 공용 기하 모듈 (위 노드들이 import)
# 오프라인 스크립트 (python3 로 직접 실행해도 됨)
#   image_augment.py, train_yolo.py, split_dataset.py
install(PROGRAMS
  scripts/yolo_geometry.py
  scripts/yolo_detector_node.py
  scripts/split_dataset.py
  scripts/auto_labeler.py
  scripts/domain_randomizer.py
  scripts/image_augment.py
  scripts/train_yolo.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
