cmake_minimum_required(VERSION 3.8)
project(kudos_perception)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/camera_perception_node.py
  scripts/lane_follow_test.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
