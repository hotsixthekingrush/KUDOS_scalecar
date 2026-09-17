cmake_minimum_required(VERSION 3.8)
project(webot_arena)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    launch
    worlds
    meshes
    media
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/make_arena.py
    scripts/mission_motion_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

