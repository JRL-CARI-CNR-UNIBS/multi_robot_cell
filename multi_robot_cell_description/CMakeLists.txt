cmake_minimum_required(VERSION 3.8)
project(multi_robot_cell_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  urdf
  launch
  controllers
  config
  srdf
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/publish_robot_description.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
