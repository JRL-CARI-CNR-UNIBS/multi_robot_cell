cmake_minimum_required(VERSION 3.8)
project(four_robot_cell)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config scenes DESTINATION share/${PROJECT_NAME})

ament_package()
