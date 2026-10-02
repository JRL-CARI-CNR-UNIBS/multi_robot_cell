cmake_minimum_required(VERSION 3.8)
project(multi_robot_cell_scene)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(spawn_object src/spawn_object.cpp)
ament_target_dependencies(spawn_object
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  geometry_msgs
  shape_msgs
)

add_executable(pick_place src/pick_place.cpp)
ament_target_dependencies(pick_place
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  moveit_msgs
  geometry_msgs
  shape_msgs
  control_msgs
  tf2
  tf2_geometry_msgs
)
target_link_libraries(pick_place yaml-cpp)

install(TARGETS spawn_object pick_place
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
