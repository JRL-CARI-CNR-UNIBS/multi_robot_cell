cmake_minimum_required(VERSION 3.8)
project(multi_robot_cell_tamp)

# Optimised by default. A plain `colcon build` passes no build type, which compiled this
# package at -O0: the generator's own Eigen math, resampling and per-sample loops ran several
# times slower (sample validation 1.9x on int_cluttered_np, 2026-10-06) for identical output.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default Release)" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(yaml-cpp REQUIRED)

# --- resample: pure C++, no ROS -------------------------------------------- #
# Deliberately free of ROS and MoveIt so the uniform-Δt logic can be unit-tested
# without a robot, a scene, or a running node. It is the piece the whole offset
# identity depends on, so it must be testable in isolation.
add_library(resample SHARED src/resample.cpp)
target_include_directories(resample PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(resample PUBLIC cxx_std_17)

install(TARGETS resample
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

# --- trajectory_generator: the offline motion stage ------------------------- #
add_executable(trajectory_generator src/trajectory_generator.cpp)
target_link_libraries(trajectory_generator resample yaml-cpp)
ament_target_dependencies(trajectory_generator
  rclcpp moveit_core moveit_ros_planning moveit_msgs
  geometry_msgs shape_msgs geometric_shapes ament_index_cpp
  tf2 tf2_geometry_msgs tf2_eigen)
install(TARGETS trajectory_generator DESTINATION lib/${PROJECT_NAME})

# --- collision_generator: inter-robot mu -> forbidden offsets ---------------- #
add_executable(collision_generator src/collision_generator.cpp)
target_link_libraries(collision_generator resample yaml-cpp)
ament_target_dependencies(collision_generator
  rclcpp moveit_core moveit_ros_planning moveit_msgs shape_msgs geometry_msgs)
install(TARGETS collision_generator DESTINATION lib/${PROJECT_NAME})

# --- plan_oracle_check: independent replay of a final plan on real geometry --- #
# Driven by scripts/plan_oracle.py (ADR-0007 addendum 2026-09-21). Reads no mu, no
# spheres, no seam: only the URDF/SRDF, the scene's shapes and a tick-by-tick replay.
add_executable(plan_oracle_check src/plan_oracle.cpp)
ament_target_dependencies(plan_oracle_check
  rclcpp moveit_core geometric_shapes)
install(TARGETS plan_oracle_check DESTINATION lib/${PROJECT_NAME})

# --- schedule_executor: replay a solved schedule on the live controllers ----- #
# Python (rclpy): reads only JSON, so it needs no OR-Tools -- the solve happened
# in the thesis venv. Installed as an executable so `ros2 run`/launch find it.
# scene_visualizer.py, gripper_commander.py and process_commander.py are sibling
# modules the executors import at runtime; all land in the SAME lib/<pkg> dir so the
# executor's `sys.path.insert(dirname(...))` finds them when run as an installed node.
install(PROGRAMS
  scripts/schedule_executor.py
  scripts/cell_registry.py
  scripts/cell_hardware.py
  scripts/trajectory_gripper_commander.py
  scripts/teach_poses.py
  scripts/grip_gate.py
  scripts/tpg_executor.py
  scripts/tpg.py
  scripts/scene_visualizer.py
  scripts/show_scene.py
  scripts/gripper_commander.py
  scripts/process_commander.py
  scripts/vamp_sphere_overlay.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_resample test/test_resample.cpp)
  target_link_libraries(test_resample resample)
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(moveit_core moveit_ros_planning)
ament_package()
