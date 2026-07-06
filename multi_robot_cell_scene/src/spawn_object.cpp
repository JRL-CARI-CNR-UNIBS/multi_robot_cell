#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("spawn_object");
  auto logger = node->get_logger();

  // Talks to move_group's planning scene via the /apply_planning_scene service.
  moveit::planning_interface::PlanningSceneInterface psi;

  // --- Build one box collision object -------------------------------------
  moveit_msgs::msg::CollisionObject box;
  box.header.frame_id = "world";   // pose expressed in the world frame
  box.id = "demo_box";             // unique name; reuse it to move/remove later

  // Geometry: a 5 cm cube.
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = primitive.BOX;
  primitive.dimensions = {0.05, 0.05, 0.05};  // x, y, z (metres)

  // Pose: sit it on the table top. Table surface is at world z = 0.75,
  // so the box centre is 0.75 + half the box height.
  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;        // no rotation
  pose.position.x = 0.0;
  pose.position.y = 0.0;
  pose.position.z = 0.75 + 0.025;  // table top + half box height

  box.primitives.push_back(primitive);
  box.primitive_poses.push_back(pose);
  box.operation = box.ADD;         // ADD = create (or overwrite if id exists)

  // --- Apply it to the scene ----------------------------------------------
  psi.applyCollisionObjects({box});
  RCLCPP_INFO(logger, "Spawned collision object '%s' on the table.", box.id.c_str());

  rclcpp::shutdown();
  return 0;
}
