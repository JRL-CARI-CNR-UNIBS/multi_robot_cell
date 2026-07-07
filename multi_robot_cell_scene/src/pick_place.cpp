// YAML-driven multi-object pick-and-place for robot1 of the multi_robot_cell.
//
// The task is described in a YAML file (see config/task.yaml): a list of
// objects (each with a size, a spawn pose, and a grasp pose expressed in the
// object frame) and an ordered task plan (each step names an object and its
// final place pose). For every step we:
//
//   pick_EE  = object.spawn (x) grasp
//   place_EE = step.place   (x) grasp
//
// then run: pre-grasp -> approach -> close -> attach -> retreat
//           -> pre-place -> lower -> open -> detach -> retreat.

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <array>
#include <thread>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <yaml-cpp/yaml.h>

using GripperCommand = control_msgs::action::GripperCommand;

// ---- Data model (parsed from YAML) -----------------------------------------
struct Settings {
  std::string planning_group, ee_link, attach_link, gripper_action, base_frame;
  double gripper_open, gripper_close, approach, vel_scale, acc_scale;
  std::vector<std::string> touch_links;   // gripper links allowed to touch a held object
};
struct ObjectDef {
  std::string id;
  std::array<double, 3> size;
  geometry_msgs::msg::Pose spawn;         // object pose in world
  geometry_msgs::msg::Pose grasp_in_obj;  // tool0 pose in the object frame
};
struct TaskStep {
  std::string object_id;
  geometry_msgs::msg::Pose place;         // object final pose in world
};
struct FixtureDef {                       // static collision geometry (never grasped)
  std::string id;
  std::array<double, 3> size;
  geometry_msgs::msg::Pose pose;          // world pose
};
struct TaskData {
  Settings settings;
  std::vector<FixtureDef> fixtures;
  std::vector<ObjectDef> objects;
  std::vector<TaskStep> plan;
};

// ---- YAML helpers ----------------------------------------------------------
static geometry_msgs::msg::Pose poseFromYaml(const YAML::Node& n)
{
  geometry_msgs::msg::Pose p;
  p.position.x = n["x"].as<double>();
  p.position.y = n["y"].as<double>();
  p.position.z = n["z"].as<double>();
  double roll  = n["roll"]  ? n["roll"].as<double>()  : 0.0;
  double pitch = n["pitch"] ? n["pitch"].as<double>() : 0.0;
  double yaw   = n["yaw"]   ? n["yaw"].as<double>()   : 0.0;
  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  p.orientation = tf2::toMsg(q);
  return p;
}

static TaskData loadTaskData(const std::string& path)
{
  YAML::Node root = YAML::LoadFile(path);
  TaskData d;
  const auto& s = root["settings"];
  d.settings.planning_group = s["planning_group"].as<std::string>();
  d.settings.ee_link        = s["ee_link"].as<std::string>();
  d.settings.attach_link    = s["attach_link"].as<std::string>();
  d.settings.gripper_action = s["gripper_action"].as<std::string>();
  d.settings.base_frame     = s["base_frame"].as<std::string>();
  d.settings.gripper_open   = s["gripper_open"].as<double>();
  d.settings.gripper_close  = s["gripper_close"].as<double>();
  d.settings.approach       = s["approach"].as<double>();
  d.settings.vel_scale      = s["vel_scale"].as<double>();
  d.settings.acc_scale      = s["acc_scale"].as<double>();
  for (const auto& l : s["touch_links"]) d.settings.touch_links.push_back(l.as<std::string>());

  if (root["fixtures"]) {
    for (const auto& n : root["fixtures"]) {
      FixtureDef f;
      f.id = n["id"].as<std::string>();
      f.size = {n["size"][0].as<double>(), n["size"][1].as<double>(), n["size"][2].as<double>()};
      f.pose = poseFromYaml(n["pose"]);
      d.fixtures.push_back(f);
    }
  }
  for (const auto& n : root["objects"]) {
    ObjectDef o;
    o.id = n["id"].as<std::string>();
    o.size = {n["size"][0].as<double>(), n["size"][1].as<double>(), n["size"][2].as<double>()};
    o.spawn = poseFromYaml(n["spawn"]);
    o.grasp_in_obj = poseFromYaml(n["grasp"]);
    d.objects.push_back(o);
  }
  for (const auto& n : root["task_plan"]) {
    TaskStep t;
    t.object_id = n["object"].as<std::string>();
    t.place = poseFromYaml(n["place"]);
    d.plan.push_back(t);
  }
  return d;
}

// ---- Pose composition: world_EE = parent (x) child -------------------------
static tf2::Transform toTf(const geometry_msgs::msg::Pose& p)
{
  tf2::Transform t;
  t.setOrigin(tf2::Vector3(p.position.x, p.position.y, p.position.z));
  t.setRotation(tf2::Quaternion(p.orientation.x, p.orientation.y,
                                p.orientation.z, p.orientation.w));
  return t;
}
static geometry_msgs::msg::Pose fromTf(const tf2::Transform& t)
{
  geometry_msgs::msg::Pose p;
  p.position.x = t.getOrigin().x();
  p.position.y = t.getOrigin().y();
  p.position.z = t.getOrigin().z();
  p.orientation = tf2::toMsg(t.getRotation());
  return p;
}
static geometry_msgs::msg::Pose compose(const geometry_msgs::msg::Pose& parent,
                                        const geometry_msgs::msg::Pose& child)
{
  return fromTf(toTf(parent) * toTf(child));
}

// ---------------------------------------------------------------------------
class PickPlace
{
public:
  PickPlace(const rclcpp::Node::SharedPtr& node, TaskData data)
  : node_(node), logger_(node->get_logger()), data_(std::move(data)),
    mgi_(node, data_.settings.planning_group)
  {
    mgi_.setPoseReferenceFrame(data_.settings.base_frame);
    mgi_.setEndEffectorLink(data_.settings.ee_link);
    mgi_.setMaxVelocityScalingFactor(data_.settings.vel_scale);
    mgi_.setMaxAccelerationScalingFactor(data_.settings.acc_scale);
    mgi_.setPlanningTime(10.0);
    // OMPL is randomized: retry a few times so a single unlucky sample set
    // doesn't fail the whole task.
    mgi_.setNumPlanningAttempts(10);
    gripper_ = rclcpp_action::create_client<GripperCommand>(
        node_, data_.settings.gripper_action);
    for (const auto& o : data_.objects) objects_[o.id] = o;
  }

  bool run()
  {
    resetScene();
    spawnFixtures();
    spawnObjects();

    for (size_t i = 0; i < data_.plan.size(); ++i) {
      const auto& step = data_.plan[i];
      auto it = objects_.find(step.object_id);
      if (it == objects_.end()) {
        RCLCPP_ERROR(logger_, "Task step %zu references unknown object '%s'",
                     i, step.object_id.c_str());
        return false;
      }
      const ObjectDef& obj = it->second;
      RCLCPP_INFO(logger_, "--- Step %zu/%zu: %s ---",
                  i + 1, data_.plan.size(), obj.id.c_str());

      // EE goal = object world pose composed with its grasp (EE-in-object).
      geometry_msgs::msg::Pose pick_ee  = compose(obj.spawn, obj.grasp_in_obj);
      geometry_msgs::msg::Pose place_ee = compose(step.place, obj.grasp_in_obj);

      if (!pickPlaceOne(obj, pick_ee, place_ee)) {
        RCLCPP_ERROR(logger_, "Step %zu (%s) failed, aborting task",
                     i, obj.id.c_str());
        return false;
      }
    }
    RCLCPP_INFO(logger_, "Task plan complete: %zu objects placed", data_.plan.size());
    return true;
  }

private:
  // One pick-and-place cycle for a single object.
  bool pickPlaceOne(const ObjectDef& obj,
                    const geometry_msgs::msg::Pose& pick_ee,
                    const geometry_msgs::msg::Pose& place_ee)
  {
    const double a = data_.settings.approach;
    auto pre_grasp = pick_ee;  pre_grasp.position.z += a;
    auto pre_place = place_ee; pre_place.position.z += a;

    if (!commandGripper(data_.settings.gripper_open)) return false;
    RCLCPP_INFO(logger_, "Pre-grasp");   if (!moveToPose(pre_grasp)) return false;
    RCLCPP_INFO(logger_, "Approach");    if (!cartesianZ(-a))        return false;
    if (!commandGripper(data_.settings.gripper_close)) return false;
    RCLCPP_INFO(logger_, "Attach %s", obj.id.c_str());
    mgi_.attachObject(obj.id, data_.settings.attach_link, data_.settings.touch_links);
    RCLCPP_INFO(logger_, "Retreat");     if (!cartesianZ(a))         return false;
    RCLCPP_INFO(logger_, "Pre-place");   if (!moveToPose(pre_place)) return false;
    RCLCPP_INFO(logger_, "Lower");       if (!cartesianZ(-a))        return false;
    if (!commandGripper(data_.settings.gripper_open)) return false;
    RCLCPP_INFO(logger_, "Detach %s", obj.id.c_str());
    mgi_.detachObject(obj.id);
    RCLCPP_INFO(logger_, "Retreat");     if (!cartesianZ(a))         return false;
    return true;
  }

  // Idempotent startup: clear every managed object (detach if a previous run
  // crashed with one held, then remove any world instance).
  void resetScene()
  {
    std::vector<std::string> ids;
    for (const auto& o : data_.objects) {
      mgi_.detachObject(o.id);   // only graspable objects can be attached
      ids.push_back(o.id);
    }
    for (const auto& f : data_.fixtures) ids.push_back(f.id);
    psi_.removeCollisionObjects(ids);
    rclcpp::sleep_for(std::chrono::milliseconds(500));
    RCLCPP_INFO(logger_, "Scene reset (%zu items cleared)", ids.size());
  }

  void spawnFixtures()
  {
    std::vector<moveit_msgs::msg::CollisionObject> objs;
    for (const auto& f : data_.fixtures) {
      moveit_msgs::msg::CollisionObject c;
      c.header.frame_id = data_.settings.base_frame;
      c.id = f.id;
      shape_msgs::msg::SolidPrimitive prim;
      prim.type = prim.BOX;
      prim.dimensions = {f.size[0], f.size[1], f.size[2]};
      c.primitives.push_back(prim);
      c.primitive_poses.push_back(f.pose);
      c.operation = c.ADD;
      objs.push_back(c);
    }
    if (!objs.empty()) {
      psi_.applyCollisionObjects(objs);
      RCLCPP_INFO(logger_, "Spawned %zu fixtures", objs.size());
    }
  }

  void spawnObjects()
  {
    std::vector<moveit_msgs::msg::CollisionObject> objs;
    for (const auto& o : data_.objects) {
      moveit_msgs::msg::CollisionObject c;
      c.header.frame_id = data_.settings.base_frame;
      c.id = o.id;
      shape_msgs::msg::SolidPrimitive prim;
      prim.type = prim.BOX;
      prim.dimensions = {o.size[0], o.size[1], o.size[2]};
      c.primitives.push_back(prim);
      c.primitive_poses.push_back(o.spawn);
      c.operation = c.ADD;
      objs.push_back(c);
    }
    psi_.applyCollisionObjects(objs);
    RCLCPP_INFO(logger_, "Spawned %zu objects", objs.size());
  }

  bool moveToPose(const geometry_msgs::msg::Pose& target)
  {
    mgi_.setPoseTarget(target);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (mgi_.plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(logger_, "Planning failed");
      return false;
    }
    return mgi_.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;
  }

  // Straight-line move of the end effector by dz along world Z.
  bool cartesianZ(double dz)
  {
    geometry_msgs::msg::Pose start = mgi_.getCurrentPose(data_.settings.ee_link).pose;
    geometry_msgs::msg::Pose end = start;
    end.position.z += dz;
    std::vector<geometry_msgs::msg::Pose> waypoints = {end};
    moveit_msgs::msg::RobotTrajectory traj;
    double fraction = mgi_.computeCartesianPath(waypoints, 0.005, traj);
    RCLCPP_INFO(logger_, "Cartesian path %.0f%% achieved", fraction * 100.0);
    if (fraction < 0.9) {
      // The straight line couldn't be fully solved (near-singular / IK jump).
      // Fall back to a free-space plan to the same end pose: we lose the
      // guaranteed straight line but the motion still completes.
      RCLCPP_WARN(logger_, "Cartesian only %.0f%%, falling back to joint-space plan",
                  fraction * 100.0);
      return moveToPose(end);
    }
    return mgi_.execute(traj) == moveit::core::MoveItErrorCode::SUCCESS;
  }

  bool commandGripper(double position)
  {
    if (!gripper_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(logger_, "Gripper action server unavailable");
      return false;
    }
    GripperCommand::Goal goal;
    goal.command.position = position;
    goal.command.max_effort = 100.0;
    auto goal_future = gripper_->async_send_goal(goal);
    if (goal_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      RCLCPP_ERROR(logger_, "Timed out sending gripper goal");
      return false;
    }
    auto handle = goal_future.get();
    if (!handle) {
      RCLCPP_ERROR(logger_, "Gripper goal rejected");
      return false;
    }
    auto result_future = gripper_->async_get_result(handle);
    if (result_future.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
      RCLCPP_ERROR(logger_, "Timed out waiting for gripper result");
      return false;
    }
    RCLCPP_INFO(logger_, "Gripper commanded to %.2f", position);
    return true;
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_;
  TaskData data_;
  std::map<std::string, ObjectDef> objects_;
  moveit::planning_interface::MoveGroupInterface mgi_;
  moveit::planning_interface::PlanningSceneInterface psi_;
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
    "pick_place",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  std::string task_file;
  node->get_parameter_or<std::string>("task_file", task_file, "");
  if (task_file.empty()) {
    RCLCPP_FATAL(node->get_logger(), "Parameter 'task_file' is not set");
    return 1;
  }

  TaskData data;
  try {
    data = loadTaskData(task_file);
    RCLCPP_INFO(node->get_logger(), "Loaded task '%s': %zu objects, %zu steps",
                task_file.c_str(), data.objects.size(), data.plan.size());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(node->get_logger(), "Failed to load task file: %s", e.what());
    return 1;
  }

  std::thread spinner([node]() { rclcpp::spin(node); });
  PickPlace demo(node, std::move(data));
  demo.run();

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
