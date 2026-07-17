// Offline trajectory generation for the multi-robot TAMP pipeline.
//
// For every (robot r, task i) pair -- BOTH robots get a trajectory for EVERY
// task, because choosing the robot is the scheduler's job -- this plans the full
//
//     home -> pre-grasp -> grasp -> [close] -> retreat
//          -> pre-place -> place -> [open]  -> retreat -> home
//
// cycle, concatenates the segments onto one timeline, resamples them to a uniform
// Δt shared by both robots, re-validates every resampled sample, and writes an
// artifact the Python scheduler consumes.
//
// WHY THIS IS NOT pick_place.cpp
//
// `multi_robot_cell_scene/src/pick_place.cpp` EXECUTES each segment and then reads
// `getCurrentPose()` to start the next. Offline there is no robot to move, so the
// start state of each segment is chained VIRTUALLY from the last waypoint of the
// previous one. Nothing here executes, nothing here touches move_group: the node
// owns its own PlanningScene and planning pipeline, which also makes it
// deterministic and free of scene-mutation races.
//
// HOME CLEARANCE IS BY CONSTRUCTION
//
// The scene holds BOTH robots. While planning robot r, robot s is pinned at its
// home configuration -- physically present, and collision-checked against. So
// every trajectory r produces is clear of a parked s by construction, which closes
// the moving-vs-parked case of ADR-0004 without a separate check. Moving-vs-moving
// is what the collision matrices are for. If s-at-home genuinely blocks r's only
// route, planning FAILS -- and that failure is the correct, honest signal.
//
// THE ENVIRONMENT (ADR-0003)
//
// Task i is planned against the precedence-pruned union environment: objects that
// must run before i are at their PLACE pose, objects that must run after are at
// their SPAWN pose, and objects unordered w.r.t. i appear at BOTH. Object i itself
// is excluded and instead attached to the gripper for the carry -- but it still
// exists at spawn during the approach and at place during the return, which is
// what `ObjectState` tracks.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/planning_pipeline/planning_pipeline.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/cartesian_interpolator.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>

#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <yaml-cpp/yaml.h>

#include "multi_robot_cell_tamp/resample.hpp"

namespace mrct = multi_robot_cell_tamp;
using mrct::ObjectState;
using mrct::Phase;
using mrct::Segment;
using mrct::TimedWaypoint;

namespace
{

// --------------------------------------------------------------------------- #
// Task specification (parsed from tamp_task.yaml)
// --------------------------------------------------------------------------- #
struct Discretisation
{
  double delta_t{0.1};
  int gripper_dwell_slots{10};
  double max_joint_step_warn{0.15};
  double max_cartesian_step_warn{0.02};
};

struct PlanningCfg
{
  std::string planner_id;
  double planning_time{20.0};
  int planning_attempts{10};
  double vel_scale{0.5};
  double acc_scale{0.5};
  double approach{0.12};
  double cartesian_step{0.005};
  int seed{7};
};

struct RobotCfg
{
  std::string name;
  std::string planning_group;   // 7 DOF: rail + arm. The scheduling group.
  std::string arm_group;        // 6 DOF: arm only. See planCartesianZ.
  std::string ee_link;
  std::string attach_link;
  std::vector<std::string> touch_links;
  std::map<std::string, double> home;   // joint -> value (7 DOF, fully determined)
};

struct ObjectDef
{
  std::string id;
  std::array<double, 3> size{};
  geometry_msgs::msg::Pose spawn;
  geometry_msgs::msg::Pose grasp_in_obj;
};

struct FixtureDef
{
  std::string id;
  std::array<double, 3> size{};
  geometry_msgs::msg::Pose pose;
};

struct TaskDef
{
  std::string id;
  std::string object_id;
  geometry_msgs::msg::Pose place;
};

struct TaskSpec
{
  Discretisation disc;
  PlanningCfg planning;
  std::string base_frame{"world"};
  std::vector<RobotCfg> robots;
  std::vector<FixtureDef> fixtures;
  std::vector<ObjectDef> objects;
  std::vector<TaskDef> tasks;
  std::vector<std::pair<std::string, std::string>> precedences;

  const ObjectDef & object(const std::string & id) const
  {
    for (const auto & o : objects) {
      if (o.id == id) {return o;}
    }
    throw std::runtime_error("unknown object: " + id);
  }
};

geometry_msgs::msg::Pose poseFromYaml(const YAML::Node & n)
{
  geometry_msgs::msg::Pose p;
  p.position.x = n["x"].as<double>();
  p.position.y = n["y"].as<double>();
  p.position.z = n["z"].as<double>();
  tf2::Quaternion q;
  q.setRPY(
    n["roll"] ? n["roll"].as<double>() : 0.0,
    n["pitch"] ? n["pitch"].as<double>() : 0.0,
    n["yaw"] ? n["yaw"].as<double>() : 0.0);
  p.orientation = tf2::toMsg(q);
  return p;
}

TaskSpec loadTaskSpec(const std::string & path)
{
  YAML::Node root = YAML::LoadFile(path);
  TaskSpec s;

  const auto & d = root["discretisation"];
  s.disc.delta_t = d["delta_t"].as<double>();
  s.disc.gripper_dwell_slots = d["gripper_dwell_slots"].as<int>();
  s.disc.max_joint_step_warn = d["max_joint_step_warn"].as<double>();
  s.disc.max_cartesian_step_warn = d["max_cartesian_step_warn"].as<double>();

  const auto & p = root["planning"];
  s.planning.planner_id = p["planner_id"].as<std::string>();
  s.planning.planning_time = p["planning_time"].as<double>();
  s.planning.planning_attempts = p["planning_attempts"].as<int>();
  s.planning.vel_scale = p["vel_scale"].as<double>();
  s.planning.acc_scale = p["acc_scale"].as<double>();
  s.planning.approach = p["approach"].as<double>();
  s.planning.cartesian_step = p["cartesian_step"].as<double>();
  s.planning.seed = p["seed"].as<int>();

  s.base_frame = root["base_frame"].as<std::string>();

  for (const auto & kv : root["robots"]) {
    RobotCfg r;
    r.name = kv.first.as<std::string>();
    const auto & n = kv.second;
    r.planning_group = n["planning_group"].as<std::string>();
    r.arm_group = n["arm_group"].as<std::string>();
    r.ee_link = n["ee_link"].as<std::string>();
    r.attach_link = n["attach_link"].as<std::string>();
    for (const auto & l : n["touch_links"]) {r.touch_links.push_back(l.as<std::string>());}
    for (const auto & j : n["home"]) {
      r.home[j.first.as<std::string>()] = j.second.as<double>();
    }
    s.robots.push_back(r);
  }
  std::sort(
    s.robots.begin(), s.robots.end(),
    [](const RobotCfg & a, const RobotCfg & b) {return a.name < b.name;});

  if (root["fixtures"]) {
    for (const auto & n : root["fixtures"]) {
      FixtureDef f;
      f.id = n["id"].as<std::string>();
      f.size = {n["size"][0].as<double>(), n["size"][1].as<double>(), n["size"][2].as<double>()};
      f.pose = poseFromYaml(n["pose"]);
      s.fixtures.push_back(f);
    }
  }
  for (const auto & n : root["objects"]) {
    ObjectDef o;
    o.id = n["id"].as<std::string>();
    o.size = {n["size"][0].as<double>(), n["size"][1].as<double>(), n["size"][2].as<double>()};
    o.spawn = poseFromYaml(n["spawn"]);
    o.grasp_in_obj = poseFromYaml(n["grasp"]);
    s.objects.push_back(o);
  }
  for (const auto & n : root["tasks"]) {
    TaskDef t;
    t.id = n["id"].as<std::string>();
    t.object_id = n["object"].as<std::string>();
    t.place = poseFromYaml(n["place"]);
    s.tasks.push_back(t);
  }
  if (root["precedences"]) {
    for (const auto & n : root["precedences"]) {
      s.precedences.emplace_back(n[0].as<std::string>(), n[1].as<std::string>());
    }
  }
  return s;
}

// --------------------------------------------------------------------------- #
// Pose helpers
// --------------------------------------------------------------------------- #
tf2::Transform toTf(const geometry_msgs::msg::Pose & p)
{
  tf2::Transform t;
  t.setOrigin(tf2::Vector3(p.position.x, p.position.y, p.position.z));
  t.setRotation(
    tf2::Quaternion(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w));
  return t;
}

geometry_msgs::msg::Pose fromTf(const tf2::Transform & t)
{
  geometry_msgs::msg::Pose p;
  p.position.x = t.getOrigin().x();
  p.position.y = t.getOrigin().y();
  p.position.z = t.getOrigin().z();
  p.orientation = tf2::toMsg(t.getRotation());
  return p;
}

/// world_EE = object_pose (x) grasp_in_object. One grasp definition serves the
/// pick (composed with the spawn pose) and the place (with the place pose).
geometry_msgs::msg::Pose compose(
  const geometry_msgs::msg::Pose & parent, const geometry_msgs::msg::Pose & child)
{
  return fromTf(toTf(parent) * toTf(child));
}

geometry_msgs::msg::Pose raised(const geometry_msgs::msg::Pose & p, double dz)
{
  auto q = p;
  q.position.z += dz;
  return q;
}

moveit_msgs::msg::CollisionObject makeBox(
  const std::string & id, const std::array<double, 3> & size,
  const geometry_msgs::msg::Pose & pose, const std::string & frame)
{
  moveit_msgs::msg::CollisionObject c;
  c.header.frame_id = frame;
  c.id = id;
  shape_msgs::msg::SolidPrimitive prim;
  prim.type = prim.BOX;
  prim.dimensions = {size[0], size[1], size[2]};
  c.primitives.push_back(prim);
  c.primitive_poses.push_back(pose);
  c.operation = c.ADD;
  return c;
}

// --------------------------------------------------------------------------- #
// Precedence closure (ADR-0003)
// --------------------------------------------------------------------------- #
/// Transitive closure of the precedence relation: `before[j]` = every task that
/// must complete before j. Used to prune the offline environment -- an object
/// whose task provably ran already is at its PLACE pose, not its spawn.
std::map<std::string, std::set<std::string>> precedenceClosure(const TaskSpec & spec)
{
  std::map<std::string, std::set<std::string>> before;
  for (const auto & t : spec.tasks) {before[t.id] = {};}
  for (const auto & [i, j] : spec.precedences) {before[j].insert(i);}

  // Floyd-Warshall-ish fixpoint; the task set is tiny, so clarity beats cleverness.
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto & [j, preds] : before) {
      std::set<std::string> grown = preds;
      for (const auto & p : preds) {
        grown.insert(before[p].begin(), before[p].end());
      }
      if (grown.size() != preds.size()) {
        preds = grown;
        changed = true;
      }
    }
  }
  return before;
}

// --------------------------------------------------------------------------- #
// The generator
// --------------------------------------------------------------------------- #
class TrajectoryGenerator
{
public:
  TrajectoryGenerator(const rclcpp::Node::SharedPtr & node, TaskSpec spec)
  : node_(node), log_(node->get_logger()), spec_(std::move(spec))
  {
    robot_model_loader::RobotModelLoader loader(node_, "robot_description");
    model_ = loader.getModel();
    if (!model_) {
      throw std::runtime_error("could not load the robot model from `robot_description`");
    }
    RCLCPP_INFO(log_, "robot model '%s' loaded", model_->getName().c_str());

    pipeline_ = std::make_shared<planning_pipeline::PlanningPipeline>(model_, node_, "ompl");
    before_ = precedenceClosure(spec_);
  }

  bool run(const std::string & out_path)
  {
    std::vector<std::string> artifact;   // one JSON object per (robot, task)
    bool all_ok = true;
    double worst_step = 0.0;

    for (const auto & robot : spec_.robots) {
      for (const auto & task : spec_.tasks) {
        RCLCPP_INFO(log_, "=== planning %s / %s ===", robot.name.c_str(), task.id.c_str());
        mrct::ResampledTrajectory traj;
        if (!planTask(robot, task, traj)) {
          RCLCPP_ERROR(
            log_, "FAILED to plan %s / %s -- every (robot, task) pair needs a trajectory, "
            "so the artifact is incomplete", robot.name.c_str(), task.id.c_str());
          all_ok = false;
          continue;
        }

        double cart_step = 0.0;
        const int bad = validateSamples(robot, task, traj, cart_step);
        if (bad > 0) {
          RCLCPP_ERROR(
            log_, "%s / %s: %d resampled sample(s) are IN COLLISION -- the interpolant left "
            "the validated path. Reduce delta_t or densify the plan.",
            robot.name.c_str(), task.id.c_str(), bad);
          all_ok = false;
        }

        worst_step = std::max(worst_step, cart_step);
        RCLCPP_INFO(
          log_, "%s / %s: K=%zu slots (%.2f s), max joint step %.4f, "
          "max link travel per slot %.4f m",
          robot.name.c_str(), task.id.c_str(), traj.num_samples,
          traj.num_samples * spec_.disc.delta_t, traj.max_joint_step, cart_step);

        artifact.push_back(toJson(robot, task, traj));
      }
    }

    // The soundness statement, in the units that matter. Compare this against the
    // THINNEST feature in the scene (here: the 2 cm tray walls). A robot link that
    // travels further than that between two consecutive samples could pass clean
    // through it without any sample ever registering a collision -- and the
    // schedule would be "provably" collision-free while being nothing of the sort.
    RCLCPP_INFO(
      log_,
      "discretisation: at delta_t=%.3f s the fastest robot link travels at most %.4f m per slot. "
      "Any obstacle thinner than that could be stepped over.",
      spec_.disc.delta_t, worst_step);
    if (worst_step > spec_.disc.max_cartesian_step_warn) {
      RCLCPP_WARN(
        log_,
        "  ^ that exceeds the %.4f m threshold. Lower delta_t, or the collision matrices may "
        "MISS collisions and the schedule will be unsound.",
        spec_.disc.max_cartesian_step_warn);
    }

    writeArtifact(out_path, artifact);
    RCLCPP_INFO(
      log_, "%s: %zu/%zu trajectories written to %s",
      all_ok ? "OK" : "INCOMPLETE", artifact.size(),
      spec_.robots.size() * spec_.tasks.size(), out_path.c_str());
    return all_ok;
  }

private:
  // ---- scene construction ------------------------------------------------- #

  /// The environment task `i` is planned against (ADR-0003), with the OTHER robot
  /// parked at its home so that home clearance holds by construction.
  planning_scene::PlanningScenePtr sceneFor(const RobotCfg & robot, const TaskDef & task)
  {
    auto scene = std::make_shared<planning_scene::PlanningScene>(model_);

    // Every robot starts at its home; the planning group only moves `robot`'s
    // joints, so the other robot stays parked exactly where we put it.
    moveit::core::RobotState & state = scene->getCurrentStateNonConst();
    state.setToDefaultValues();
    for (const auto & r : spec_.robots) {
      for (const auto & [joint, value] : r.home) {
        state.setJointPositions(joint, &value);
      }
    }
    state.update();

    std::vector<moveit_msgs::msg::CollisionObject> world;
    for (const auto & f : spec_.fixtures) {
      world.push_back(makeBox(f.id, f.size, f.pose, spec_.base_frame));
    }

    // Precedence-pruned union. Object `i` itself is excluded: it is handled by the
    // phase machinery (spawn -> attached -> place), not by the static world.
    for (const auto & other : spec_.tasks) {
      if (other.id == task.id) {continue;}
      const ObjectDef & obj = spec_.object(other.object_id);
      const bool other_first = before_.at(task.id).count(other.id) > 0;
      const bool task_first = before_.at(other.id).count(task.id) > 0;

      if (other_first) {
        world.push_back(makeBox(obj.id, obj.size, other.place, spec_.base_frame));
      } else if (task_first) {
        world.push_back(makeBox(obj.id, obj.size, obj.spawn, spec_.base_frame));
      } else {
        // Unordered: it could be at either pose when task i runs, so BOTH are
        // obstacles. Conservative, and order-independent -- which is what keeps the
        // offline stage decoupled from the schedule.
        world.push_back(makeBox(obj.id + "__spawn", obj.size, obj.spawn, spec_.base_frame));
        world.push_back(makeBox(obj.id + "__place", obj.size, other.place, spec_.base_frame));
      }
    }
    for (auto & c : world) {scene->processCollisionObjectMsg(c);}
    return scene;
  }

  /// Put object `i` into the scene in the state a given phase implies.
  void setObjectState(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const TaskDef & task, ObjectState st)
  {
    const ObjectDef & obj = spec_.object(task.object_id);

    // Clear whatever representation is currently there.
    moveit_msgs::msg::AttachedCollisionObject detach;
    detach.object.id = obj.id;
    detach.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    detach.link_name = robot.attach_link;
    scene->processAttachedCollisionObjectMsg(detach);

    moveit_msgs::msg::CollisionObject remove;
    remove.id = obj.id;
    remove.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    scene->processCollisionObjectMsg(remove);

    if (st == ObjectState::Attached) {
      moveit_msgs::msg::AttachedCollisionObject aco;
      aco.link_name = robot.attach_link;
      aco.touch_links = robot.touch_links;
      aco.object = makeBox(obj.id, obj.size, obj.spawn, spec_.base_frame);
      // The pose is ignored once attached (it rides the link), but the frame must
      // be one MoveIt can resolve; the grasp transform is what fixes it to the
      // gripper, and that is implied by the state at which we attach.
      scene->processAttachedCollisionObjectMsg(aco);
    } else {
      const auto & pose = (st == ObjectState::AtSpawn) ? obj.spawn : task.place;
      auto add = makeBox(obj.id, obj.size, pose, spec_.base_frame);
      scene->processCollisionObjectMsg(add);
    }
  }

  // ---- planning ----------------------------------------------------------- #

  /// IK to a world EE pose, rejecting solutions that collide with the scene.
  bool ikTo(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const geometry_msgs::msg::Pose & target, const moveit::core::RobotState & seed,
    moveit::core::RobotState & out)
  {
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);
    out = seed;

    // Reject IK solutions that are in collision: an IK solver will happily hand
    // back a configuration buried in the table.
    auto valid = [&](moveit::core::RobotState * s, const moveit::core::JointModelGroup * g,
        const double * values) {
        s->setJointGroupPositions(g, values);
        s->update();
        return !scene->isStateColliding(*s, g->getName());
      };

    Eigen::Isometry3d goal;
    tf2::fromMsg(target, goal);
    return out.setFromIK(jmg, goal, robot.ee_link, 0.5, valid);
  }

  /// Free-space plan between two joint configurations (OMPL).
  bool planJoint(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const moveit::core::RobotState & start, const moveit::core::RobotState & goal,
    Phase phase, Segment & out)
  {
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);

    planning_interface::MotionPlanRequest req;
    req.group_name = robot.planning_group;
    req.planner_id = spec_.planning.planner_id;
    req.allowed_planning_time = spec_.planning.planning_time;
    req.num_planning_attempts = spec_.planning.planning_attempts;
    req.max_velocity_scaling_factor = spec_.planning.vel_scale;
    req.max_acceleration_scaling_factor = spec_.planning.acc_scale;
    moveit::core::robotStateToRobotStateMsg(start, req.start_state);
    req.goal_constraints.push_back(kinematic_constraints::constructGoalConstraints(goal, jmg));

    planning_interface::MotionPlanResponse res;
    if (!pipeline_->generatePlan(scene, req, res) ||
      res.error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
    {
      RCLCPP_WARN(log_, "OMPL failed (error %d)", res.error_code.val);
      return false;
    }
    return toSegment(*res.trajectory, phase, out);
  }

  /// Straight-line EE motion along world Z (the approach / retreat).
  ///
  /// Interpolated on the **6-DOF arm group**, not the 7-DOF rail+arm group, with
  /// the rail pinned wherever the pre-grasp IK left it.
  ///
  /// Why: the 7-DOF group is kinematically redundant, and asking KDL to solve the
  /// Cartesian IK for it stalls a few centimetres in -- reproducibly, at the same
  /// step, for every robot and every task (an IK failure, not an obstacle; the
  /// diagnostic below distinguishes the two). The rail has no business moving
  /// during a 12 cm vertical approach anyway: holding it still makes the descent
  /// well-conditioned, deterministic, and physically what you want.
  ///
  /// The waypoints are still emitted in the 7-DOF ordering the scheduler expects --
  /// the rail column is simply constant across the segment.
  bool planCartesianZ(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const moveit::core::RobotState & start, double dz, Phase phase, Segment & out,
    moveit::core::RobotState & end_state)
  {
    const auto * arm = model_->getJointModelGroup(robot.arm_group);
    const auto * full = model_->getJointModelGroup(robot.planning_group);
    const auto * link = model_->getLinkModel(robot.ee_link);

    auto state = std::make_shared<moveit::core::RobotState>(start);
    std::vector<std::shared_ptr<moveit::core::RobotState>> path;

    auto valid = [&](moveit::core::RobotState * s, const moveit::core::JointModelGroup * g,
        const double * values) {
        s->setJointGroupPositions(g, values);
        s->update();
        // Check the WHOLE robot, not just the arm: the group being interpolated is
        // the arm, but a collision anywhere (rail carriage, the other robot, the
        // table) still invalidates the state.
        return !scene->isStateColliding(*s, robot.planning_group);
      };

    // NOTE the return value. The Eigen::Vector3d (translation) overload returns the
    // DISTANCE ACHIEVED IN METRES -- only the Isometry3d (pose-target) overload
    // returns a 0..1 fraction. Comparing this against a fraction silently rejects
    // every successful path (a fully-achieved 12 cm descent "fails" a `< 0.99`
    // test), which looks exactly like a planner problem and is not one.
    const double achieved_m = moveit::core::CartesianInterpolator::computeCartesianPath(
      state.get(), arm, path, link, Eigen::Vector3d(0, 0, dz), /*global_reference_frame=*/true,
      moveit::core::MaxEEFStep(spec_.planning.cartesian_step),
      moveit::core::JumpThreshold::disabled(), valid);

    const double wanted_m = std::abs(dz);
    if (achieved_m < 0.99 * wanted_m || path.size() < 2) {
      RCLCPP_WARN(
        log_, "Cartesian path achieved %.4f m of %.4f m (%.0f%%)", achieved_m, wanted_m,
        100.0 * achieved_m / wanted_m);
      diagnoseCartesianFailure(scene, robot, start, dz);
      return false;
    }

    // CartesianInterpolator returns a geometric path with no timing. Time it on the
    // 7-DOF group, so the segment carries the same joint set as every other segment.
    robot_trajectory::RobotTrajectory rt(model_, full);
    for (const auto & s : path) {rt.addSuffixWayPoint(*s, 0.0);}
    if (!timeParameterise(rt)) {return false;}

    end_state = *path.back();
    return toSegment(rt, phase, out);
  }

  /// Say WHY a straight-line approach stalled, instead of just how far it got.
  ///
  /// Two very different failures look identical from the achieved fraction: IK ran
  /// out of reach/dexterity, or a perfectly reachable pose was rejected as
  /// colliding. Re-running the interpolation with collision checking OFF separates
  /// them, and if it is a collision, the contacting link pair is what you actually
  /// need to see.
  void diagnoseCartesianFailure(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const moveit::core::RobotState & start, double dz)
  {
    const auto * arm = model_->getJointModelGroup(robot.arm_group);
    const auto * link = model_->getLinkModel(robot.ee_link);

    auto state = std::make_shared<moveit::core::RobotState>(start);
    std::vector<std::shared_ptr<moveit::core::RobotState>> path;
    const double geometric_m = moveit::core::CartesianInterpolator::computeCartesianPath(
      state.get(), arm, path, link, Eigen::Vector3d(0, 0, dz), true,
      moveit::core::MaxEEFStep(spec_.planning.cartesian_step),
      moveit::core::JumpThreshold::disabled());   // no validity callback: IK only

    const double wanted_m = std::abs(dz);
    if (geometric_m < 0.99 * wanted_m) {
      RCLCPP_ERROR(
        log_, "  ... and only %.4f m of %.4f m is reachable even IGNORING collisions -- this is "
        "an IK / reach / singularity failure, not an obstacle.", geometric_m, wanted_m);
      return;
    }

    RCLCPP_ERROR(
      log_, "  ... but the full %.4f m is reachable when collisions are ignored -- so an OBSTACLE "
      "is blocking it. Contacts along the path:", wanted_m);

    collision_detection::CollisionRequest req;
    req.contacts = true;
    req.max_contacts = 8;
    req.max_contacts_per_pair = 1;

    for (std::size_t k = 0; k < path.size(); ++k) {
      collision_detection::CollisionResult res;
      scene->checkCollision(req, res, *path[k]);
      if (!res.collision) {continue;}
      for (const auto & [pair, contacts] : res.contacts) {
        RCLCPP_ERROR(
          log_, "    at %.0f%% of the descent: '%s' touches '%s'",
          100.0 * static_cast<double>(k) / static_cast<double>(path.size() - 1),
          pair.first.c_str(), pair.second.c_str());
      }
      return;   // the first blocked step is the informative one
    }
  }

  bool timeParameterise(robot_trajectory::RobotTrajectory & rt)
  {
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    if (!totg.computeTimeStamps(rt, spec_.planning.vel_scale, spec_.planning.acc_scale)) {
      RCLCPP_WARN(log_, "time parameterisation failed");
      return false;
    }
    return true;
  }

  /// RobotTrajectory -> the resampler's Segment. Positions AND velocities: the
  /// velocities are what let the resampler use cubic Hermite, which reproduces the
  /// profile TOTG computed instead of cutting its corners.
  bool toSegment(const robot_trajectory::RobotTrajectory & rt, Phase phase, Segment & out)
  {
    if (rt.getWayPointCount() < 2) {
      RCLCPP_WARN(log_, "trajectory has < 2 waypoints");
      return false;
    }
    const auto * jmg = rt.getGroup();

    out.phase = phase;
    out.waypoints.clear();
    for (std::size_t k = 0; k < rt.getWayPointCount(); ++k) {
      const auto & s = rt.getWayPoint(k);
      TimedWaypoint wp;
      wp.time_from_start = rt.getWayPointDurationFromStart(k);
      s.copyJointGroupPositions(jmg, wp.positions);
      s.copyJointGroupVelocities(jmg, wp.velocities);
      out.waypoints.push_back(std::move(wp));
    }
    return true;
  }

  /// The whole home -> pick -> transport -> place -> home cycle for one pair.
  bool planTask(const RobotCfg & robot, const TaskDef & task, mrct::ResampledTrajectory & out)
  {
    auto scene = sceneFor(robot, task);
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);
    const ObjectDef & obj = spec_.object(task.object_id);

    const auto pick_ee = compose(obj.spawn, obj.grasp_in_obj);
    const auto place_ee = compose(task.place, obj.grasp_in_obj);
    const double a = spec_.planning.approach;

    moveit::core::RobotState home(scene->getCurrentState());   // both robots parked
    std::vector<Segment> segs;

    // --- approach: object i sits at its SPAWN pose --------------------------- #
    setObjectState(scene, robot, task, ObjectState::AtSpawn);

    moveit::core::RobotState pre_grasp(home);
    if (!ikTo(scene, robot, raised(pick_ee, a), home, pre_grasp)) {
      RCLCPP_WARN(log_, "no collision-free IK for the pre-grasp pose");
      return false;
    }

    Segment s1;
    if (!planJoint(scene, robot, home, pre_grasp, Phase::ToPick, s1)) {return false;}
    segs.push_back(s1);

    Segment s2;
    moveit::core::RobotState grasp(pre_grasp);
    if (!planCartesianZ(scene, robot, pre_grasp, -a, Phase::ToPick, s2, grasp)) {return false;}
    segs.push_back(s2);

    // --- close the gripper: arm frozen, slots consumed ----------------------- #
    std::vector<double> grasp_q;
    grasp.copyJointGroupPositions(jmg, grasp_q);
    segs.push_back(
      mrct::makeDwell(
        grasp_q, Phase::GripClose, spec_.disc.gripper_dwell_slots, spec_.disc.delta_t));

    // --- carry: the object rides the gripper --------------------------------- #
    scene->getCurrentStateNonConst() = grasp;
    setObjectState(scene, robot, task, ObjectState::Attached);

    Segment s3;
    moveit::core::RobotState lifted(grasp);
    if (!planCartesianZ(scene, robot, grasp, a, Phase::Carrying, s3, lifted)) {return false;}
    segs.push_back(s3);

    moveit::core::RobotState pre_place(lifted);
    if (!ikTo(scene, robot, raised(place_ee, a), lifted, pre_place)) {
      RCLCPP_WARN(log_, "no collision-free IK for the pre-place pose (object attached)");
      return false;
    }

    Segment s4;
    if (!planJoint(scene, robot, lifted, pre_place, Phase::Carrying, s4)) {return false;}
    segs.push_back(s4);

    Segment s5;
    moveit::core::RobotState placed(pre_place);
    if (!planCartesianZ(scene, robot, pre_place, -a, Phase::Carrying, s5, placed)) {return false;}
    segs.push_back(s5);

    // --- open the gripper ---------------------------------------------------- #
    std::vector<double> place_q;
    placed.copyJointGroupPositions(jmg, place_q);
    segs.push_back(
      mrct::makeDwell(
        place_q, Phase::GripOpen, spec_.disc.gripper_dwell_slots, spec_.disc.delta_t));

    // --- return: the object now sits at its PLACE pose ----------------------- #
    scene->getCurrentStateNonConst() = placed;
    setObjectState(scene, robot, task, ObjectState::AtPlace);

    Segment s6;
    moveit::core::RobotState retreated(placed);
    if (!planCartesianZ(scene, robot, placed, a, Phase::ToHome, s6, retreated)) {return false;}
    segs.push_back(s6);

    Segment s7;
    if (!planJoint(scene, robot, retreated, home, Phase::ToHome, s7)) {return false;}
    segs.push_back(s7);

    out = mrct::resampleUniform(segs, spec_.disc.delta_t);
    return true;
  }

  // ---- verification ------------------------------------------------------- #

  /// Re-check EVERY resampled sample against the scene.
  ///
  /// This is not belt-and-braces. A resampled configuration lies BETWEEN waypoints
  /// the planner validated, so it is a configuration nobody has ever checked. On a
  /// 7-DOF arm skirting the tray, the interpolant can bulge into contact between
  /// two collision-free waypoints. If that sample ends up in a trajectory, it
  /// poisons the collision matrix and the "provably collision-free schedule" claim
  /// is simply false.
  ///
  /// The scene is stepped through the object's phases as we go, so a sample is
  /// checked against the world it actually executes in.
  int validateSamples(
    const RobotCfg & robot, const TaskDef & task, const mrct::ResampledTrajectory & traj,
    double & max_cartesian_step)
  {
    auto scene = sceneFor(robot, task);
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);
    moveit::core::RobotState state(scene->getCurrentState());

    // Every link belonging to THIS robot, including the gripper (which is not in
    // the planning group but is very much part of the geometry that can collide).
    std::vector<const moveit::core::LinkModel *> links;
    for (const auto * lm : model_->getLinkModels()) {
      if (lm->getName().rfind(robot.name + "_", 0) == 0 && !lm->getShapes().empty()) {
        links.push_back(lm);
      }
    }

    int bad = 0;
    ObjectState current = ObjectState::AtSpawn;
    setObjectState(scene, robot, task, current);

    max_cartesian_step = 0.0;
    std::vector<Eigen::Vector3d> prev;

    for (std::size_t k = 0; k < traj.num_samples; ++k) {
      state.setJointGroupPositions(jmg, traj.sample(k));
      state.update();

      if (traj.object_state[k] != current) {
        current = traj.object_state[k];
        scene->getCurrentStateNonConst() = state;      // attach at the pose we are AT
        setObjectState(scene, robot, task, current);
      }

      if (scene->isStateColliding(state, robot.planning_group)) {
        if (bad < 5) {
          RCLCPP_WARN(
            log_, "  sample %zu (phase %s) is in collision", k,
            mrct::phaseName(traj.phase[k]).c_str());
        }
        ++bad;
      }

      // The REAL discretisation metric. Joint radians are not comparable across
      // joints -- a radian of wrist roll moves almost nothing, a radian of shoulder
      // sweeps the whole arm -- and they are not comparable to a scene feature size
      // in metres either. What decides whether the sampling can step OVER a thin
      // obstacle is how far the geometry actually travels between two slots.
      std::vector<Eigen::Vector3d> now;
      now.reserve(links.size());
      for (const auto * lm : links) {
        now.push_back(state.getGlobalLinkTransform(lm).translation());
      }
      if (!prev.empty()) {
        for (std::size_t i = 0; i < now.size(); ++i) {
          max_cartesian_step = std::max(max_cartesian_step, (now[i] - prev[i]).norm());
        }
      }
      prev = std::move(now);
    }
    return bad;
  }

  // ---- artifact ----------------------------------------------------------- #

  std::string toJson(
    const RobotCfg & robot, const TaskDef & task, const mrct::ResampledTrajectory & traj)
  {
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);
    std::ostringstream o;
    o.precision(17);

    o << "    {\n";
    o << "      \"robot\": \"" << robot.name << "\",\n";
    o << "      \"task\": \"" << task.id << "\",\n";
    o << "      \"object\": \"" << task.object_id << "\",\n";
    o << "      \"num_samples\": " << traj.num_samples << ",\n";
    o << "      \"used_velocities\": " << (traj.used_velocities ? "true" : "false") << ",\n";
    o << "      \"max_joint_step\": " << traj.max_joint_step << ",\n";

    o << "      \"joint_names\": [";
    const auto & names = jmg->getActiveJointModelNames();
    for (std::size_t j = 0; j < names.size(); ++j) {
      o << (j ? ", " : "") << "\"" << names[j] << "\"";
    }
    o << "],\n";

    o << "      \"positions\": [";
    for (std::size_t k = 0; k < traj.num_samples; ++k) {
      o << (k ? ", " : "") << "[";
      for (std::size_t j = 0; j < traj.num_joints; ++j) {
        o << (j ? ", " : "") << traj.sample(k)[j];
      }
      o << "]";
    }
    o << "],\n";

    o << "      \"phase\": [";
    for (std::size_t k = 0; k < traj.num_samples; ++k) {
      o << (k ? ", " : "") << static_cast<int>(traj.phase[k]);
    }
    o << "],\n";

    o << "      \"object_state\": [";
    for (std::size_t k = 0; k < traj.num_samples; ++k) {
      o << (k ? ", " : "") << static_cast<int>(traj.object_state[k]);
    }
    o << "]\n";
    o << "    }";
    return o.str();
  }

  void writeArtifact(const std::string & path, const std::vector<std::string> & entries)
  {
    std::ofstream f(path);
    if (!f) {throw std::runtime_error("cannot write " + path);}
    f.precision(17);

    f << "{\n";
    f << "  \"delta_t\": " << spec_.disc.delta_t << ",\n";
    f << "  \"gripper_dwell_slots\": " << spec_.disc.gripper_dwell_slots << ",\n";

    f << "  \"robots\": [";
    for (std::size_t r = 0; r < spec_.robots.size(); ++r) {
      f << (r ? ", " : "") << "\"" << spec_.robots[r].name << "\"";
    }
    f << "],\n";

    f << "  \"tasks\": [";
    for (std::size_t t = 0; t < spec_.tasks.size(); ++t) {
      f << (t ? ", " : "") << "\"" << spec_.tasks[t].id << "\"";
    }
    f << "],\n";

    f << "  \"precedences\": [";
    for (std::size_t p = 0; p < spec_.precedences.size(); ++p) {
      f << (p ? ", " : "") << "[\"" << spec_.precedences[p].first << "\", \""
        << spec_.precedences[p].second << "\"]";
    }
    f << "],\n";

    f << "  \"homes\": {\n";
    for (std::size_t r = 0; r < spec_.robots.size(); ++r) {
      const auto * jmg = model_->getJointModelGroup(spec_.robots[r].planning_group);
      f << "    \"" << spec_.robots[r].name << "\": [";
      const auto & names = jmg->getActiveJointModelNames();
      for (std::size_t j = 0; j < names.size(); ++j) {
        f << (j ? ", " : "") << spec_.robots[r].home.at(names[j]);
      }
      f << "]" << (r + 1 < spec_.robots.size() ? "," : "") << "\n";
    }
    f << "  },\n";

    f << "  \"trajectories\": [\n";
    for (std::size_t i = 0; i < entries.size(); ++i) {
      f << entries[i] << (i + 1 < entries.size() ? ",\n" : "\n");
    }
    f << "  ]\n}\n";
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger log_;
  TaskSpec spec_;
  moveit::core::RobotModelPtr model_;
  planning_pipeline::PlanningPipelinePtr pipeline_;
  std::map<std::string, std::set<std::string>> before_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
    "trajectory_generator",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  const std::string task_file = node->get_parameter("task_file").as_string();
  const std::string out_file = node->get_parameter("out_file").as_string();

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::thread spinner([&exec]() {exec.spin();});

  int rc = 0;
  try {
    TrajectoryGenerator gen(node, loadTaskSpec(task_file));
    rc = gen.run(out_file) ? 0 : 1;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(node->get_logger(), "%s", e.what());
    rc = 2;
  }

  exec.cancel();
  spinner.join();
  rclcpp::shutdown();
  return rc;
}
