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
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_messages.h>
#include <geometric_shapes/shape_operations.h>

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
#include <shape_msgs/msg/mesh.hpp>
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
  // How long the arc-strike and arc-out dwells last, in slots. Defaults to
  // `gripper_dwell_slots` -- the same "an actuator takes real time and the slots
  // are counted" idea, for a different actuator.
  int process_dwell_slots{10};
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
  // Tool travel speed along a process seam, in m/s. A property of the PROCESS,
  // not of the robot: 50 mm/s is a GMAW travel speed, and it is roughly a fifth
  // of what TOTG would run the same 25 mm path at.
  double process_speed{0.05};
  // How many EXTRA times `run` replans a (robot, task) whose plan failed or whose
  // resampled samples failed `validateSamples`. OMPL is unseeded, so a replan is an
  // independent draw; a scene with a thin obstacle beside the corridor (the
  // `tower_wall` bars) loses 1-4 of its 48 trajectories per run to a contorted plan
  // whose Hermite interpolant leaves the validated path, and a second draw is
  // almost always clean. Sound by construction: every accepted trajectory has
  // passed `validateSamples`. A scene that plans cleanly never retries.
  int plan_retries{6};
  int seed{7};
  // Whether an OCCUPIED place slot enters another task's world as its object's MESH rather
  // than its bounding box, when every candidate that could fill the slot carries the same
  // mesh file and scale (`sceneFor`, (a)). Off by default: the box is conservative and the
  // shipped scenes were measured with it. A part married onto another one needs it: the
  // bounding box of a placed door inner covers every flange its welds must reach.
  bool placed_meshes{false};
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
  // Link-name prefixes that belong to this robot (default: "<name>_"). Every link
  // used for the per-slot Cartesian step check must match one of these -- a robot
  // whose links are named e.g. "arm_left_*" needs this set explicitly, since its
  // link names do not start with its scene-graph robot name "left".
  std::vector<std::string> link_prefixes;
  // The gripper's approach axis in `ee_link`'s own frame (default "z", the UR
  // tool0 convention). A parallel-jaw grasp is symmetric under a half-turn about
  // THIS axis, so the grasp-symmetry flip must rotate about it. PAL's
  // gripper_*_grasping_link approaches along +x: flipping about z there would
  // turn the fingers upward instead of swapping them.
  Eigen::Vector3d tool_approach_axis{Eigen::Vector3d::UnitZ()};
};

/// Optional real geometry for an object or a fixture: an STL/DAE/OBJ file that assimp
/// can read, in METRES after `scale`, whose origin is the CENTRE of its bounding box.
///
/// That origin convention is what makes a mesh a drop-in for a box: `spawn`, `place`,
/// `grasp` and `pose` mean exactly what they mean for a box of the same `size`, because a
/// box's centre already is its origin. `size` stays mandatory and is the mesh's bounding
/// box (`loadTaskSpec` checks that it is), so every other stage -- the VAMP/FCL collision
/// matrices, refinement, coordination -- keeps reading `size` alone and never sees a mesh:
/// the seam stays geometry-free (ADR-0002). Where a mesh is used and where its box
/// stands in: the mesh addendum of ADR-0003 and `sceneFor`.
struct MeshRef
{
  std::string file;      ///< absolute path, resolved once at load time
  double scale{1.0};
};

struct ObjectDef
{
  std::string id;
  std::array<double, 3> size{};   ///< bounding box; the mesh (if any) lies inside it
  std::optional<MeshRef> mesh;
  geometry_msgs::msg::Pose spawn;
  geometry_msgs::msg::Pose grasp_in_obj;
  /// Fixtures this object may also rest on, besides the support surface: their contact
  /// with it is tolerated exactly where the table's is (`supportAcm`). Optional; for a part
  /// set onto another part, e.g. a door inner onto its skin, whose flanges touch.
  std::vector<std::string> supports;
};

struct FixtureDef
{
  std::string id;
  std::array<double, 3> size{};   ///< bounding box; the mesh (if any) lies inside it
  std::optional<MeshRef> mesh;
  geometry_msgs::msg::Pose pose;
};

/// What a task IS. A task has always been "a trajectory, a duration and a
/// collision profile" as far as everything downstream is concerned -- nothing in
/// the seam, the solver or the plan graph inspects what the arm is doing while it
/// runs. That is what lets a second kind of task in here without a model change:
/// `kind` never leaves this file.
enum class TaskKind : std::uint8_t
{
  PickPlace = 0,  ///< fetch an object from its spawn pose and set it at its place pose
  Weld = 1,       ///< run a tool along a seam; nothing is grasped, nothing moves
};

struct TaskDef
{
  std::string id;
  TaskKind kind{TaskKind::PickPlace};

  /// Which SLOT of the plan this task is a candidate for.
  ///
  /// A slot is a place in the assembly that several tasks compete to fill -- the
  /// same tower level, reachable with any of four interchangeable cubes. Exactly
  /// one candidate per slot runs, and each physical object is consumed at most
  /// once (APEX-MR's delta_jt). A task declared the ordinary way under `tasks:`
  /// is its OWN slot, so every scene that predates `slots:` is the degenerate
  /// singleton case and nothing about it changes.
  std::string slot_id;

  // --- PickPlace ---------------------------------------------------------- #
  std::string object_id;
  geometry_msgs::msg::Pose place;

  // --- Weld --------------------------------------------------------------- #
  geometry_msgs::msg::Pose seam_start;   ///< world pose of the seam's first point
  geometry_msgs::msg::Pose seam_end;     ///< world pose of the seam's last point
  geometry_msgs::msg::Pose tool;         ///< EE pose relative to a seam point
  double speed{0.05};                    ///< traverse speed, m/s
  double approach{0.12};                 ///< how far above the seam the pre-start sits
};

struct TaskSpec
{
  Discretisation disc;
  PlanningCfg planning;
  std::string base_frame{"world"};
  /// The surface a task's object rests on at its spawn and place poses, whose
  /// contact with THAT object is tolerated during pick and place only (see
  /// `supportAcm`). Optional in the YAML; the cell's table link by default.
  std::string support_surface{"table_top"};
  std::vector<RobotCfg> robots;
  std::vector<FixtureDef> fixtures;
  std::vector<ObjectDef> objects;
  std::vector<TaskDef> tasks;
  std::vector<std::pair<std::string, std::string>> precedences;
  /// One entry per precedence, in the same order: how the pair must be enforced.
  /// Derived HERE, offline, from what the tasks are -- so the seam still carries
  /// no types and the solver never learns what a weld is (ADR-0001/0002).
  std::vector<std::string> precedence_modes;

  /// True iff the scene declared a `slots:` block.
  ///
  /// This gates whether the slot keys reach the seam AT ALL, and that is not
  /// cosmetic: `SchedulingProblem.slot_of is not None` is what the solver's
  /// `objective="apex"` guard and the Gurobi backends' `_require_no_slots` test.
  /// Emitting all-singleton slot groups for an ordinary scene would be a no-op
  /// for CP-SAT/makespan and an outright NotImplementedError for every existing
  /// apex and Gurobi study. So a scene without `slots:` writes no slot keys and
  /// its seam stays byte-identical.
  bool has_slots{false};
  /// Ordering between SLOTS, independent of which candidate wins each of them.
  std::vector<std::pair<std::string, std::string>> slot_precedences;
  /// Parallel to `slot_precedences`, same vocabulary as `precedence_modes`.
  std::vector<std::string> slot_precedence_modes;

  const ObjectDef & object(const std::string & id) const
  {
    for (const auto & o : objects) {
      if (o.id == id) {return o;}
    }
    throw std::runtime_error("unknown object: " + id);
  }

  const TaskDef & task(const std::string & id) const
  {
    for (const auto & t : tasks) {
      if (t.id == id) {return t;}
    }
    throw std::runtime_error("unknown task: " + id);
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

// --------------------------------------------------------------------------- #
// Mesh geometry
// --------------------------------------------------------------------------- #
/// The `shape_msgs::Mesh` for a file at a scale, loaded ONCE per process.
///
/// `sceneFor` rebuilds the world on every call -- several times per trajectory -- and
/// reading and triangulating the file each time would dominate. What is cached is the
/// message; MoveIt still builds its own collision BVH from it inside
/// `processCollisionObjectMsg`, which is the unavoidable part of the cost.
const shape_msgs::msg::Mesh & meshMsg(const MeshRef & ref)
{
  static std::map<std::pair<std::string, double>, shape_msgs::msg::Mesh> cache;
  const auto key = std::make_pair(ref.file, ref.scale);
  if (const auto it = cache.find(key); it != cache.end()) {return it->second;}

  std::unique_ptr<shapes::Mesh> mesh(
    shapes::createMeshFromResource("file://" + ref.file, Eigen::Vector3d::Constant(ref.scale)));
  if (!mesh || mesh->triangle_count == 0) {
    throw std::runtime_error("mesh '" + ref.file + "' could not be loaded (or has no triangles)");
  }
  shapes::ShapeMsg msg;
  if (!shapes::constructMsgFromShape(mesh.get(), msg)) {
    throw std::runtime_error("mesh '" + ref.file + "' could not be converted to a shape message");
  }
  return cache.emplace(key, boost::get<shape_msgs::msg::Mesh>(msg)).first->second;
}

/// Absolute path of a mesh named in a scene YAML. Relative to the YAML's own directory
/// first -- a scene archived under `artifacts/runs/<name>/` carries its meshes beside it
/// -- then to the installed `config/` of this package, where the shipped meshes live.
std::string resolveMeshFile(const std::string & file, const std::string & yaml_path)
{
  namespace fs = std::filesystem;
  if (fs::path(file).is_absolute()) {
    if (fs::exists(file)) {return file;}
    throw std::runtime_error("mesh file '" + file + "' does not exist");
  }
  const fs::path beside = fs::absolute(fs::path(yaml_path)).parent_path() / file;
  if (fs::exists(beside)) {return beside.lexically_normal().string();}
  std::string tried = beside.lexically_normal().string();
  try {
    const fs::path shared =
      fs::path(ament_index_cpp::get_package_share_directory("multi_robot_cell_tamp")) /
      "config" / file;
    if (fs::exists(shared)) {return shared.lexically_normal().string();}
    tried += "' nor '" + shared.lexically_normal().string();
  } catch (const std::exception &) {
    // package not installed (running from a build tree): the sibling path is all there is
  }
  throw std::runtime_error("mesh file '" + file + "' not found (looked in '" + tried + "')");
}

/// Parse the optional `mesh: {file: ..., scale: ...}` of an object or fixture and check it
/// against `size`.
///
/// `size` is what every other stage sees, so a mesh that disagrees with it would make the
/// planning scene and the collision matrices describe different objects -- and the check is
/// asymmetric on purpose. The mesh must lie INSIDE the `size` box centred on its origin
/// (0.1 mm slack: STL stores float32), which is what makes the box a true superset and the
/// bounding-box stand-ins `sceneFor` builds for the OTHER objects sound; and the box may not
/// be more than 2 mm looser than the mesh, so `size` stays a bounding box and not a guess.
/// The second rule also bounds how far the mesh's centre can be from its origin, which is
/// the origin convention.
std::optional<MeshRef> parseMesh(
  const YAML::Node & n, const std::string & what, const std::array<double, 3> & size,
  const std::string & yaml_path)
{
  if (!n["mesh"]) {return std::nullopt;}
  const auto & m = n["mesh"];
  if (!m.IsMap() || !m["file"]) {
    throw std::runtime_error(what + ": `mesh:` must be a map with a `file:` key");
  }
  MeshRef ref;
  ref.file = resolveMeshFile(m["file"].as<std::string>(), yaml_path);
  ref.scale = m["scale"] ? m["scale"].as<double>() : 1.0;
  if (!(ref.scale > 0.0)) {
    throw std::runtime_error(what + ": mesh `scale` must be positive");
  }

  const auto & verts = meshMsg(ref).vertices;
  std::array<double, 3> lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30};
  for (const auto & v : verts) {
    const double c[3] = {v.x, v.y, v.z};
    for (int k = 0; k < 3; ++k) {lo[k] = std::min(lo[k], c[k]); hi[k] = std::max(hi[k], c[k]);}
  }
  constexpr double kSlack = 1e-4;    // float32 STL vertices
  constexpr double kLoose = 2e-3;    // how much bigger than the mesh `size` may be
  static const char axis[3] = {'x', 'y', 'z'};
  for (int k = 0; k < 3; ++k) {
    const double half = 0.5 * size[k];
    const bool inside = lo[k] >= -half - kSlack && hi[k] <= half + kSlack;
    const bool tight = size[k] - (hi[k] - lo[k]) <= kLoose;
    if (!inside || !tight) {
      std::ostringstream o;
      o.precision(4);
      o << std::fixed << what << ": mesh '" << m["file"].as<std::string>()
        << "' does not match `size` along " << axis[k] << ": the mesh spans [" << lo[k] << ", "
        << hi[k] << "] m (extent " << hi[k] - lo[k] << ") but `size` " << size[k]
        << " puts the box at [" << -half << ", " << half << "]. `size` is the bounding box the "
        "other stages see: the mesh must lie inside it, its origin must be the centre of its "
        "bounding box, and `size` may exceed the mesh by at most 2 mm (check `scale` and units).";
      throw std::runtime_error(o.str());
    }
  }
  return ref;
}

TaskSpec loadTaskSpec(const std::string & path)
{
  YAML::Node root = YAML::LoadFile(path);
  TaskSpec s;

  const auto & d = root["discretisation"];
  s.disc.delta_t = d["delta_t"].as<double>();
  s.disc.gripper_dwell_slots = d["gripper_dwell_slots"].as<int>();
  // NOTE the guard, and note that the in-struct default above is dead code: the
  // parser dereferences every key it names, so an absent key throws. Every key
  // added from here on must be optional in exactly this shape, or the eighteen
  // scenes that predate it stop loading.
  s.disc.process_dwell_slots = d["process_dwell_slots"]
    ? d["process_dwell_slots"].as<int>() : s.disc.gripper_dwell_slots;
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
  s.planning.process_speed = p["process_speed"] ? p["process_speed"].as<double>() : 0.05;
  s.planning.seed = p["seed"].as<int>();
  s.planning.plan_retries = p["plan_retries"] ? p["plan_retries"].as<int>() : 6;
  s.planning.placed_meshes = p["placed_meshes"] ? p["placed_meshes"].as<bool>() : false;

  s.base_frame = root["base_frame"].as<std::string>();
  s.support_surface = root["support_surface"]
    ? root["support_surface"].as<std::string>() : std::string{"table_top"};

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
    if (n["tool_approach_axis"]) {
      const auto axis = n["tool_approach_axis"].as<std::string>();
      if (axis == "x") {r.tool_approach_axis = Eigen::Vector3d::UnitX();}
      else if (axis == "y") {r.tool_approach_axis = Eigen::Vector3d::UnitY();}
      else if (axis == "z") {r.tool_approach_axis = Eigen::Vector3d::UnitZ();}
      else {
        throw std::runtime_error(
          "robot '" + r.name + "': tool_approach_axis must be x, y or z, got '" + axis + "'");
      }
    }
    if (n["link_prefixes"]) {
      for (const auto & lp : n["link_prefixes"]) {r.link_prefixes.push_back(lp.as<std::string>());}
    } else {
      r.link_prefixes.push_back(r.name + "_");
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
      f.mesh = parseMesh(n, "fixture '" + f.id + "'", f.size, path);
      f.pose = poseFromYaml(n["pose"]);
      s.fixtures.push_back(f);
    }
  }
  for (const auto & n : root["objects"]) {
    ObjectDef o;
    o.id = n["id"].as<std::string>();
    o.size = {n["size"][0].as<double>(), n["size"][1].as<double>(), n["size"][2].as<double>()};
    o.mesh = parseMesh(n, "object '" + o.id + "'", o.size, path);
    o.spawn = poseFromYaml(n["spawn"]);
    o.grasp_in_obj = poseFromYaml(n["grasp"]);
    for (const auto & f : n["supports"]) {
      const auto id = f.as<std::string>();
      const bool known = std::any_of(
        s.fixtures.begin(), s.fixtures.end(), [&](const FixtureDef & fx) {return fx.id == id;});
      if (!known) {
        throw std::runtime_error("object '" + o.id + "': supports '" + id + "', which is no fixture");
      }
      o.supports.push_back(id);
    }
    s.objects.push_back(o);
  }
  for (const auto & n : root["tasks"]) {
    TaskDef t;
    t.kind = TaskKind::PickPlace;
    t.id = n["id"].as<std::string>();
    t.slot_id = t.id;             // a hand-written task is its own singleton slot
    t.object_id = n["object"].as<std::string>();
    t.place = poseFromYaml(n["place"]);
    s.tasks.push_back(t);
  }

  // Interchangeable slots. Declared in their own block for the same reason
  // `welds:` is (see below): the `tasks:` loop above dereferences `object`
  // unconditionally, and a slot names a LIST of candidate objects instead.
  //
  // One slot expands to one task PER CANDIDATE -- same place pose, different
  // object -- all tagged with the slot id. They are ordinary pick-and-place tasks
  // from here on: the generator plans every one of them for both robots, and the
  // seam tells the solver that exactly one per slot runs and that no physical
  // object is consumed twice. Nothing between the two knows what a slot is.
  //
  // The expanded id is `<slot_id>__<object_id>`: deterministic, and unique as long
  // as the candidate list has no duplicates (checked) -- so a scene reads the same
  // way in the schedule, the plan graph and the executors' logs.
  if (root["slots"]) {
    s.has_slots = true;
    std::set<std::string> slot_ids;
    for (const auto & n : root["slots"]) {
      const std::string slot_id = n["id"].as<std::string>();
      if (!slot_ids.insert(slot_id).second) {
        throw std::runtime_error("two slots share the id '" + slot_id + "'");
      }
      const auto place = poseFromYaml(n["place"]);
      if (!n["candidates"] || n["candidates"].size() == 0) {
        throw std::runtime_error("slot '" + slot_id + "' has no candidates");
      }
      std::set<std::string> seen;
      for (const auto & c : n["candidates"]) {
        TaskDef t;
        t.kind = TaskKind::PickPlace;
        t.slot_id = slot_id;
        t.object_id = c.as<std::string>();
        if (!seen.insert(t.object_id).second) {
          throw std::runtime_error(
            "slot '" + slot_id + "' lists candidate '" + t.object_id + "' twice");
        }
        s.object(t.object_id);    // throws if the scene does not declare it
        t.id = slot_id + "__" + t.object_id;
        for (const auto & existing : s.tasks) {
          if (existing.id == t.id) {
            throw std::runtime_error("slot expansion collides with task id '" + t.id + "'");
          }
        }
        t.place = place;
        s.tasks.push_back(t);
      }
    }
  }

  // Welds are declared in their own top-level block, not as a discriminated entry
  // inside `tasks:`. The loop above dereferences `object` and `place`
  // unconditionally; making those conditional is the kind of edit that changes
  // behaviour for an existing scene without anyone noticing. A separate block is
  // guarded by one `if`, exactly like `fixtures:`, and cannot touch a scene that
  // does not use the key.
  //
  // They land in the SAME task vector, though: a weld is a scheduled task like
  // any other -- it needs a duration, forbidden offsets and milestones, and
  // `precedences` has to be able to name it in one id namespace. Only the YAML
  // surface is separate. Order is pick-places first, then welds: deterministic,
  // which is all anything downstream needs.
  if (root["welds"]) {
    for (const auto & n : root["welds"]) {
      TaskDef t;
      t.kind = TaskKind::Weld;
      t.id = n["id"].as<std::string>();
      t.slot_id = t.id;           // a weld is its own singleton slot
      t.seam_start = poseFromYaml(n["start"]);
      t.seam_end = poseFromYaml(n["end"]);
      // Same convention as `objects[].grasp`: an EE pose expressed relative to
      // the point being worked on, composed with it to give the world EE pose.
      t.tool = poseFromYaml(n["tool"]);
      t.speed = n["speed"] ? n["speed"].as<double>() : s.planning.process_speed;
      t.approach = n["approach"] ? n["approach"].as<double>() : s.planning.approach;
      s.tasks.push_back(t);
    }
  }

  if (root["precedences"]) {
    auto kindOf = [&s](const std::string & id) {
        for (const auto & t : s.tasks) {
          if (t.id == id) {return t.kind;}
        }
        throw std::runtime_error("a precedence names a task the scene does not define: " + id);
      };
    for (const auto & n : root["precedences"]) {
      const std::string i = n[0].as<std::string>();
      const std::string j = n[1].as<std::string>();
      const TaskKind ki = kindOf(i);
      const TaskKind kj = kindOf(j);

      // Which relation(s) a pair must be enforced with. The seam carries only the
      // strings; what they mean lives in the solver and the plan graph:
      //
      //   pipeline   start[j] >= m0[i]  and  m1[i] <= m1[j]
      //   gate       m0[j] >= m1[i]
      //
      // PickPlace -> PickPlace: `pipeline`, as it always was.
      // Weld -> Weld:           `gate` -- the closing seam must not strike before
      //                          the tack is finished.
      // PickPlace -> Weld:      BOTH, as two entries. `gate` says the arc may not
      //                          strike before the part is released, but on its own
      //                          it does NOT imply the pick gate: the weld robot
      //                          could set off before the part has even been picked.
      //                          ADR-0003 plans the weld against the part at its
      //                          PLACE pose only (the precedence prunes the spawn
      //                          copy away), so a weld trajectory that starts while
      //                          the part still sits in its nest is unsound. The
      //                          pipeline pair adds start[j] >= m0[i], which closes
      //                          it. The solver accepts one pair listed once per mode.
      // Weld -> PickPlace:      `pipeline` (no gate semantics were asked for).
      //
      // An explicit third element overrides the derivation with that single mode.
      std::vector<std::string> modes;
      if (n.size() > 2) {
        const std::string mode = n[2].as<std::string>();
        if (mode != "pipeline" && mode != "gate") {
          throw std::runtime_error(
            "precedence [" + i + ", " + j + "] names an unknown mode '" + mode +
            "' (expected 'pipeline' or 'gate')");
        }
        if (mode == "gate" && ki == TaskKind::PickPlace && kj == TaskKind::Weld) {
          RCLCPP_WARN(
            rclcpp::get_logger("trajectory_generator"),
            "precedence [%s, %s, gate]: an explicit gate on a pick-and-place -> weld pair "
            "drops the pick gate. The weld is planned with %s's part at its place pose "
            "ONLY (ADR-0003), so without start[weld] >= pick[%s] the plan is unsound. "
            "Remove the third element to get both relations.",
            i.c_str(), j.c_str(), i.c_str(), i.c_str());
        }
        modes.push_back(mode);
      } else if (ki == TaskKind::PickPlace && kj == TaskKind::Weld) {
        modes = {"pipeline", "gate"};
      } else if (ki == TaskKind::Weld && kj == TaskKind::Weld) {
        modes = {"gate"};
      } else {
        modes = {"pipeline"};
      }
      for (const auto & mode : modes) {
        s.precedences.emplace_back(i, j);
        s.precedence_modes.push_back(mode);
      }
    }
  }

  // Ordering between SLOTS. A task precedence cannot express it: it names two
  // specific tasks, and which task wins each slot is exactly what the solver is
  // deciding. "level 1 before level 2" has to hold whichever cube fills either.
  //
  // Same surface syntax as `precedences:` (an optional third element overrides
  // the mode), but no derivation: a slot has no kind, so the default is the
  // `pipeline` every pick-and-place pair already gets.
  if (root["slot_precedences"]) {
    std::set<std::string> slot_ids;
    for (const auto & t : s.tasks) {slot_ids.insert(t.slot_id);}
    for (const auto & n : root["slot_precedences"]) {
      const std::string a = n[0].as<std::string>();
      const std::string b = n[1].as<std::string>();
      for (const auto & id : {a, b}) {
        if (!slot_ids.count(id)) {
          throw std::runtime_error(
            "a slot precedence names a slot the scene does not define: " + id);
        }
      }
      std::string mode = "pipeline";
      if (n.size() > 2) {
        mode = n[2].as<std::string>();
        if (mode != "pipeline" && mode != "gate") {
          throw std::runtime_error(
            "slot precedence [" + a + ", " + b + "] names an unknown mode '" + mode +
            "' (expected 'pipeline' or 'gate')");
        }
      }
      s.slot_precedences.emplace_back(a, b);
      s.slot_precedence_modes.push_back(mode);
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

/// What to call a straight-line EE step in a log line. The approach and the
/// retreat are overwhelmingly the common cases and deserve their own words; a
/// process traverse is anything else. Naming the step matters more than it looks:
/// the diagnostic below is the only thing that tells you WHY a Cartesian motion
/// stalled, and it used to say "descent" whatever the direction of travel.
const char * motionName(const Eigen::Vector3d & delta)
{
  if (std::abs(delta.z()) >= 0.999 * delta.norm()) {
    return delta.z() < 0.0 ? "descent" : "retreat";
  }
  return "traverse";
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

/// A box of `size`, or -- when `mesh` is given -- the mesh itself.
///
/// The pose means the same thing either way: a mesh's origin is the centre of its bounding
/// box (see `MeshRef`), exactly as a box's is, so callers pass the same `spawn`/`place`/
/// `pose` and the same `size` and only choose WHICH representation they want. The choice is
/// not made here: it is the scene builder's, and it is a decision about cost, not about
/// correctness -- see `sceneFor`. An attached mesh behaves as an attached box does: MoveIt
/// takes the pose in `frame` at the state it is attached in and rides the link from there.
moveit_msgs::msg::CollisionObject makeObject(
  const std::string & id, const std::array<double, 3> & size,
  const std::optional<MeshRef> & mesh, const geometry_msgs::msg::Pose & pose,
  const std::string & frame)
{
  if (!mesh) {return makeBox(id, size, pose, frame);}
  moveit_msgs::msg::CollisionObject c;
  c.header.frame_id = frame;
  c.id = id;
  c.meshes.push_back(meshMsg(*mesh));
  c.mesh_poses.push_back(pose);
  c.operation = c.ADD;
  return c;
}

// --------------------------------------------------------------------------- #
// Precedence closure (ADR-0003)
// --------------------------------------------------------------------------- #
/// Transitive closure of the precedence relation: `before[j]` = every task that
/// must complete before j. Used to prune the offline environment -- an object
/// whose task provably ran already is at its PLACE pose, not its spawn. With slots
/// the environment is assembled per SLOT and per OBJECT from this task-level
/// relation, not per task: see `slotTable` and `sceneFor`.
std::map<std::string, std::set<std::string>> precedenceClosure(const TaskSpec & spec)
{
  std::map<std::string, std::set<std::string>> before;
  for (const auto & t : spec.tasks) {before[t.id] = {};}
  for (const auto & [i, j] : spec.precedences) {before[j].insert(i);}

  // A slot precedence carries the same GEOMETRIC weight as a task precedence
  // (ADR-0003), and it has to be expanded to task pairs HERE to get it. "slot a
  // before slot b" means whichever candidate wins a has run before whichever wins
  // b -- so for the purposes of the environment, EVERY candidate of a is before
  // EVERY candidate of b. That is sound because only one candidate per slot ever
  // runs: the union over candidates of a is the union over the possible worlds,
  // which is precisely what ADR-0003's union environment is.
  //
  // Without this the six-level interchangeable tower would plan every level
  // against every other level's place pose as an "unordered" obstacle -- a box
  // sitting exactly where the arm has to descend -- and come back INFEASIBLE.
  // The seam still carries the slot-level pairs, never this expansion: the solver
  // must not be told that a specific candidate precedes another specific one.
  if (!spec.slot_precedences.empty()) {
    std::map<std::string, std::vector<std::string>> members;
    for (const auto & t : spec.tasks) {members[t.slot_id].push_back(t.id);}
    for (const auto & [a, b] : spec.slot_precedences) {
      for (const auto & ti : members[a]) {
        for (const auto & tj : members[b]) {before[tj].insert(ti);}
      }
    }
  }

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

/// What the environment builder knows about the SLOTS of a scene (ADR-0003, the
/// 2026-09-19 addendum). A scene without `slots:` is the degenerate case -- every
/// task is its own singleton slot -- so this is the one vocabulary for both.
struct SlotTable
{
  /// Slot id -> the tasks (by index into `spec.tasks`) that compete to fill it.
  std::map<std::string, std::vector<std::size_t>> members;
  /// `certainly_before[b]` = the slots that have run BEFORE slot b in EVERY
  /// schedule, whichever candidate wins either. See `slotTable` for why "every".
  std::map<std::string, std::set<std::string>> certainly_before;
  /// Slot id -> the object it is filled with, if it has exactly ONE pick-and-place
  /// candidate. With one candidate the object is known, not conjectured.
  std::map<std::string, std::string> sole_object;
};

SlotTable slotTable(
  const TaskSpec & spec, const std::map<std::string, std::set<std::string>> & before)
{
  SlotTable t;
  for (std::size_t i = 0; i < spec.tasks.size(); ++i) {
    t.members[spec.tasks[i].slot_id].push_back(i);
  }
  for (const auto & [sid, idx] : t.members) {
    if (idx.size() == 1 && spec.tasks[idx.front()].kind == TaskKind::PickPlace) {
      t.sole_object[sid] = spec.tasks[idx.front()].object_id;
    }
  }
  // Slot `a` certainly precedes slot `b` iff EVERY candidate of `a` precedes EVERY
  // candidate of `b` in the task closure -- "all", never "any". The relation is
  // used in two directions and they want opposite approximations: an object is
  // dropped from the world only if its slot CERTAINLY ran before, and a place pose
  // is dropped only if its slot CERTAINLY runs after. "any" would be optimistic in
  // both. With slot precedences alone (all the scenes so far) the two coincide,
  // because `precedenceClosure` expands them to every candidate pair; they differ
  // only if a task-level `precedences:` entry names a single expanded candidate id.
  for (const auto & [b, bidx] : t.members) {
    for (const auto & [a, aidx] : t.members) {
      if (a == b) {continue;}
      bool all = true;
      for (const auto ib : bidx) {
        const auto & preds = before.at(spec.tasks[ib].id);
        for (const auto ia : aidx) {
          if (!preds.count(spec.tasks[ia].id)) {all = false;}
        }
      }
      if (all) {t.certainly_before[b].insert(a);}
    }
  }
  return t;
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

    for (const auto & f : spec_.fixtures) {logMesh("fixture", f.id, f.mesh);}
    for (const auto & o : spec_.objects) {logMesh("object", o.id, o.mesh);}

    pipeline_ = std::make_shared<planning_pipeline::PlanningPipeline>(model_, node_, "ompl");
    before_ = precedenceClosure(spec_);
    slots_ = slotTable(spec_, before_);
  }

  bool run(const std::string & out_path)
  {
    std::vector<std::string> artifact;   // one JSON object per (robot, task)
    int n_missing = 0;       // pairs with no trajectory at all
    int n_invalid = 0;       // pairs written, but failing validateSamples
    int total_retries = 0;   // replans over the whole run (0 for a scene that plans clean)
    double worst_step = 0.0;

    for (const auto & robot : spec_.robots) {
      for (const auto & task : spec_.tasks) {
        RCLCPP_INFO(log_, "=== planning %s / %s ===", robot.name.c_str(), task.id.c_str());
        // Plan, then re-verify every resampled sample; on failure of either, draw again
        // (up to `plan_retries` more times) and keep the FIRST trajectory that verifies.
        // If none does, keep the last attempt's outcome, exactly as a single attempt would.
        mrct::ResampledTrajectory traj;
        double cart_step = 0.0;
        bool planned = false;
        int bad = 0;
        int used_retries = 0;
        const int max_tries = 1 + std::max(0, spec_.planning.plan_retries);
        // Wall time of the pair, split so a heavier scene (real meshes) can be costed:
        // planning, the re-validation of every resampled sample, and -- inside both --
        // the world rebuilds `sceneFor` does, which is where a mesh's BVH is built.
        using Clock = std::chrono::steady_clock;
        const auto seconds = [](Clock::time_point a) {
            return std::chrono::duration<double>(Clock::now() - a).count();
          };
        double plan_s = 0.0, validate_s = 0.0;
        const double ik_s0 = ik_seconds_;
        const double scene_s0 = scene_seconds_;
        const long scene_n0 = scene_builds_;
        for (int attempt = 0; attempt < max_tries; ++attempt) {
          mrct::ResampledTrajectory cand;
          const auto t_plan = Clock::now();
          const bool ok = planTask(robot, task, cand);
          plan_s += seconds(t_plan);
          int cand_bad = 0;
          double cand_step = 0.0;
          if (ok) {
            const auto t_val = Clock::now();
            cand_bad = validateSamples(robot, task, cand, cand_step);
            validate_s += seconds(t_val);
          }
          if (ok && cand_bad == 0) {
            traj = std::move(cand);
            cart_step = cand_step;
            planned = true;
            bad = 0;
            used_retries = attempt;
            break;
          }
          if (ok) {          // keep the latest planned-but-invalid one for the report
            traj = std::move(cand);
            cart_step = cand_step;
            planned = true;
            bad = cand_bad;
          }
          used_retries = attempt;
          if (attempt + 1 < max_tries) {
            const std::string why =
              ok ? std::to_string(cand_bad) + " sample(s) in collision" : "no plan";
            RCLCPP_WARN(
              log_, "%s / %s: attempt %d discarded (%s) -- retry %d/%d",
              robot.name.c_str(), task.id.c_str(), attempt + 1, why.c_str(), attempt + 1,
              max_tries - 1);
          }
        }
        total_retries += used_retries;
        if (!planned) {
          RCLCPP_ERROR(
            log_, "FAILED to plan %s / %s after %d attempt(s) -- every (robot, task) pair "
            "needs a trajectory, so the artifact is incomplete", robot.name.c_str(),
            task.id.c_str(), max_tries);
          ++n_missing;
          continue;
        }
        if (bad > 0) {
          RCLCPP_ERROR(
            log_, "%s / %s: %d resampled sample(s) are IN COLLISION after %d attempt(s) -- "
            "the interpolant left the validated path. Reduce delta_t or densify the plan.",
            robot.name.c_str(), task.id.c_str(), bad, max_tries);
          ++n_invalid;
        }

        worst_step = std::max(worst_step, cart_step);
        RCLCPP_INFO(
          log_, "%s / %s: K=%zu slots (%.2f s), max joint step %.4f, "
          "max link travel per slot %.4f m",
          robot.name.c_str(), task.id.c_str(), traj.num_samples,
          traj.num_samples * spec_.disc.delta_t, traj.max_joint_step, cart_step);
        RCLCPP_INFO(
          log_, "%s / %s: timing: plan %.2f s, validateSamples %.2f s (%d attempt(s)); "
          "%ld scene build(s), %.2f s", robot.name.c_str(), task.id.c_str(), plan_s, validate_s,
          used_retries + 1, scene_builds_ - scene_n0, scene_seconds_ - scene_s0);

        const PairTiming timing{ik_seconds_ - ik_s0, plan_s, validate_s, used_retries + 1};
        artifact.push_back(toJson(robot, task, traj, &timing));
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
    const std::size_t expected = spec_.robots.size() * spec_.tasks.size();
    if (total_retries > 0) {
      RCLCPP_WARN(log_, "plan retries used over the whole run: %d", total_retries);
    }
    const bool all_ok = (n_missing == 0 && n_invalid == 0);
    if (all_ok) {
      RCLCPP_INFO(
        log_, "OK: %zu/%zu trajectories written to %s", artifact.size(), expected,
        out_path.c_str());
    } else if (n_missing > 0) {
      RCLCPP_ERROR(
        log_, "INCOMPLETE: %zu/%zu trajectories written to %s (%d FAILED to plan, %d failed "
        "validation)", artifact.size(), expected, out_path.c_str(), n_missing, n_invalid);
    } else {
      RCLCPP_ERROR(
        log_, "INVALID: %zu/%zu written to %s, %d failed validation (in collision after "
        "%d attempt(s) each)", artifact.size(), expected, out_path.c_str(), n_invalid,
        1 + std::max(0, spec_.planning.plan_retries));
    }
    return all_ok;
  }

  /// Replan each robot's SCHEDULED tasks as one continuous chain, skipping home entirely.
  ///
  /// **This mode produces plans that cannot be executed, and is retained only to reproduce
  /// that result** (ADR-0008). It is never run by the pipeline; `runTransits` is.
  ///
  /// The motivation was sound -- the commute is most of all motion, and chaining removes a
  /// large part of it (ADR-0008 records the figures). What it also removes is the arms' only way of
  /// getting out of each other's way. Each task is planned with the other robot parked at
  /// home (ADR-0003), an approximation the unrefined plan survives *because* every task
  /// begins and ends at home, so conflicting motions can always be separated in time.
  /// Chained, the arms are permanently in the shared workspace: the resulting pair of paths
  /// admits no monotone coordination at all, at any timing, for either robot alone or both
  /// (`scripts/coordinate.py` reports it in about a second).
  ///
  /// The shipped refinement keeps the retreat and cuts only its overshoot.
  bool runChains(const std::string & out_path, const std::string & schedule_file)
  {
    chains_ = readSchedule(schedule_file);
    std::vector<std::string> artifact;
    bool all_ok = true;
    double worst_step = 0.0;

    for (const auto & robot : spec_.robots) {
      const auto & seq = chains_[robot.name];
      if (seq.empty()) {
        RCLCPP_INFO(log_, "%s is assigned no task; nothing to chain", robot.name.c_str());
        continue;
      }
      moveit::core::RobotState state = homeState();
      std::size_t chained = 0;

      for (std::size_t k = 0; k < seq.size(); ++k) {
        const TaskDef & task = taskById(seq[k]);
        const bool last = (k + 1 == seq.size());
        RCLCPP_INFO(
          log_, "=== chaining %s / %s (%zu of %zu)%s ===", robot.name.c_str(),
          task.id.c_str(), k + 1, seq.size(), last ? ", returns home" : "");

        std::vector<Segment> segs;
        moveit::core::RobotState end(state);
        if (k > 0 && planTaskFrom(robot, task, state, last, segs, end)) {
          ++chained;
        } else {
          // Either this is the first task (it genuinely starts at home), or the direct
          // transit was infeasible. Falling back to the home round-trip costs time but is
          // always available, so refinement can never make a scene unplannable.
          segs.clear();
          if (k > 0) {
            RCLCPP_WARN(
              log_, "  direct transit into %s failed; keeping the home round-trip",
              task.id.c_str());
            Segment via;
            auto scene = sceneFor(robot, task);
            if (task.kind == TaskKind::PickPlace) {
              setObjectState(scene, robot, task, ObjectState::AtSpawn);
            }
            if (!planJoint(scene, robot, state, homeState(), Phase::ToPick, via)) {
              RCLCPP_ERROR(log_, "  and so did the flight home -- %s is unreachable",
                           task.id.c_str());
              all_ok = false;
              continue;
            }
            segs.push_back(via);
          }
          std::vector<Segment> rest;
          if (!planTaskFrom(robot, task, homeState(), last, rest, end)) {
            RCLCPP_ERROR(log_, "FAILED to plan %s / %s", robot.name.c_str(), task.id.c_str());
            all_ok = false;
            continue;
          }
          segs.insert(segs.end(), rest.begin(), rest.end());
        }

        auto traj = mrct::resampleUniform(segs, spec_.disc.delta_t);
        flattenObjectState(task, traj);
        double cart_step = 0.0;
        const int bad = validateSamples(robot, task, traj, cart_step);
        if (bad > 0) {
          RCLCPP_ERROR(
            log_, "%s / %s: %d refined sample(s) are IN COLLISION", robot.name.c_str(),
            task.id.c_str(), bad);
          all_ok = false;
        }
        worst_step = std::max(worst_step, cart_step);
        RCLCPP_INFO(
          log_, "%s / %s: K=%zu slots (%.2f s)", robot.name.c_str(), task.id.c_str(),
          traj.num_samples, traj.num_samples * spec_.disc.delta_t);

        artifact.push_back(toJson(robot, task, traj));
        state = end;
      }
      RCLCPP_INFO(
        log_, "%s: %zu of %zu transits went direct", robot.name.c_str(), chained,
        seq.size() - 1);
    }

    RCLCPP_INFO(
      log_, "discretisation: max link travel %.4f m per slot", worst_step);
    if (worst_step > spec_.disc.max_cartesian_step_warn) {
      RCLCPP_WARN(log_, "  ^ exceeds the %.4f m threshold", spec_.disc.max_cartesian_step_warn);
    }

    writeArtifact(out_path, artifact);
    RCLCPP_INFO(
      log_, "%s: %zu refined trajectories written to %s", all_ok ? "OK" : "INCOMPLETE",
      artifact.size(), out_path.c_str());
    return all_ok;
  }

  /// Plan the connecting motions a refinement asks for, and nothing else.
  ///
  /// `spec_file` lists one entry per splice: which robot, which task's environment to plan
  /// in, and the two configurations to join. Those configurations are chosen upstream
  /// (`refine_yield.py`) as poses that are clear of everything the other robot ever does,
  /// which is what makes the resulting shortcut safe to drop into the schedule -- but
  /// choosing them needs `mu`, and planning between them needs MoveIt, and the two live in
  /// different processes. Hence this narrow entry point: configurations in, trajectory out.
  bool runTransits(const std::string & out_path, const std::string & spec_file)
  {
    YAML::Node spec = YAML::LoadFile(spec_file);
    std::vector<std::string> artifact;
    // A rung that will not plan is an ordinary outcome, not a failure: the caller proposes
    // a ladder of shortcuts from boldest to safest precisely because the bold ones may not
    // exist, and records `planned: false` so the splice stage moves down the ladder. Only
    // a malformed spec or an unwritable output is an error here.
    std::size_t planned = 0;

    for (const auto & entry : spec["transits"]) {
      const std::string id = entry["id"].as<std::string>();
      const std::string robot_name = entry["robot"].as<std::string>();
      const std::string task_id = entry["task"].as<std::string>();
      const RobotCfg & robot = robotByName(robot_name);
      const TaskDef & task = taskById(task_id);
      RCLCPP_INFO(log_, "=== transit %s ===", id.c_str());

      auto scene = sceneFor(robot, task);
      if (task.kind == TaskKind::PickPlace) {
        setObjectState(scene, robot, task, ObjectState::AtSpawn);
      }
      const auto * jmg = model_->getJointModelGroup(robot.planning_group);

      moveit::core::RobotState from(homeState()), to(homeState());
      from.setJointGroupPositions(jmg, entry["from"].as<std::vector<double>>());
      to.setJointGroupPositions(jmg, entry["to"].as<std::vector<double>>());
      from.update();
      to.update();

      Segment seg;
      if (!planJoint(scene, robot, from, to, Phase::ToPick, seg)) {
        RCLCPP_INFO(log_, "  no plan for this rung");
        artifact.push_back(transitJson(id, nullptr));
        continue;
      }
      auto traj = mrct::resampleUniform({seg}, spec_.disc.delta_t);
      double cart_step = 0.0;
      const int bad = validateSamples(robot, task, traj, cart_step);
      if (bad > 0) {
        RCLCPP_INFO(log_, "  %d sample(s) of this rung are in collision", bad);
        artifact.push_back(transitJson(id, nullptr));
        continue;
      }
      RCLCPP_INFO(log_, "  %zu slots (%.2f s)", traj.num_samples,
                  traj.num_samples * spec_.disc.delta_t);
      artifact.push_back(transitJson(id, &traj));
      ++planned;
    }

    std::ofstream f(out_path);
    if (!f) {throw std::runtime_error("cannot write " + out_path);}
    f.precision(17);
    f << "{\n  \"delta_t\": " << spec_.disc.delta_t << ",\n  \"transits\": [\n";
    for (std::size_t i = 0; i < artifact.size(); ++i) {
      f << artifact[i] << (i + 1 < artifact.size() ? ",\n" : "\n");
    }
    f << "  ]\n}\n";
    RCLCPP_INFO(log_, "OK: %zu of %zu candidate transits planned -> %s", planned,
                artifact.size(), out_path.c_str());
    return true;
  }

private:
  std::string transitJson(const std::string & id, const mrct::ResampledTrajectory * traj)
  {
    std::ostringstream o;
    o.precision(17);
    o << "    {\"id\": \"" << id << "\", ";
    if (traj == nullptr) {
      o << "\"planned\": false, \"positions\": []}";
      return o.str();
    }
    o << "\"planned\": true, \"positions\": [";
    for (std::size_t k = 0; k < traj->num_samples; ++k) {
      o << (k ? ", " : "") << "[";
      for (std::size_t j = 0; j < traj->num_joints; ++j) {
        o << (j ? ", " : "") << traj->sample(k)[j];
      }
      o << "]";
    }
    o << "]}";
    return o.str();
  }

  const RobotCfg & robotByName(const std::string & name) const
  {
    for (const auto & r : spec_.robots) {
      if (r.name == name) {return r;}
    }
    throw std::runtime_error("no such robot in the scene: " + name);
  }

  /// One line per mesh-backed object or fixture, so the log says what geometry the
  /// planner is working with (and a mistyped `mesh:` is visible without a debugger).
  void logMesh(const char * kind, const std::string & id, const std::optional<MeshRef> & mesh)
  {
    if (!mesh) {return;}
    const auto & m = meshMsg(*mesh);
    RCLCPP_INFO(
      log_, "%s '%s': mesh %s (scale %g), %zu vertices, %zu triangles", kind, id.c_str(),
      mesh->file.c_str(), mesh->scale, m.vertices.size(), m.triangles.size());
  }

  // ---- schedule ------------------------------------------------------------ #

  /// Per-robot task order from the solver's schedule. JSON is valid YAML, so the
  /// solution file parses with the loader already linked in -- no new dependency.
  std::map<std::string, std::vector<std::string>> readSchedule(const std::string & path)
  {
    YAML::Node sol = YAML::LoadFile(path);
    std::vector<std::pair<int, std::pair<std::string, std::string>>> rows;
    for (const auto & kv : sol["assignments"]) {
      rows.push_back(
        {kv.second["start_slot"].as<int>(),
         {kv.second["robot"].as<std::string>(), kv.first.as<std::string>()}});
    }
    std::sort(rows.begin(), rows.end());
    std::map<std::string, std::vector<std::string>> out;
    for (const auto & r : spec_.robots) {out[r.name] = {};}
    for (const auto & [slot, rt] : rows) {
      if (!out.count(rt.first)) {
        throw std::runtime_error("schedule assigns a task to unknown robot " + rt.first);
      }
      out[rt.first].push_back(rt.second);
    }
    return out;
  }

  const TaskDef & taskById(const std::string & id) const
  {
    for (const auto & t : spec_.tasks) {
      if (t.id == id) {return t;}
    }
    throw std::runtime_error("the schedule names a task the scene does not define: " + id);
  }

  /// Every robot parked at home -- the state both the scene builder and ADR-0004 assume.
  moveit::core::RobotState homeState() const
  {
    moveit::core::RobotState s(model_);
    s.setToDefaultValues();
    for (const auto & r : spec_.robots) {
      for (const auto & [joint, value] : r.home) {s.setJointPositions(joint, &value);}
    }
    s.update();
    return s;
  }

  // ---- scene construction ------------------------------------------------- #

  /// The environment task `i` is planned against (ADR-0003), with the OTHER robot
  /// parked at its home so that home clearance holds by construction.
  planning_scene::PlanningScenePtr sceneFor(const RobotCfg & robot, const TaskDef & task)
  {
    const auto t_build = std::chrono::steady_clock::now();
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

    // WHICH objects get their real mesh (ADR-0003, mesh addendum). Fixtures do: they are
    // the geometry the arm has to work around, they are few, and a fixture's shape is the
    // reason to have a mesh at all. The TASK'S OWN object does too (see `setObjectState`):
    // it is the thing being carried, and it is what touches the fixture. Everything else
    // here -- the `spawn__<object>` and `place__<slot>` stand-ins below -- stays a
    // bounding box. `size` is a true superset of the mesh (`parseMesh` enforces it), so the
    // box is CONSERVATIVE: the planner may keep a little further from another part than it
    // strictly must, and can never come closer than the real one allows. It also keeps the
    // world at a handful of triangles per stand-in instead of one BVH per (object, slot)
    // pair, and `sceneFor` runs many times per trajectory.
    std::vector<moveit_msgs::msg::CollisionObject> world;
    for (const auto & f : spec_.fixtures) {
      world.push_back(makeObject(f.id, f.size, f.mesh, f.pose, spec_.base_frame));
    }

    // Precedence-pruned union over the POSSIBLE WORLDS in which `task` runs
    // (ADR-0003 and its 2026-09-19 addendum). With interchangeable slots one
    // object is a candidate of several slots and it is not known which object
    // fills which, so the world is not "a task's place / a task's spawn" but two
    // independent questions:
    //
    //   * WHERE COULD A SLOT'S OBJECT BE?   One box per SLOT, at the slot's place
    //     pose. Occupied unless the slot certainly runs AFTER `task` -- i.e. it is
    //     before `task` or unordered with it. Never for `task`'s own slot.
    //   * WHERE COULD AN OBJECT BE?         One box per OBJECT, at its spawn pose.
    //     Occupied ALWAYS -- whether it has been consumed before `task` is not
    //     known -- except for an object that is the only candidate of a slot that
    //     certainly precedes `task`: that one is certainly already at its place.
    //
    // For a scene without `slots:` every task is its own single-candidate slot and
    // this is exactly the previous per-task rule (place if it precedes, spawn if it
    // follows, both if unordered). It is NOT the same for a slot with several
    // candidates: the old rule keyed both boxes on the candidate TASK and added a
    // spawn only via the tasks that come AFTER `task`, so the LAST slot of a class
    // never saw a single stock object (nothing comes after it), and a slot's own
    // candidates -- skipped whole as siblings -- never saw their spawns either.
    // The planner then threaded fingers through cubes that stand on the table.
    //
    // Object `task.object_id` itself is excluded: it is handled by the phase
    // machinery (spawn -> attached -> place), not by the static world.

    // (a) places, one box per slot.
    for (const auto & [sid, idx] : slots_.members) {
      if (sid == task.slot_id) {continue;}
      // Certainly after `task`: still empty, nothing to avoid.
      const auto sb = slots_.certainly_before.find(sid);
      if (sb != slots_.certainly_before.end() && sb->second.count(task.slot_id)) {continue;}
      // Who could be sitting on that place. A weld moves no geometry; a candidate
      // moving `task`'s own object cannot be there in a world where `task` runs
      // (an object is consumed once), which is what the old same-object skip said.
      // The box is the componentwise maximum over the remaining candidates -- the
      // largest of them when they nest, which is the usual case -- so it is sound
      // whichever one wins.
      bool any = false;
      std::array<double, 3> box{0.0, 0.0, 0.0};
      geometry_msgs::msg::Pose place;
      // The one mesh every remaining candidate shares, if there is one (`placed_meshes`).
      std::optional<MeshRef> shared_mesh;
      bool one_mesh = true;
      for (const auto i : idx) {
        const TaskDef & cand = spec_.tasks[i];
        if (cand.kind == TaskKind::Weld) {continue;}
        if (!task.object_id.empty() && cand.object_id == task.object_id) {continue;}
        const ObjectDef & obj = spec_.object(cand.object_id);
        for (int k = 0; k < 3; ++k) {box[k] = std::max(box[k], obj.size[k]);}
        if (!obj.mesh) {
          one_mesh = false;
        } else if (!shared_mesh) {
          if (any) {one_mesh = false;}       // an earlier candidate had no mesh
          shared_mesh = obj.mesh;
        } else if (shared_mesh->file != obj.mesh->file || shared_mesh->scale != obj.mesh->scale) {
          one_mesh = false;
        }
        place = cand.place;    // one pose per slot, by construction of the expansion
        any = true;
      }
      if (!any) {continue;}
      // A box by default: the slot may be won by any of several candidates, so in general no
      // single mesh describes it (and a bbox is what the componentwise maximum is). With
      // `placed_meshes`, a slot whose every candidate carries the SAME mesh is that mesh: the
      // mesh origin is its bbox centre, which is where `place` puts it. `box` is then that
      // mesh's `size` too, unless the candidates' sizes differ -- which the same file and
      // scale forbid, since `parseMesh` pins `size` to the mesh's bounding box.
      if (spec_.planning.placed_meshes && one_mesh && shared_mesh) {
        world.push_back(makeObject("place__" + sid, box, shared_mesh, place, spec_.base_frame));
      } else {
        world.push_back(makeBox("place__" + sid, box, place, spec_.base_frame));
      }
    }

    // (b) spawns, one box per object, in the scene's own object order.
    std::set<std::string> movable;
    for (const auto & other : spec_.tasks) {
      if (other.kind == TaskKind::PickPlace) {movable.insert(other.object_id);}
    }
    std::set<std::string> already_moved;
    if (const auto sb = slots_.certainly_before.find(task.slot_id);
      sb != slots_.certainly_before.end())
    {
      for (const auto & sid : sb->second) {
        if (const auto so = slots_.sole_object.find(sid); so != slots_.sole_object.end()) {
          already_moved.insert(so->second);
        }
      }
    }
    for (const auto & obj : spec_.objects) {
      if (!movable.count(obj.id) || obj.id == task.object_id || already_moved.count(obj.id)) {
        continue;
      }
      // Bounding box of another object's mesh: a superset, see the note on fixtures above.
      world.push_back(makeBox("spawn__" + obj.id, obj.size, obj.spawn, spec_.base_frame));
    }
    for (auto & c : world) {scene->processCollisionObjectMsg(c);}
    scene_seconds_ += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_build).count();
    ++scene_builds_;
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
      aco.object = makeObject(obj.id, obj.size, obj.mesh, obj.spawn, spec_.base_frame);
      // The pose is ignored once attached (it rides the link), but the frame must
      // be one MoveIt can resolve; the grasp transform is what fixes it to the
      // gripper, and that is implied by the state at which we attach.
      scene->processAttachedCollisionObjectMsg(aco);
    } else {
      const auto & pose = (st == ObjectState::AtSpawn) ? obj.spawn : task.place;
      auto add = makeObject(obj.id, obj.size, obj.mesh, pose, spec_.base_frame);
      scene->processCollisionObjectMsg(add);
    }
  }

  // ---- planning ----------------------------------------------------------- #

  /// `positions`' joint values, on a copy of the scene's CURRENT state.
  ///
  /// This is what makes a carried object exist for a collision check. MoveIt keeps
  /// an attached body on a RobotState, and `setObjectState` attaches the object to
  /// the scene's own current state -- but `PlanningScene::isStateColliding(state)`
  /// checks the attached bodies of the state it is GIVEN. Every check in this file
  /// is handed a separately built state (the Cartesian path, the IK candidates, the
  /// OMPL start, the resampled samples), and none of those carried the attachment.
  /// The object was therefore invisible to every check while carried: measured on
  /// `weldprobe` (2026-09-13), a bracket driven 30 mm into its fixture planned and
  /// validated clean. Rebuilding each state from the scene's current state brings
  /// the attachment along, and positions are the only thing taken from `positions`.
  ///
  /// The OMPL start goes through it as well, so the carry commute's start state
  /// carries the attachment explicitly instead of relying on how MoveIt merges a
  /// start-state message into its copy of the scene (not verified either way).
  static moveit::core::RobotState stateInScene(
    const planning_scene::PlanningScenePtr & scene, const moveit::core::RobotState & positions)
  {
    moveit::core::RobotState s(scene->getCurrentState());
    s.setVariablePositions(positions.getVariablePositions());
    s.update();
    return s;
  }

  /// The scene's ACM, plus ONE allowance: `object` may touch the support surface (and the
  /// fixtures listed in its `supports:`, under the same scope).
  ///
  /// An object resting on the table touches it at exactly zero clearance, and once
  /// the object is attached (see `stateInScene`) FCL reports that contact as a
  /// collision -- on every GripClose at a spawn and every place descent onto the
  /// table, in every scene. MoveIt's own pick/place answers this with
  /// `support_surface_name`, scoped to the approach, grasp and retreat, and so does
  /// this: callers pass an object id only for the pick descent, GripClose and lift,
  /// and the place descent, GripOpen and retreat. The transfer between them, every
  /// other obstacle (fixtures, tray walls, other objects), and weld tasks get the
  /// scene's ACM untouched. The allowance lives on a COPY used for one check, never
  /// on the scene, so it cannot leak into a later check.
  collision_detection::AllowedCollisionMatrix supportAcm(
    const planning_scene::PlanningScenePtr & scene, const std::string & object) const
  {
    collision_detection::AllowedCollisionMatrix acm = scene->getAllowedCollisionMatrix();
    if (!object.empty()) {
      acm.setEntry(object, spec_.support_surface, true);
      for (const auto & f : spec_.object(object).supports) {acm.setEntry(object, f, true);}
    }
    return acm;
  }

  /// `isStateColliding`, against an explicit ACM (same request otherwise).
  static bool collides(
    const planning_scene::PlanningScenePtr & scene, const moveit::core::RobotState & state,
    const std::string & group, const collision_detection::AllowedCollisionMatrix & acm)
  {
    collision_detection::CollisionRequest req;
    req.group_name = group;
    collision_detection::CollisionResult res;
    scene->checkCollision(req, res, state, acm);
    return res.collision;
  }

  /// IK to a world EE pose, rejecting solutions that collide with the scene.
  ///
  /// No support allowance here, deliberately: every IK target in this file is a
  /// RAISED pose (pre-grasp, pre-place, pre-start), 12 cm clear of any support.
  /// The resting poses are reached by Cartesian descent, which is where the
  /// allowance applies; granting it to IK would only widen it.
  bool ikTo(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const geometry_msgs::msg::Pose & target, const moveit::core::RobotState & seed,
    moveit::core::RobotState & out)
  {
    // Wall time of every IK call, collision checks of the candidates included, for the
    // per-trajectory `timing.ik_s` in the artifact (see `run`).
    struct IkClock
    {
      double & acc;
      std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
      ~IkClock()
      {
        acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      }
    } ik_clock{ik_seconds_};
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);
    // Candidates are built from the seed's joint values on the scene's current
    // state, so a carried object is part of what "in collision" means.
    const moveit::core::RobotState seeded = stateInScene(scene, seed);
    out = seeded;

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

    // A parallel-jaw gripper is symmetric about its own axis: rotating the tool by
    // pi swaps the two fingers and leaves the grasp physically identical. Solving
    // only for the yaw the task file happens to name therefore costs one arm half a
    // turn of wrist_3 on every pick and every place, purely because the two bases sit
    // 180 degrees apart -- measured at 0.88 pi of wrist travel per task against the
    // other arm's 0.13 pi, which is 13 % of its cycle time and enough to bias every
    // allocation the scheduler makes.
    //
    // So solve both equivalent orientations and keep whichever lands nearer the seed.
    // The object is a cube and is likewise symmetric under the flip, so the attached
    // representation is unaffected.
    const Eigen::Isometry3d flipped =
      goal * Eigen::AngleAxisd(M_PI, robot.tool_approach_axis);

    moveit::core::RobotState as_named(seeded), as_flipped(seeded);
    const bool ok_named = as_named.setFromIK(jmg, goal, robot.ee_link, 0.5, valid);
    const bool ok_flipped = as_flipped.setFromIK(jmg, flipped, robot.ee_link, 0.5, valid);
    if (!ok_named && !ok_flipped) {return false;}
    if (ok_named && ok_flipped) {
      out = (seed.distance(as_named, jmg) <= seed.distance(as_flipped, jmg))
        ? as_named : as_flipped;
    } else {
      out = ok_named ? as_named : as_flipped;
    }
    return true;
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
    moveit::core::robotStateToRobotStateMsg(stateInScene(scene, start), req.start_state);
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

  /// Straight-line EE motion by a world-frame translation `delta`.
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
  ///
  /// Pinning the rail is about the 7-DOF redundancy, NOT about the direction of
  /// travel, so a lateral step is as well-conditioned as a vertical one -- and a
  /// 25 mm process traverse is four times shorter than the 120 mm approach that
  /// already works on every scene. The rail position is chosen by the 7-DOF IK at
  /// the pose this step departs from.
  ///
  /// `speed > 0` re-times the result at that constant EE speed instead of letting
  /// TOTG run it as fast as the joints allow; see `timeParameteriseAtSpeed`.
  bool planCartesianStep(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const moveit::core::RobotState & start, const Eigen::Vector3d & delta, Phase phase,
    Segment & out, moveit::core::RobotState & end_state,
    const std::string & support_object = std::string{}, double speed = 0.0)
  {
    const auto * arm = model_->getJointModelGroup(robot.arm_group);
    const auto * full = model_->getJointModelGroup(robot.planning_group);
    const auto * link = model_->getLinkModel(robot.ee_link);

    auto state = std::make_shared<moveit::core::RobotState>(stateInScene(scene, start));
    std::vector<std::shared_ptr<moveit::core::RobotState>> path;

    // `support_object` non-empty: this is a pick/place leg and that object may
    // touch the support surface (see `supportAcm`). Empty: the scene's ACM as is.
    const auto acm = supportAcm(scene, support_object);
    auto valid = [&](moveit::core::RobotState * s, const moveit::core::JointModelGroup * g,
        const double * values) {
        s->setJointGroupPositions(g, values);
        s->update();
        // Check the WHOLE robot, not just the arm: the group being interpolated is
        // the arm, but a collision anywhere (rail carriage, the other robot, the
        // table) still invalidates the state.
        return !collides(scene, *s, robot.planning_group, acm);
      };

    // NOTE the return value. The Eigen::Vector3d (translation) overload returns the
    // DISTANCE ACHIEVED IN METRES -- only the Isometry3d (pose-target) overload
    // returns a 0..1 fraction. Comparing this against a fraction silently rejects
    // every successful path (a fully-achieved 12 cm descent "fails" a `< 0.99`
    // test), which looks exactly like a planner problem and is not one.
    const double achieved_m = moveit::core::CartesianInterpolator::computeCartesianPath(
      state.get(), arm, path, link, delta, /*global_reference_frame=*/true,
      moveit::core::MaxEEFStep(spec_.planning.cartesian_step),
      moveit::core::JumpThreshold::disabled(), valid);

    const double wanted_m = delta.norm();
    if (achieved_m < 0.99 * wanted_m || path.size() < 2) {
      RCLCPP_WARN(
        log_, "Cartesian %s achieved %.4f m of %.4f m (%.0f%%)", motionName(delta), achieved_m,
        wanted_m, 100.0 * achieved_m / wanted_m);
      diagnoseCartesianFailure(scene, robot, start, delta, acm);
      return false;
    }

    // CartesianInterpolator returns a geometric path with no timing. Time it on the
    // 7-DOF group, so the segment carries the same joint set as every other segment.
    robot_trajectory::RobotTrajectory rt(model_, full);
    for (const auto & s : path) {rt.addSuffixWayPoint(*s, 0.0);}
    if (speed > 0.0) {
      if (!timeParameteriseAtSpeed(rt, robot, speed)) {return false;}
    } else if (!timeParameterise(rt)) {
      return false;
    }

    end_state = *path.back();
    out.support_contact = !support_object.empty();
    return toSegment(rt, phase, out);
  }

  /// The vertical case, which is every approach and every retreat. A thin wrapper
  /// so the four call sites that predate the generalisation read as they always did.
  bool planCartesianZ(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const moveit::core::RobotState & start, double dz, Phase phase, Segment & out,
    moveit::core::RobotState & end_state, const std::string & support_object = std::string{})
  {
    return planCartesianStep(
      scene, robot, start, Eigen::Vector3d(0, 0, dz), phase, out, end_state, support_object);
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
    const moveit::core::RobotState & start, const Eigen::Vector3d & delta,
    const collision_detection::AllowedCollisionMatrix & acm)
  {
    const auto * arm = model_->getJointModelGroup(robot.arm_group);
    const auto * link = model_->getLinkModel(robot.ee_link);

    auto state = std::make_shared<moveit::core::RobotState>(stateInScene(scene, start));
    std::vector<std::shared_ptr<moveit::core::RobotState>> path;
    const double geometric_m = moveit::core::CartesianInterpolator::computeCartesianPath(
      state.get(), arm, path, link, delta, true,
      moveit::core::MaxEEFStep(spec_.planning.cartesian_step),
      moveit::core::JumpThreshold::disabled());   // no validity callback: IK only

    const double wanted_m = delta.norm();
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
      scene->checkCollision(req, res, *path[k], acm);
      if (!res.collision) {continue;}
      for (const auto & [pair, contacts] : res.contacts) {
        RCLCPP_ERROR(
          log_, "    at %.0f%% of the %s: '%s' touches '%s'",
          100.0 * static_cast<double>(k) / static_cast<double>(path.size() - 1),
          motionName(delta), pair.first.c_str(), pair.second.c_str());
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

  /// Re-time a geometric Cartesian path so the tool travels the seam at a
  /// CONSTANT, prescribed speed.
  ///
  /// TOTG answers a different question -- "how fast may this path be run without
  /// violating a joint limit?" -- and for a process pass that is the wrong
  /// question. It would run a 25 mm seam at roughly 250 mm/s, five times a GMAW
  /// travel speed. A traverse that ignores its own speed is not a weld: the speed
  /// is a specification of the PROCESS, and the robot has to obey it, not the
  /// other way round.
  ///
  /// So the path is timed by ARC LENGTH: waypoint k is reached at (the Cartesian
  /// distance the EE has travelled up to k) / v. The interpolator emits a
  /// waypoint every `cartesian_step` (5 mm), so the grid is fine and the joint
  /// velocities follow by central difference -- the discrete statement of dq/dt
  /// along the path. The two endpoints use a one-sided difference rather than
  /// zero, so the speed is genuinely constant at every knot of the seam; the arc
  /// is struck and cut during the dwells either side, where the arm is frozen.
  ///
  /// The implied joint velocities are checked against the model's limits and the
  /// re-timing FAILS if any is exceeded. At 50 mm/s over 25 mm none ever will,
  /// which is exactly why the check is cheap enough to keep -- it is what makes
  /// "the seam is run at the process speed" a claim rather than an assumption.
  bool timeParameteriseAtSpeed(
    robot_trajectory::RobotTrajectory & rt, const RobotCfg & robot, double speed)
  {
    if (speed <= 0.0) {
      RCLCPP_ERROR(log_, "process speed must be positive (got %.4f m/s)", speed);
      return false;
    }
    const std::size_t n = rt.getWayPointCount();
    if (n < 2) {
      RCLCPP_WARN(log_, "a process path needs at least two waypoints");
      return false;
    }
    const auto * link = model_->getLinkModel(robot.ee_link);
    const auto * jmg = rt.getGroup();

    // ---- arc length -> time ------------------------------------------------ #
    std::vector<double> t(n, 0.0);
    double arc = 0.0;
    for (std::size_t k = 1; k < n; ++k) {
      const Eigen::Vector3d a = rt.getWayPoint(k - 1).getGlobalLinkTransform(link).translation();
      const Eigen::Vector3d b = rt.getWayPoint(k).getGlobalLinkTransform(link).translation();
      arc += (b - a).norm();
      t[k] = arc / speed;
    }
    if (!(arc > 0.0)) {
      RCLCPP_ERROR(log_, "the process path has zero Cartesian length; nothing to traverse");
      return false;
    }

    // ---- joint velocities by central difference ---------------------------- #
    std::vector<std::vector<double>> q(n);
    for (std::size_t k = 0; k < n; ++k) {rt.getWayPoint(k).copyJointGroupPositions(jmg, q[k]);}
    const std::size_t dof = q.front().size();

    std::vector<std::vector<double>> v(n, std::vector<double>(dof, 0.0));
    for (std::size_t k = 0; k < n; ++k) {
      const std::size_t lo = (k == 0) ? 0 : k - 1;
      const std::size_t hi = (k + 1 == n) ? k : k + 1;
      const double span = t[hi] - t[lo];
      if (span > 1e-12) {
        for (std::size_t j = 0; j < dof; ++j) {v[k][j] = (q[hi][j] - q[lo][j]) / span;}
      }
    }

    // ---- honesty check: does the process speed fit inside the joint limits? -- #
    const auto & joints = jmg->getActiveJointModels();
    if (joints.size() != dof) {
      RCLCPP_WARN(
        log_, "cannot check process joint velocities: %zu active joints for %zu variables",
        joints.size(), dof);
    } else {
      double worst_ratio = 0.0;
      std::size_t worst_joint = 0;
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t j = 0; j < dof; ++j) {
          const auto & b = joints[j]->getVariableBounds().front();
          if (!b.velocity_bounded_ || b.max_velocity_ <= 0.0) {continue;}
          const double ratio = std::abs(v[k][j]) / b.max_velocity_;
          if (ratio > worst_ratio) {
            worst_ratio = ratio;
            worst_joint = j;
          }
        }
      }
      if (worst_ratio > 1.0) {
        RCLCPP_ERROR(
          log_,
          "a %.3f m/s traverse needs %.0f%% of joint '%s' velocity limit -- the process speed "
          "does not fit the kinematics on this seam. Slow the process down, or move the seam.",
          speed, 100.0 * worst_ratio, joints[worst_joint]->getName().c_str());
        return false;
      }
      RCLCPP_INFO(
        log_, "  process traverse: %.4f m at %.3f m/s = %.2f s, worst joint %.0f%% of limit (%s)",
        arc, speed, t.back(), 100.0 * worst_ratio, joints[worst_joint]->getName().c_str());
    }

    // ---- commit ------------------------------------------------------------ #
    for (std::size_t k = 0; k < n; ++k) {
      rt.getWayPointPtr(k)->setJointGroupVelocities(jmg, v[k]);
      rt.getWayPointPtr(k)->update();
      rt.setWayPointDurationFromPrevious(k, k == 0 ? 0.0 : t[k] - t[k - 1]);
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

  /// One task's pick -> transport -> place cycle, starting wherever the arm already is.
  ///
  /// `from` is the configuration the arm begins at and `to_home` says whether it must
  /// finish parked at home. Both are parameters rather than constants because that is the
  /// ONLY difference between a stand-alone task and a link in a chain (ADR-0008): a chained
  /// task departs from the previous task's retreat pose instead of from home, and an
  /// intermediate one stops at its own retreat pose instead of flying back. Everything
  /// between the two is identical, which is the point -- the refined plan is the same
  /// motion planning problem with different endpoints, not a different pipeline.
  ///
  /// `end` receives the configuration the arm is left in.
  ///
  /// A task is one of two kinds and this is where they part company; everything
  /// on either side of it -- the chain machinery, the resampling, the validation,
  /// the artifact -- is written against `TaskDef` and never asks which.
  bool planTaskFrom(
    const RobotCfg & robot, const TaskDef & task, const moveit::core::RobotState & from,
    bool to_home, std::vector<Segment> & segs, moveit::core::RobotState & end)
  {
    switch (task.kind) {
      case TaskKind::PickPlace:
        return planPickPlaceFrom(robot, task, from, to_home, segs, end);
      case TaskKind::Weld:
        return planWeldFrom(robot, task, from, to_home, segs, end);
    }
    return false;
  }

  /// A process pass: take the tool to the seam start, strike, run the seam at the
  /// process speed, cut, retreat. Nothing is grasped, so there is no gripper
  /// actuation, no attached body and no object anywhere in the cycle.
  ///
  ///     home -> pre-start (seam start, raised)        Phase::ToPick
  ///          -> Cartesian descent onto the seam       Phase::ToPick
  ///          -> arc-strike dwell                      Phase::ProcessOn
  ///          -> traverse seam start -> seam end       Phase::Processing
  ///          -> arc-out dwell                         Phase::ProcessOff
  ///          -> Cartesian retreat off the seam        Phase::ToHome
  ///          -> home                                  Phase::ToHome
  ///
  /// The commutes REUSE `ToPick` and `ToHome` rather than getting phases of their
  /// own, and that is load-bearing: `refine_yield.py` finds a courtesy parking
  /// pose by scanning the outgoing task for `ToHome` samples and the incoming one
  /// for `ToPick` samples. Reusing the two means the ADR-0008 refinement works on
  /// a weld commute without knowing a weld exists; inventing new phases would
  /// have made every weld commute silently unrefinable.
  bool planWeldFrom(
    const RobotCfg & robot, const TaskDef & task, const moveit::core::RobotState & from,
    bool to_home, std::vector<Segment> & segs, moveit::core::RobotState & end)
  {
    auto scene = sceneFor(robot, task);
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);

    // Same composition rule as a grasp: the tool pose is given relative to the
    // point being worked on, and composing it with the world pose of that point
    // gives the world EE pose.
    const auto start_ee = compose(task.seam_start, task.tool);
    const auto end_ee = compose(task.seam_end, task.tool);
    const double a = task.approach;

    moveit::core::RobotState home(scene->getCurrentState());   // both robots parked
    moveit::core::RobotState start(from);
    start.update();

    if (scene->isStateColliding(start, robot.planning_group)) {
      RCLCPP_WARN(
        log_, "the start configuration collides in this task's environment -- the previous "
        "task left the arm somewhere %s cannot legally begin from", task.id.c_str());
      return false;
    }

    // --- fly out to the seam start, raised ----------------------------------- #
    moveit::core::RobotState pre_start(start);
    if (!ikTo(scene, robot, raised(start_ee, a), start, pre_start)) {
      RCLCPP_WARN(log_, "no collision-free IK for the pre-start pose");
      return false;
    }

    Segment s1;
    if (!planJoint(scene, robot, start, pre_start, Phase::ToPick, s1)) {return false;}
    segs.push_back(s1);

    Segment s2;
    moveit::core::RobotState at_start(pre_start);
    if (!planCartesianZ(scene, robot, pre_start, -a, Phase::ToPick, s2, at_start)) {return false;}
    segs.push_back(s2);

    // --- strike the arc: arm frozen, slots consumed -------------------------- #
    std::vector<double> start_q;
    at_start.copyJointGroupPositions(jmg, start_q);
    segs.push_back(
      mrct::makeDwell(
        start_q, Phase::ProcessOn, spec_.disc.process_dwell_slots, spec_.disc.delta_t));

    // --- run the seam, at the process speed ---------------------------------- #
    const Eigen::Vector3d seam(
      end_ee.position.x - start_ee.position.x,
      end_ee.position.y - start_ee.position.y,
      end_ee.position.z - start_ee.position.z);

    Segment s3;
    moveit::core::RobotState at_end(at_start);
    if (!planCartesianStep(
        scene, robot, at_start, seam, Phase::Processing, s3, at_end, std::string{}, task.speed))
    {
      return false;
    }
    segs.push_back(s3);

    // --- cut the arc --------------------------------------------------------- #
    std::vector<double> end_q;
    at_end.copyJointGroupPositions(jmg, end_q);
    segs.push_back(
      mrct::makeDwell(
        end_q, Phase::ProcessOff, spec_.disc.process_dwell_slots, spec_.disc.delta_t));

    // --- retreat and (maybe) go home ----------------------------------------- #
    scene->getCurrentStateNonConst() = at_end;

    Segment s4;
    moveit::core::RobotState retreated(at_end);
    if (!planCartesianZ(scene, robot, at_end, a, Phase::ToHome, s4, retreated)) {return false;}
    segs.push_back(s4);

    if (to_home) {
      Segment s5;
      if (!planJoint(scene, robot, retreated, home, Phase::ToHome, s5)) {return false;}
      segs.push_back(s5);
      end = home;
    } else {
      end = retreated;
    }
    return true;
  }

  /// Solve a raised pre-pose, plan the free-space approach to it, and Cartesian-
  /// descend `dz` onto the resting pose -- retried a bounded number of times.
  ///
  /// WHY: `ikTo` and the Cartesian descent it feeds are solved independently.
  /// `ikTo` picks whichever symmetric solution lands nearest the seed, with no
  /// foresight into whether the resulting REDUNDANT-axis value (the rail on the
  /// UR cell; the shoulder-yaw joint on a fully-revolute redundant arm like
  /// TIAGo's) will also support the FULL descent that follows -- a valid
  /// pre-grasp/pre-place pose can still leave the descent geometrically stuck a
  /// few centimetres in (verified directly, tiago_cell, 2026-09-17: "Cartesian
  /// descent achieved 70-95%" off an already-valid, collision-free pre-pose).
  /// Retrying the whole trio gets a genuinely different candidate for free on a
  /// solver like TRAC-IK, which races an SQP-from-seed thread against several
  /// randomized-restart threads internally -- a second call from the IDENTICAL
  /// seed routinely returns a different valid configuration. Byte-identical to
  /// the old single-attempt code on the FIRST try, so a chain where attempt 1
  /// already succeeds (as on the UR cell) sees no behaviour change at all.
  ///
  /// Each retry re-runs the free-space OMPL plan too (up to `planning_time`
  /// again), not just the cheap descent -- unavoidable, since a different
  /// pre-pose candidate needs its own path to it. Costs wall-clock time only on
  /// the failure path, where the task was going to fail outright anyway.
  bool planApproachAndDescend(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const geometry_msgs::msg::Pose & raised_target, const moveit::core::RobotState & from,
    double dz, Phase approach_phase, Phase descent_phase, const std::string & support_object,
    const char * pose_name, std::vector<Segment> & segs, moveit::core::RobotState & end)
  {
    constexpr int kIkRetries = 3;
    for (int attempt = 0; attempt < kIkRetries; ++attempt) {
      moveit::core::RobotState pre(from);
      if (!ikTo(scene, robot, raised_target, from, pre)) {
        if (attempt + 1 == kIkRetries) {
          RCLCPP_WARN(log_, "no collision-free IK for the %s", pose_name);
        }
        continue;
      }

      Segment approach_seg;
      if (!planJoint(scene, robot, from, pre, approach_phase, approach_seg)) {continue;}

      Segment descent_seg;
      moveit::core::RobotState landed(pre);
      if (!planCartesianZ(
          scene, robot, pre, dz, descent_phase, descent_seg, landed, support_object))
      {
        continue;
      }

      segs.push_back(approach_seg);
      segs.push_back(descent_seg);
      end = landed;
      return true;
    }
    return false;
  }

  /// The pick-and-place cycle. See `planTaskFrom` for what `from`/`to_home` mean.
  bool planPickPlaceFrom(
    const RobotCfg & robot, const TaskDef & task, const moveit::core::RobotState & from,
    bool to_home, std::vector<Segment> & segs, moveit::core::RobotState & end)
  {
    auto scene = sceneFor(robot, task);
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);
    const ObjectDef & obj = spec_.object(task.object_id);

    const auto pick_ee = compose(obj.spawn, obj.grasp_in_obj);
    const auto place_ee = compose(task.place, obj.grasp_in_obj);
    const double a = spec_.planning.approach;

    moveit::core::RobotState home(scene->getCurrentState());   // both robots parked

    // The arm starts where we are told; the scene's OTHER robot stays at home, which is
    // what keeps the offline stage schedule-independent (ADR-0003).
    moveit::core::RobotState start(from);
    start.update();

    // --- approach: object i sits at its SPAWN pose --------------------------- #
    setObjectState(scene, robot, task, ObjectState::AtSpawn);

    if (scene->isStateColliding(start, robot.planning_group)) {
      RCLCPP_WARN(
        log_, "the start configuration collides in this task's environment -- the previous "
        "task left the arm somewhere %s cannot legally begin from", task.id.c_str());
      return false;
    }

    // Seeding IK from the actual start rather than from home costs nothing and tends to
    // return the nearer of the redundant solutions, which is exactly what we want here.
    // Pick descent, GripClose and lift may touch the support surface (see supportAcm).
    moveit::core::RobotState grasp(start);
    if (!planApproachAndDescend(
        scene, robot, raised(pick_ee, a), start, -a, Phase::ToPick, Phase::ToPick, obj.id,
        "pre-grasp pose", segs, grasp))
    {
      return false;
    }

    // --- close the gripper: arm frozen, slots consumed ----------------------- #
    std::vector<double> grasp_q;
    grasp.copyJointGroupPositions(jmg, grasp_q);
    segs.push_back(
      mrct::makeDwell(
        grasp_q, Phase::GripClose, spec_.disc.gripper_dwell_slots, spec_.disc.delta_t));
    segs.back().support_contact = true;

    // --- carry: the object rides the gripper --------------------------------- #
    scene->getCurrentStateNonConst() = grasp;
    setObjectState(scene, robot, task, ObjectState::Attached);

    Segment s3;
    moveit::core::RobotState lifted(grasp);
    if (!planCartesianZ(scene, robot, grasp, a, Phase::Carrying, s3, lifted, obj.id)) {
      return false;
    }
    segs.push_back(s3);

    // Place descent, GripOpen and retreat may touch the support surface. The OMPL
    // transfer above may not: it is checked in full.
    moveit::core::RobotState placed(lifted);
    if (!planApproachAndDescend(
        scene, robot, raised(place_ee, a), lifted, -a, Phase::Carrying, Phase::Carrying,
        obj.id, "pre-place pose (object attached)", segs, placed))
    {
      return false;
    }

    // --- open the gripper ---------------------------------------------------- #
    std::vector<double> place_q;
    placed.copyJointGroupPositions(jmg, place_q);
    segs.push_back(
      mrct::makeDwell(
        place_q, Phase::GripOpen, spec_.disc.gripper_dwell_slots, spec_.disc.delta_t));
    segs.back().support_contact = true;

    // --- return: the object now sits at its PLACE pose ----------------------- #
    scene->getCurrentStateNonConst() = placed;
    setObjectState(scene, robot, task, ObjectState::AtPlace);

    Segment s6;
    moveit::core::RobotState retreated(placed);
    if (!planCartesianZ(scene, robot, placed, a, Phase::ToHome, s6, retreated, obj.id)) {
      return false;
    }
    segs.push_back(s6);

    // The retreat is never optional -- it lifts the gripper clear of the box just placed.
    // The flight home afterwards is, and skipping it is the whole of the refinement.
    if (to_home) {
      Segment s7;
      if (!planJoint(scene, robot, retreated, home, Phase::ToHome, s7)) {return false;}
      segs.push_back(s7);
      end = home;
    } else {
      end = retreated;
    }
    return true;
  }

  /// The stand-alone home -> ... -> home cycle for one (robot, task) pair.
  bool planTask(const RobotCfg & robot, const TaskDef & task, mrct::ResampledTrajectory & out)
  {
    auto scene = sceneFor(robot, task);
    moveit::core::RobotState home(scene->getCurrentState()), end(home);
    std::vector<Segment> segs;
    if (!planTaskFrom(robot, task, home, true, segs, end)) {return false;}
    out = mrct::resampleUniform(segs, spec_.disc.delta_t);
    flattenObjectState(task, out);
    return true;
  }

  /// A process task has no object, so `object_state` must not pretend otherwise.
  ///
  /// The commutes reuse ToPick and ToHome (see `planWeldFrom`), and the phase ->
  /// state rule maps those to AtSpawn and AtPlace. Left alone, a weld would
  /// therefore report an object appearing at a spawn pose it does not have and
  /// then at a place pose it does not have either -- which splits the downstream
  /// run-length decomposition into two blocks for no reason, and would have the
  /// collision stage attach geometry that does not exist.
  static void flattenObjectState(const TaskDef & task, mrct::ResampledTrajectory & traj)
  {
    if (task.kind != TaskKind::Weld) {return;}
    std::fill(traj.object_state.begin(), traj.object_state.end(), ObjectState::AtSpawn);
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
  /// True if `link_name` starts with any of `prefixes`.
  static bool matchesAnyPrefix(
    const std::string & link_name, const std::vector<std::string> & prefixes)
  {
    for (const auto & p : prefixes) {
      if (link_name.rfind(p, 0) == 0) {return true;}
    }
    return false;
  }

  int validateSamples(
    const RobotCfg & robot, const TaskDef & task, const mrct::ResampledTrajectory & traj,
    double & max_cartesian_step)
  {
    auto scene = sceneFor(robot, task);
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);
    moveit::core::RobotState state(scene->getCurrentState());

    // Every link belonging to THIS robot, including the gripper (which is not in
    // the planning group but is very much part of the geometry that can collide).
    // `link_prefixes` defaults to {"<name>_"} (see YAML parsing) so this matches
    // today's behaviour on every existing scene unless the field is overridden.
    std::vector<const moveit::core::LinkModel *> links;
    for (const auto * lm : model_->getLinkModels()) {
      if (!lm->getShapes().empty() && matchesAnyPrefix(lm->getName(), robot.link_prefixes)) {
        links.push_back(lm);
      }
    }
    if (links.empty()) {
      throw std::runtime_error(
        "robot '" + robot.name + "' matched zero collision links under its "
        "link_prefixes -- check the scene YAML's link_prefixes against the URDF "
        "link names");
    }

    // A process task owns no object, so there is no spawn -> attached -> place
    // machinery to step the scene through: `sceneFor` already built the world it
    // executes in, and nothing about it changes over the trajectory.
    const bool has_object = (task.kind == TaskKind::PickPlace);

    int bad = 0;
    ObjectState current = ObjectState::AtSpawn;
    if (has_object) {setObjectState(scene, robot, task, current);}

    max_cartesian_step = 0.0;
    std::vector<Eigen::Vector3d> prev;

    for (std::size_t k = 0; k < traj.num_samples; ++k) {
      state.setJointGroupPositions(jmg, traj.sample(k));
      state.update();

      if (has_object && traj.object_state[k] != current) {
        current = traj.object_state[k];
        scene->getCurrentStateNonConst() = state;      // attach at the pose we are AT
        setObjectState(scene, robot, task, current);
        // ...and carry the result back: the attach (or detach) happened on the
        // scene's state, and `state` is what gets checked from here on.
        state = scene->getCurrentState();
      }

      // The support allowance follows the segment the sample was taken from, and
      // only a task that owns an object has one to grant.
      const std::string support_object =
        (has_object && traj.support_contact[k]) ? task.object_id : std::string{};
      const auto acm = supportAcm(scene, support_object);   // a copy; the scene is untouched
      if (collides(scene, state, robot.planning_group, acm)) {
        if (bad == 0) {
          // Name the first contact: "sample k is in collision" says nothing about
          // what to change.
          collision_detection::CollisionRequest req;
          req.contacts = true;
          req.max_contacts = 8;
          req.max_contacts_per_pair = 1;
          collision_detection::CollisionResult res;
          scene->checkCollision(req, res, state, acm);
          for (const auto & [pair, contacts] : res.contacts) {
            RCLCPP_WARN(
              log_, "  sample %zu: '%s' touches '%s'", k, pair.first.c_str(),
              pair.second.c_str());
          }
        }
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

  /// Where one (robot, task) pair's planning time went, summed over every attempt
  /// (`attempts` = 1 means no retry). `ik_s` is the part of `plan_s` spent in `ikTo`.
  /// Written as `timing` by `run` only: the APEX-MR comparison charges the IK of every
  /// candidate but the planning of the chosen trajectories only.
  struct PairTiming
  {
    double ik_s, plan_s, validate_s;
    int attempts;
  };

  std::string toJson(
    const RobotCfg & robot, const TaskDef & task, const mrct::ResampledTrajectory & traj,
    const PairTiming * timing = nullptr)
  {
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);
    std::ostringstream o;
    o.precision(17);

    o << "    {\n";
    o << "      \"robot\": \"" << robot.name << "\",\n";
    o << "      \"task\": \"" << task.id << "\",\n";
    // Which slot this task is a candidate for. Always present and always
    // non-empty: a task with no competitors is its own singleton slot, so this
    // equals the task id on every scene written before `slots:` existed.
    o << "      \"slot\": \"" << task.slot_id << "\",\n";
    // Empty for a process task, and deliberately still present: an explicit empty
    // string is a discriminator every downstream consumer can test, where a
    // missing key would make each of them fail differently.
    o << "      \"object\": \""
      << (task.kind == TaskKind::PickPlace ? task.object_id : std::string{}) << "\",\n";
    o << "      \"num_samples\": " << traj.num_samples << ",\n";
    o << "      \"used_velocities\": " << (traj.used_velocities ? "true" : "false") << ",\n";
    o << "      \"max_joint_step\": " << traj.max_joint_step << ",\n";
    if (timing != nullptr) {
      o << "      \"timing\": {\"ik_s\": " << timing->ik_s << ", \"plan_s\": " << timing->plan_s
        << ", \"validate_s\": " << timing->validate_s << ", \"attempts\": " << timing->attempts
        << "},\n";
    }

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

    // Which tasks this artifact is ABOUT. Normally every task in the scene. In a
    // refined artifact (`chains_` non-empty) it is the tasks the schedule actually
    // runs -- the refinement replans only those, chained, and a task with no
    // trajectory has no candidate robot, which the solver rightly refuses as
    // trivially infeasible. It is a no-op for every scene without slots, where
    // every task is scheduled; with slots, three candidates per slot lose to a
    // fourth and must not follow their winner into the refined problem.
    std::set<std::string> kept;
    if (chains_.empty()) {
      for (const auto & t : spec_.tasks) {kept.insert(t.id);}
    } else {
      for (const auto & [robot, seq] : chains_) {
        (void)robot;
        kept.insert(seq.begin(), seq.end());
      }
    }

    f << "  \"tasks\": [";
    bool first_task = true;
    for (const auto & t : spec_.tasks) {
      if (!kept.count(t.id)) {continue;}
      f << (first_task ? "" : ", ") << "\"" << t.id << "\"";
      first_task = false;
    }
    f << "],\n";

    f << "  \"precedences\": [";
    bool first_prec = true;
    std::vector<std::size_t> prec_idx;
    for (std::size_t p = 0; p < spec_.precedences.size(); ++p) {
      if (!kept.count(spec_.precedences[p].first) ||
        !kept.count(spec_.precedences[p].second))
      {
        continue;
      }
      prec_idx.push_back(p);
      f << (first_prec ? "" : ", ") << "[\"" << spec_.precedences[p].first << "\", \""
        << spec_.precedences[p].second << "\"]";
      first_prec = false;
    }
    f << "],\n";

    // Parallel to `precedences`: how each pair must be enforced. Derived offline
    // from what the tasks ARE, so the seam stays geometry-free and type-free --
    // downstream reads two strings, "pipeline" or "gate", and never learns that
    // one of them exists because a task happens to be a weld.
    f << "  \"precedence_modes\": [";
    for (std::size_t k = 0; k < prec_idx.size(); ++k) {
      f << (k ? ", " : "") << "\"" << spec_.precedence_modes[prec_idx[k]] << "\"";
    }
    f << "],\n";

    // ---- interchangeable slots -------------------------------------------- #
    // Written ONLY by a scene that declares `slots:`. `slot_of` is what the
    // solver tests to decide whether a problem has slots at all (and `apex` and
    // the Gurobi backends refuse one that does), so emitting all-singleton groups
    // for an ordinary scene would turn every existing apex/Gurobi study into a
    // NotImplementedError for no gain. Absent, the seam is what it always was.
    //
    // Still geometry-free (ADR-0002): four id-to-id relations. `slot_of` groups
    // the candidates competing for one place in the plan (exactly one runs);
    // `object_of` names the physical object each candidate consumes (at most one
    // task per object), and it is a SEPARATE grouping because the same cube is a
    // candidate for several slots. Welds are omitted from `object_of` -- they
    // consume nothing, and lumping them under one empty id would make them
    // mutually exclusive with each other.
    if (spec_.has_slots) {
      std::set<std::string> live_slots;
      for (const auto & t : spec_.tasks) {
        if (kept.count(t.id)) {live_slots.insert(t.slot_id);}
      }

      f << "  \"slot_of\": {";
      bool first_slot = true;
      for (const auto & t : spec_.tasks) {
        if (!kept.count(t.id)) {continue;}
        f << (first_slot ? "" : ", ") << "\"" << t.id << "\": \"" << t.slot_id << "\"";
        first_slot = false;
      }
      f << "},\n";

      f << "  \"object_of\": {";
      bool first_obj = true;
      for (const auto & t : spec_.tasks) {
        if (t.kind != TaskKind::PickPlace || !kept.count(t.id)) {continue;}
        f << (first_obj ? "" : ", ") << "\"" << t.id << "\": \"" << t.object_id << "\"";
        first_obj = false;
      }
      f << "},\n";

      std::vector<std::size_t> slot_idx;
      for (std::size_t p = 0; p < spec_.slot_precedences.size(); ++p) {
        if (live_slots.count(spec_.slot_precedences[p].first) &&
          live_slots.count(spec_.slot_precedences[p].second))
        {
          slot_idx.push_back(p);
        }
      }
      f << "  \"slot_precedences\": [";
      for (std::size_t k = 0; k < slot_idx.size(); ++k) {
        f << (k ? ", " : "") << "[\"" << spec_.slot_precedences[slot_idx[k]].first << "\", \""
          << spec_.slot_precedences[slot_idx[k]].second << "\"]";
      }
      f << "],\n";

      f << "  \"slot_precedence_modes\": [";
      for (std::size_t k = 0; k < slot_idx.size(); ++k) {
        f << (k ? ", " : "") << "\"" << spec_.slot_precedence_modes[slot_idx[k]] << "\"";
      }
      f << "],\n";
    }

    // Present only in a refined artifact. It tells downstream stages that a robot's
    // trajectories are a CONTINUOUS chain in this order -- each one starts where the
    // previous ended, so they can no longer be reordered or executed in isolation.
    if (!chains_.empty()) {
      f << "  \"chains\": {\n";
      for (auto it = chains_.begin(); it != chains_.end(); ++it) {
        f << "    \"" << it->first << "\": [";
        for (std::size_t t = 0; t < it->second.size(); ++t) {
          f << (t ? ", " : "") << "\"" << it->second[t] << "\"";
        }
        f << "]" << (std::next(it) != chains_.end() ? "," : "") << "\n";
      }
      f << "  },\n";
    }

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
  SlotTable slots_;
  std::map<std::string, std::vector<std::string>> chains_;   // empty unless refining
  // Cumulative cost of `sceneFor` (see `run`'s per-pair timing line).
  double scene_seconds_{0.0};
  long scene_builds_{0};
  // Cumulative wall time inside `ikTo` (see `run`'s per-pair `timing`).
  double ik_seconds_{0.0};
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
  // Empty (the default): plan every (robot, task) pair independently, which is what the
  // scheduler needs as INPUT. Set: replan only the scheduled tasks, chained -- the
  // refinement pass, which needs the schedule and therefore runs after it (ADR-0008).
  std::string schedule_file, transit_file;
  node->get_parameter_or("schedule_file", schedule_file, std::string{});
  // Third mode: plan only the connecting motions a refinement asked for (ADR-0008).
  node->get_parameter_or("transit_file", transit_file, std::string{});

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::thread spinner([&exec]() {exec.spin();});

  int rc = 0;
  try {
    TrajectoryGenerator gen(node, loadTaskSpec(task_file));
    bool ok;
    if (!transit_file.empty()) {
      ok = gen.runTransits(out_file, transit_file);
    } else if (!schedule_file.empty()) {
      ok = gen.runChains(out_file, schedule_file);
    } else {
      ok = gen.run(out_file);
    }
    rc = ok ? 0 : 1;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(node->get_logger(), "%s", e.what());
    rc = 2;
  }

  exec.cancel();
  spinner.join();
  rclcpp::shutdown();
  return rc;
}
