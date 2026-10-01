// Offline trajectory generation for the multi-robot TAMP pipeline.
//
// For every ELIGIBLE (robot r, task i) pair -- every robot gets a trajectory for
// every task it may run, because choosing among them is the scheduler's job; in a
// scene that declares no `tool:` that is every pair (see `TaskSpec::eligible`) --
// this plans the full
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
  // How far from the seam, along the descent onto it and the retreat off it, a weld's
  // process allowance (`processAcm`) reaches. Only a weld that HAS an allowance is affected:
  // see `planWeldFrom`. 15 mm: a nozzle a few mm off the plate, not a torch sunk through it.
  double process_contact_depth{0.015};
  // Cap on the Cartesian speed of every robot link, m/s; 0 (default, every scene before it):
  // no cap. See `toSegment`. Pick it from Delta-t and the thinnest feature (ADR-0006):
  // cap * delta_t below `max_cartesian_step_warn`, with margin for the interpolant.
  double max_link_speed{0.0};
  // Synchronous hold (precedence mode `hold`, fabricator v2): how much longer than the
  // longest arc span of the held tack the pick-place keeps the part gripped at its place
  // pose, in seconds. The approach of the torch onto the first spot and its retreat off the
  // last one fall in this slack. See `run` for how the hold is sized.
  double hold_slack_s{2.0};
  // `hold_size: max` (default): the hold covers the LONGEST near span of the held process over
  // its planned welders; `min`: the shortest (the fastest welder), leaving the slower ones to
  // the solver (their windows may then be empty).
  bool hold_size_min{false};
  // `eligibility: strict | auto` (fabricator v3). strict -- every scene without `tool:`, and
  // the behaviour of every scene before v3: every eligible pair is planned, any failure makes
  // the run fail, the artifact has no `eligibility` block. auto -- the default for a scene
  // that declares `tool:`: the candidates (tool, then a task's `robots:`) go through an IK
  // GATE in the task's own world before planning, a pair that fails the gate or planning is
  // DROPPED with its reason, and the run fails only if a task is left with no trajectory.
  bool eligibility_auto{false};
  // Wall-time cap of ONE pair's planning (all its retries), s; 0: none. auto mode only.
  double pair_timeout_s{0.0};
  // Model the gripper fingers (fabricator v3): open while the hands are empty, closed at the
  // robot's `gripper_close` from GripClose on while the part is held, both states checked on
  // the GripClose and GripOpen dwells. Default: on iff the scene declares `tool:`, so every
  // scene before it keeps its fingers at the model's default (open), as it always had.
  bool model_fingers{false};
};

/// What a robot carries at its flange, which decides the tasks it may be given.
///
/// The TCP is `ee_link` whatever the tool: a torch robot names its wire tip there
/// (`robotN_torch_tcp`), a gripper robot its tool flange, and IK, the Cartesian legs and
/// the arc-length timing all work on that one link. The tool itself only feeds
/// `TaskSpec::eligible`, and never leaves this file: the seam learns a robot cannot do a
/// task by the ABSENCE of its (robot, task) duration, which is how the solver already
/// derives candidate robots (`model.py`).
enum class ToolKind : std::uint8_t
{
  Gripper = 0,
  Torch = 1,
};

struct RobotCfg
{
  std::string name;
  ToolKind tool{ToolKind::Gripper};
  /// Whether the YAML said `tool:` at all. Not a property of the robot: `TaskSpec` needs it
  /// to tell a scene written for tools from one written before them (see `eligible`).
  bool tool_declared{false};
  /// The EE pose relative to a seam point, for EVERY weld this robot runs. When present it
  /// replaces the per-weld `tool:` -- a torch's work angle and stickout are the torch's,
  /// not the seam's -- so a scene of several welders writes each seam once.
  std::optional<geometry_msgs::msg::Pose> weld_tool;
  /// Links that may TOUCH the parts a weld names in its `touch:` (a nozzle skimming the
  /// plate it welds), during the process only -- see `processAcm`. Empty: no allowance.
  std::vector<std::string> process_links;
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
  /// How many rotations about `tool_approach_axis` a WELD may be started at: 0 (default) is
  /// the task's own attitude and its half-turn flip, exactly as for a grasp; N > 0 tries the
  /// N equally spaced rotations k * 2 pi / N and keeps the collision-free ones (see
  /// `ikCandidates`). For a torch that rotation is not part of the process -- work and travel
  /// angle are set by the wire axis alone -- but with a bent swan neck it decides where the
  /// torch body and the wrist go, so leaving it free is what gets a torch into a channel.
  int tool_axis_free{0};
  /// `weld_approach: tool`: come onto a seam and leave it along the tool axis instead of
  /// vertically (see `planWeldFrom`). Default false (`vertical`), as every scene before it.
  bool weld_approach_along_tool{false};
  /// The finger joint the gripper is driven by, and its open / close values (`gripper_open`,
  /// `gripper_close`; `gripper_joint` defaults to `<name>_robotiq_85_left_knuckle_joint` when
  /// the model has it). Used only with `planning.model_fingers`; mimic joints follow.
  std::string gripper_joint;
  double gripper_open{0.0};
  double gripper_close{0.0};
  bool has_gripper_values{false};
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
  /// World poses of the seam's waypoints, >= 2: a polyline, run point to point. The
  /// `start`/`end` spelling is the two-point case and loads as exactly that.
  std::vector<geometry_msgs::msg::Pose> seam;
  /// The seam's LEGS: one entry per pass run with the arc on, >= 1. A `path:` (or
  /// `start`/`end`) seam is one leg, equal to `seam`. `legs:` (a tack of several spots,
  /// fabricator v2) gives several, run in order by the same robot with the arc OFF between
  /// them: off the joint along the tool axis by `leg_retreat`, across, and back on -- all
  /// Cartesian with the rail pinned (`planWeldFrom`). `seam` is then the first leg.
  std::vector<std::vector<geometry_msgs::msg::Pose>> legs;
  double leg_retreat{0.03};              ///< arc-off stand-off between two legs, m
  geometry_msgs::msg::Pose tool;         ///< EE pose relative to a seam point
  bool has_tool{false};                  ///< `tool:` given; else every robot brings `weld_tool`
  double speed{0.05};                    ///< traverse speed, m/s
  /// Weld: how far above the seam the pre-start sits. Pick-and-place (since the fabricator
  /// scene): how far above the spawn and the place poses the pre-grasp and pre-place sit --
  /// `planning.approach` unless the task says `approach:`. A plate set into a channel or
  /// against a beam end must clear the beam at its pre-place pose, and a tall end plate needs
  /// more than a stiffener does.
  double approach{0.12};
  /// World collision-object ids the robot's `process_links` may touch during this weld
  /// (the plate being welded, the beam). Resolved at load time from the YAML's `touch:`,
  /// which names fixtures and objects: an object becomes the `place__<slot>` stand-in(s)
  /// `sceneFor` puts where it will have been placed.
  std::vector<std::string> touch;

  // --- Both --------------------------------------------------------------- #
  /// The robots allowed to run this task, as the YAML's `robots:` lists them. Empty: the
  /// default rule by tool (`TaskSpec::eligible`).
  std::vector<std::string> robots;
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
  /// The `hold` precedences (pick-place i, process task j), also listed in `precedences`
  /// with mode "hold". Empty for every scene before the synchronous hold, and then nothing
  /// the hold adds -- the tail, the planning order, `hold_slots` -- happens at all.
  std::vector<std::pair<std::string, std::string>> holds;

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

  /// Whether `robot` may run `task`, i.e. whether the (robot, task) pair gets a trajectory.
  ///
  /// A task's own `robots:` decides when it has one. Otherwise the tool does -- a gripper
  /// robot fetches and places, a torch robot welds -- but ONLY in a scene that declares
  /// tools: in one where no robot says `tool:` every robot is eligible for every task, which
  /// is every scene written before tools existed, and what makes their artifacts (and seams)
  /// unchanged. An ineligible pair is simply absent from the artifact.
  bool eligible(const RobotCfg & robot, const TaskDef & t) const
  {
    if (!t.robots.empty()) {
      return std::find(t.robots.begin(), t.robots.end(), robot.name) != t.robots.end();
    }
    if (!tools_declared) {return true;}
    return (t.kind == TaskKind::PickPlace) == (robot.tool == ToolKind::Gripper);
  }

  /// True iff at least one robot declares `tool:` (see `eligible`).
  bool tools_declared{false};
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
///
/// `mesh: {file, visual_file, scale}`: `visual_file` is a finer mesh for RViz only and is
/// deliberately never read here -- planning and every check use `file`, the collision mesh.
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
  s.planning.max_link_speed = p["max_link_speed"] ? p["max_link_speed"].as<double>() : 0.0;
  s.planning.process_contact_depth =
    p["process_contact_depth"] ? p["process_contact_depth"].as<double>() : 0.015;
  s.planning.hold_slack_s = p["hold_slack_s"] ? p["hold_slack_s"].as<double>() : 2.0;
  if (s.planning.hold_slack_s < 0.0) {
    throw std::runtime_error("planning.hold_slack_s must be >= 0");
  }

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
    // What the robot carries (see `ToolKind`). Optional, and a scene where no robot names
    // one keeps the all-pairs rule it was written for (`TaskSpec::eligible`).
    if (n["tool"]) {
      const auto tool = n["tool"].as<std::string>();
      if (tool == "gripper") {r.tool = ToolKind::Gripper;}
      else if (tool == "torch") {r.tool = ToolKind::Torch;}
      else {
        throw std::runtime_error(
          "robot '" + r.name + "': tool must be gripper or torch, got '" + tool + "'");
      }
      r.tool_declared = true;
      s.tools_declared = true;
    }
    if (n["weld_tool"]) {r.weld_tool = poseFromYaml(n["weld_tool"]);}
    if (n["weld_approach"]) {
      const auto how = n["weld_approach"].as<std::string>();
      if (how == "tool") {r.weld_approach_along_tool = true;}
      else if (how != "vertical") {
        throw std::runtime_error(
          "robot '" + r.name + "': weld_approach must be vertical or tool, got '" + how + "'");
      }
    }
    if (n["tool_axis_free"]) {
      r.tool_axis_free = n["tool_axis_free"].as<int>();
      if (r.tool_axis_free < 0) {
        throw std::runtime_error("robot '" + r.name + "': tool_axis_free must be >= 0");
      }
    }
    for (const auto & l : n["process_links"]) {r.process_links.push_back(l.as<std::string>());}
    if (n["gripper_close"]) {
      r.gripper_close = n["gripper_close"].as<double>();
      r.gripper_open = n["gripper_open"] ? n["gripper_open"].as<double>() : 0.0;
      r.has_gripper_values = true;
    }
    if (n["gripper_joint"]) {r.gripper_joint = n["gripper_joint"].as<std::string>();}
    s.robots.push_back(r);
  }

  // v3 keys whose defaults depend on whether the scene declares `tool:` (known only now).
  {
    const std::string elig = p["eligibility"] ? p["eligibility"].as<std::string>() :
      std::string(s.tools_declared ? "auto" : "strict");
    if (elig != "auto" && elig != "strict") {
      throw std::runtime_error("planning.eligibility must be strict or auto, got '" + elig + "'");
    }
    s.planning.eligibility_auto = (elig == "auto");
    s.planning.pair_timeout_s = p["pair_timeout_s"] ? p["pair_timeout_s"].as<double>() : 0.0;
    s.planning.model_fingers =
      p["model_fingers"] ? p["model_fingers"].as<bool>() : s.tools_declared;
    const std::string hs = p["hold_size"] ? p["hold_size"].as<std::string>() : std::string("max");
    if (hs != "max" && hs != "min") {
      throw std::runtime_error("planning.hold_size must be max or min, got '" + hs + "'");
    }
    s.planning.hold_size_min = (hs == "min");
  }

  // `robots:` of a task, a slot or a weld: every name must be a robot of the scene. A
  // misspelt name would otherwise just make the task quietly unassignable to it.
  auto parseRobots = [&s](const YAML::Node & n, const std::string & what) {
      std::vector<std::string> out;
      for (const auto & r : n["robots"]) {
        const auto name = r.as<std::string>();
        const bool known = std::any_of(
          s.robots.begin(), s.robots.end(), [&](const RobotCfg & rc) {return rc.name == name;});
        if (!known) {
          throw std::runtime_error(what + ": robots names '" + name + "', which is no robot");
        }
        if (std::find(out.begin(), out.end(), name) == out.end()) {out.push_back(name);}
      }
      if (n["robots"] && out.empty()) {
        throw std::runtime_error(what + ": `robots:` is present but empty");
      }
      return out;
    };
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
    t.robots = parseRobots(n, "task '" + t.id + "'");
    t.approach = n["approach"] ? n["approach"].as<double>() : s.planning.approach;
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
      const auto slot_robots = parseRobots(n, "slot '" + slot_id + "'");   // every candidate
      const double slot_approach = n["approach"] ? n["approach"].as<double>() : s.planning.approach;
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
        t.robots = slot_robots;
        t.approach = slot_approach;
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
      const std::string what = "weld '" + t.id + "'";
      // The seam: `path:`, a polyline of >= 2 waypoints, or `start`/`end`, the two-point
      // shorthand every scene before polylines uses -- never both. Or `legs:`, a list of
      // such polylines run with the arc off between them (a tack of several spots).
      // A polyline as `path:` spells it: >= 2 waypoints, no zero-length segment. A
      // zero-length segment is a pure reorientation on the spot. The traverse is timed by ARC
      // LENGTH, so it would take no time at all -- infinite joint speed. Reject it here
      // rather than let `timeParameteriseAtSpeed` report it as a kinematic limit.
      auto polyline = [&what](const YAML::Node & list, const std::string & key) {
          std::vector<geometry_msgs::msg::Pose> out;
          for (const auto & wp : list) {out.push_back(poseFromYaml(wp));}
          if (out.size() < 2) {
            throw std::runtime_error(what + ": `" + key + "` needs at least two waypoints");
          }
          for (std::size_t k = 1; k < out.size(); ++k) {
            const auto & a = out[k - 1].position;
            const auto & b = out[k].position;
            if (std::hypot(b.x - a.x, b.y - a.y, b.z - a.z) < 1e-4) {
              throw std::runtime_error(
                what + ": `" + key + "` waypoints " + std::to_string(k - 1) + " and " +
                std::to_string(k) + " coincide; a segment must have length (the traverse is "
                "timed by arc length)");
            }
          }
          return out;
        };
      if (n["legs"]) {
        if (n["path"] || n["start"] || n["end"]) {
          throw std::runtime_error(what + ": give `legs:`, `path:` or `start`/`end`, only one");
        }
        if (!n["legs"].IsSequence() || n["legs"].size() == 0) {
          throw std::runtime_error(what + ": `legs:` must be a non-empty list of paths");
        }
        for (std::size_t k = 0; k < n["legs"].size(); ++k) {
          t.legs.push_back(polyline(n["legs"][k], "legs[" + std::to_string(k) + "]"));
        }
        t.seam = t.legs.front();
        t.leg_retreat = n["leg_retreat"] ? n["leg_retreat"].as<double>() : 0.03;
        if (!(t.leg_retreat > 0.0)) {
          throw std::runtime_error(what + ": `leg_retreat` must be positive");
        }
      } else if (n["path"]) {
        if (n["start"] || n["end"]) {
          throw std::runtime_error(what + ": give either `path:` or `start`/`end`, not both");
        }
        t.seam = polyline(n["path"], "path:");
        t.legs = {t.seam};
      } else {
        t.seam = {poseFromYaml(n["start"]), poseFromYaml(n["end"])};
        t.legs = {t.seam};
      }
      // Same convention as `objects[].grasp`: an EE pose expressed relative to
      // the point being worked on, composed with it to give the world EE pose.
      // Optional since `weld_tool:`: a robot that brings its own ignores this one, and a
      // weld without it must be run only by such robots (checked below).
      if (n["tool"]) {
        t.tool = poseFromYaml(n["tool"]);
        t.has_tool = true;
      }
      t.speed = n["speed"] ? n["speed"].as<double>() : s.planning.process_speed;
      t.approach = n["approach"] ? n["approach"].as<double>() : s.planning.approach;
      t.robots = parseRobots(n, what);
      for (const auto & id : n["touch"]) {t.touch.push_back(id.as<std::string>());}
      s.tasks.push_back(t);
    }
  }

  // ---- eligibility and the weld allowances, now that every task is known ----- #
  for (auto & t : s.tasks) {
    std::vector<const RobotCfg *> able;
    for (const auto & r : s.robots) {
      if (s.eligible(r, t)) {able.push_back(&r);}
    }
    // A task no robot may run has no candidate robot, which the solver would refuse as
    // trivially infeasible; say so here, where the cause is still visible.
    if (able.empty()) {
      throw std::runtime_error(
        "task '" + t.id + "' has no eligible robot: " +
        (t.robots.empty() ? std::string{"no robot carries the tool it needs ("} +
        (t.kind == TaskKind::Weld ? "torch" : "gripper") + "); give it `robots:`" :
        std::string{"check its `robots:`"}));
    }
    for (const auto * r : able) {
      // Listed explicitly, a robot runs the task whatever it carries; say when that is a
      // torch asked to grasp, or a gripper asked to weld, so it is visibly deliberate.
      if (r->tool_declared &&
        (t.kind == TaskKind::PickPlace) != (r->tool == ToolKind::Gripper))
      {
        RCLCPP_WARN(
          rclcpp::get_logger("trajectory_generator"),
          "task '%s' lists robot '%s', whose declared tool (%s) is not the one this kind of "
          "task needs -- planned anyway, as `robots:` asks", t.id.c_str(), r->name.c_str(),
          r->tool == ToolKind::Torch ? "torch" : "gripper");
      }
      if (t.kind == TaskKind::Weld && !t.has_tool && !r->weld_tool) {
        throw std::runtime_error(
          "weld '" + t.id + "' has no `tool:` and robot '" + r->name + "', which may run it, "
          "has no `weld_tool:` -- the EE pose over the seam is undefined");
      }
    }
    if (t.kind != TaskKind::Weld) {continue;}

    // `touch:` names parts; the world `sceneFor` builds names stand-ins. A fixture is itself.
    // An object is wherever a pick-and-place will have PLACED it -- the `place__<slot>` box
    // of every slot one of whose candidates carries it -- since a weld joins placed parts.
    std::vector<std::string> resolved;
    for (const auto & id : t.touch) {
      const bool fixture = std::any_of(
        s.fixtures.begin(), s.fixtures.end(), [&](const FixtureDef & f) {return f.id == id;});
      if (fixture) {
        resolved.push_back(id);
        continue;
      }
      std::set<std::string> slots;
      for (const auto & other : s.tasks) {
        if (other.kind == TaskKind::PickPlace && other.object_id == id) {
          slots.insert(other.slot_id);
        }
      }
      if (slots.empty()) {
        throw std::runtime_error(
          "weld '" + t.id + "': touch names '" + id + "', which is neither a fixture nor an "
          "object some task places (a part that never moves belongs under `fixtures:`)");
      }
      for (const auto & sid : slots) {resolved.push_back("place__" + sid);}
    }
    t.touch = resolved;
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
      //
      // `hold` (fabricator v2, PickPlace -> Weld only): the pick-place keeps its part gripped
      // at the place pose until the weld's arc is out. For the solver
      //
      //   hold       start[j] >= m0[i],  h[i] <= m0[j],  e[j] <= m1[i]
      //
      // with h[i] the first sample of i's hold tail (`hold_offsets`) and e[j] one past j's
      // last ProcessOff sample (`process_end_offsets`). It is emitted ALONE: the default
      // `pipeline` pair's m1[i] <= m1[j] contradicts it (the release follows the arc-out),
      // and `gate`'s m0[j] >= m1[i] is exactly what a hold replaces. Here, for the world the
      // tasks are planned in, it counts as a precedence like any other (the part is at its
      // place pose in j's world, `precedenceClosure`), and it gives i a hold tail.
      std::vector<std::string> modes;
      if (n.size() > 2) {
        const std::string mode = n[2].as<std::string>();
        if (mode != "pipeline" && mode != "gate" && mode != "hold") {
          throw std::runtime_error(
            "precedence [" + i + ", " + j + "] names an unknown mode '" + mode +
            "' (expected 'pipeline', 'gate' or 'hold')");
        }
        if (mode == "hold") {
          if (ki != TaskKind::PickPlace || kj != TaskKind::Weld) {
            throw std::runtime_error(
              "precedence [" + i + ", " + j + ", hold]: a hold pairs a pick-and-place (the "
              "part held) with a weld (the process run on it), in that order");
          }
          s.holds.emplace_back(i, j);
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

    // A process link the model does not have would make the allowance a silent no-op (an
    // ACM entry for a name nothing carries), and the weld would then fail on the very
    // contact it was meant to allow -- far from the typo that caused it.
    for (const auto & r : spec_.robots) {
      for (const auto & l : r.process_links) {
        if (!model_->hasLinkModel(l)) {
          throw std::runtime_error(
            "robot '" + r.name + "': process link '" + l + "' is not a link of the model");
        }
      }
      std::size_t n = 0;
      for (const auto & t : spec_.tasks) {n += spec_.eligible(r, t) ? 1 : 0;}
      RCLCPP_INFO(
        log_, "%s: tool %s, eligible for %zu of %zu task(s)", r.name.c_str(),
        !spec_.tools_declared ? "not declared (every task)" :
        (r.tool == ToolKind::Torch ? "torch" : "gripper"), n, spec_.tasks.size());
    }

    // Fingers (v3): each gripper robot's finger joint, defaulted from the Robotiq naming.
    if (spec_.planning.model_fingers) {
      for (auto & r : spec_.robots) {
        if (!r.has_gripper_values) {continue;}
        if (r.gripper_joint.empty() &&
          model_->hasJointModel(r.name + "_robotiq_85_left_knuckle_joint"))
        {
          r.gripper_joint = r.name + "_robotiq_85_left_knuckle_joint";
        }
        if (!r.gripper_joint.empty() && !model_->hasJointModel(r.gripper_joint)) {
          throw std::runtime_error(
            "robot '" + r.name + "': gripper_joint '" + r.gripper_joint + "' is not a joint of the model");
        }
        if (!r.gripper_joint.empty()) {
          RCLCPP_INFO(
            log_, "%s: fingers modelled on '%s' (open %.3f, closed %.3f)", r.name.c_str(),
            r.gripper_joint.c_str(), r.gripper_open, r.gripper_close);
        }
      }
    }
    RCLCPP_INFO(
      log_, "eligibility: %s%s", spec_.planning.eligibility_auto ? "auto (IK gate, failures drop "
      "the pair)" : "strict (every eligible pair must plan)",
      spec_.planning.model_fingers ? "; fingers modelled" : "");

    pipeline_ = std::make_shared<planning_pipeline::PlanningPipeline>(model_, node_, "ompl");
    before_ = precedenceClosure(spec_);
    slots_ = slotTable(spec_, before_);
  }

  /// Plan only some pairs (`only_pairs`, a debugging aid): comma-separated `robot/task`,
  /// `task` or `robot/` tokens. Empty (the default) plans every eligible pair. A partial
  /// artifact is for inspecting one pair's planning, never for the downstream stages.
  void setOnly(const std::string & csv)
  {
    std::stringstream ss(csv);
    for (std::string tok; std::getline(ss, tok, ',');) {
      if (!tok.empty()) {only_.push_back(tok);}
    }
    if (!only_.empty()) {
      RCLCPP_WARN(log_, "only_pairs=%s: the artifact will hold these pairs only", csv.c_str());
    }
  }

  bool selected(const RobotCfg & robot, const TaskDef & task) const
  {
    if (only_.empty()) {return true;}
    for (const auto & t : only_) {
      if (t == robot.name + "/" + task.id || t == task.id || t == robot.name + "/") {return true;}
    }
    return false;
  }

  bool run(const std::string & out_path)
  {
    int n_missing = 0;       // pairs with no trajectory at all
    int n_invalid = 0;       // pairs written, but failing validateSamples
    int total_retries = 0;   // replans over the whole run (0 for a scene that plans clean)
    std::size_t expected = 0;   // eligible pairs: the ones that need a trajectory
    double worst_step = 0.0;

    // Planning ORDER vs artifact order. The artifact lists the pairs robot by robot, task by
    // task, as it always has. A synchronous hold (`hold` precedence, pick-place i holding
    // for process j) is planned in three steps, then everything else:
    //   pass 0  the holding pick-place i, WITHOUT its hold: it gives the configuration the
    //           handler holds the part in (its GripOpen configuration, the place pose);
    //   pass 1  the held process j, for every robot that may run it, in a world where the
    //           handler STANDS at that configuration (`sceneFor`, `hold_pose_`): every sample
    //           of j is thereby validated against the handler holding the part, on the exact
    //           geometry -- the hold's clearance, certified by construction as HOME's is;
    //   then    i's hold is sized from j's trajectories and inserted into i as a dwell
    //           before GripOpen (`insertHoldTail`), and i is re-validated.
    // Without a hold passes 0 and 1 are empty and nothing changes.
    std::set<std::string> held_tasks;       // j of some hold (i, j)
    for (const auto & [i, j] : spec_.holds) {held_tasks.insert(j);}
    std::set<std::string> holders;          // i of some hold (i, j)
    for (const auto & [i, j] : spec_.holds) {
      holders.insert(i);
      std::vector<std::string> who;
      for (const auto & r : spec_.robots) {if (spec_.eligible(r, spec_.task(i))) {who.push_back(r.name);}}
      // One handler per holding task: the held process is planned against THE handler's
      // hold configuration, and a second candidate would need a world of its own.
      if (who.size() != 1) {
        throw std::runtime_error(
          "hold [" + i + ", " + j + "]: the holding task must have exactly one eligible robot");
      }
    }
    // A held process is planned even when `only_pairs` leaves it out, if a selected
    // pick-place needs its span; it is then not written.
    std::set<std::string> needed;
    for (const auto & [i, j] : spec_.holds) {
      for (const auto & r : spec_.robots) {
        if (spec_.eligible(r, spec_.task(i)) && selected(r, spec_.task(i))) {needed.insert(j);}
      }
    }

    std::map<std::pair<std::size_t, std::size_t>, std::string> written;   // (robot, task) index
    // Trajectories kept aside until their hold is known (pass 0), to be written after it.
    struct Deferred
    {
      mrct::ResampledTrajectory traj;
      PairTiming timing;
      double cart_step;
    };
    std::map<std::pair<std::size_t, std::size_t>, Deferred> deferred;
    std::map<std::string, std::vector<int>> spans;   // held task -> near span of each planned robot
    const bool autoElig = spec_.planning.eligibility_auto;
    eligibility_.clear();

    auto planPair = [&](std::size_t ri, std::size_t ti, bool write) {
        const RobotCfg & robot = spec_.robots[ri];
        const TaskDef & task = spec_.tasks[ti];
        // auto eligibility: the IK gate first; a pair that fails it is dropped, with its reason
        if (autoElig) {
          std::string why;
          if (!gatePair(robot, task, why)) {
            RCLCPP_WARN(log_, "%s / %s: skipped by the gate (%s)", robot.name.c_str(),
                        task.id.c_str(), why.c_str());
            eligibility_[task.id][robot.name] = "gate: " + why;
            return;
          }
        }
        if (write) {++expected;}
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
        const auto t_pair = Clock::now();
        if (autoElig && spec_.planning.pair_timeout_s > 0.0) {
          pair_deadline_ = t_pair + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(spec_.planning.pair_timeout_s));
        }
        for (int attempt = 0; attempt < max_tries; ++attempt) {
          if (attempt > 0 && timedOut()) {
            RCLCPP_WARN(log_, "%s / %s: pair time cap (%.0f s) reached after %d attempt(s)",
                        robot.name.c_str(), task.id.c_str(), spec_.planning.pair_timeout_s, attempt);
            break;
          }
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
        pair_deadline_.reset();
        if (write) {total_retries += used_retries;}
        if (autoElig && (!planned || bad > 0)) {
          // auto: a pair that did not plan, or whose samples stayed in collision, is DROPPED
          const std::string why = !planned ?
            "no plan after " + std::to_string(used_retries + 1) + " attempt(s)" :
            std::to_string(bad) + " sample(s) in collision after " +
            std::to_string(used_retries + 1) + " attempt(s)";
          RCLCPP_WARN(log_, "%s / %s: dropped (%s, %.1f s)", robot.name.c_str(), task.id.c_str(),
                      why.c_str(), seconds(t_pair));
          eligibility_[task.id][robot.name] = "failed: " + why;
          if (write) {--expected;}
          return;
        }
        if (!planned) {
          RCLCPP_ERROR(
            log_, "FAILED to plan %s / %s after %d attempt(s) -- every eligible (robot, task) "
            "pair needs a trajectory, so the artifact is incomplete", robot.name.c_str(),
            task.id.c_str(), max_tries);
          if (write) {++n_missing;}
          return;
        }
        if (bad > 0) {
          RCLCPP_ERROR(
            log_, "%s / %s: %d resampled sample(s) are IN COLLISION after %d attempt(s) -- "
            "the interpolant left the validated path. Reduce delta_t or densify the plan.",
            robot.name.c_str(), task.id.c_str(), bad, max_tries);
          if (write) {++n_invalid;}
        }
        if (held_tasks.count(task.id)) {
          const int arc = processSpan(traj);
          const int sp = nearSpan(robot, task, traj);
          RCLCPP_INFO(
            log_, "%s / %s: arc span (first ProcessOn .. last ProcessOff) %d slots (%.2f s); "
            "near span (tool within approach of the seam) %d slots (%.2f s)",
            robot.name.c_str(), task.id.c_str(), arc, arc * spec_.disc.delta_t, sp,
            sp * spec_.disc.delta_t);
          spans[task.id].push_back(std::max(sp, arc));
        }
        if (!write) {return;}

        worst_step = std::max(worst_step, cart_step);
        RCLCPP_INFO(
          log_, "%s / %s: K=%zu slots (%.2f s), max joint step %.4f, "
          "max link travel per slot %.4f m",
          robot.name.c_str(), task.id.c_str(), traj.num_samples,
          traj.num_samples * spec_.disc.delta_t, traj.max_joint_step, cart_step);
        RCLCPP_INFO(
          log_, "%s / %s: fastest link '%s' at sample %zu (phase %s)", robot.name.c_str(),
          task.id.c_str(), fastest_link_.c_str(), fastest_sample_, fastest_phase_.c_str());
        RCLCPP_INFO(
          log_, "%s / %s: timing: plan %.2f s, validateSamples %.2f s (%d attempt(s)); "
          "%ld scene build(s), %.2f s", robot.name.c_str(), task.id.c_str(), plan_s, validate_s,
          used_retries + 1, scene_builds_ - scene_n0, scene_seconds_ - scene_s0);

        const PairTiming timing{ik_seconds_ - ik_s0, plan_s, validate_s, used_retries + 1};
        if (autoElig) {eligibility_[task.id][robot.name] = "planned";}
        if (holders.count(task.id)) {
          // pass 0: the hold configuration for the held process's world; written later
          std::vector<double> place_q;
          for (std::size_t k = 0; k < traj.num_samples; ++k) {
            if (traj.phase[k] == Phase::GripOpen) {
              place_q.assign(traj.sample(k), traj.sample(k) + traj.num_joints);
              break;
            }
          }
          for (const auto & [hi, hj] : spec_.holds) {
            if (hi == task.id) {hold_pose_[hj] = HoldPose{robot.name, task.slot_id, place_q, task.id};}
          }
          if (bad == 0) {deferred[{ri, ti}] = Deferred{traj, timing, cart_step};}
          return;
        }
        written[{ri, ti}] = toJson(robot, task, traj, &timing);
      };

    // Pass 0: the holding pick-places, without their hold (none without a `hold`).
    for (std::size_t ri = 0; ri < spec_.robots.size(); ++ri) {
      for (std::size_t ti = 0; ti < spec_.tasks.size(); ++ti) {
        const auto & robot = spec_.robots[ri];
        const auto & task = spec_.tasks[ti];
        if (!holders.count(task.id) || !spec_.eligible(robot, task)) {continue;}
        planPair(ri, ti, true);
      }
    }

    // Pass 1: the held processes, against the handler at its hold configuration.
    for (std::size_t ri = 0; ri < spec_.robots.size(); ++ri) {
      for (std::size_t ti = 0; ti < spec_.tasks.size(); ++ti) {
        const auto & robot = spec_.robots[ri];
        const auto & task = spec_.tasks[ti];
        if (!held_tasks.count(task.id) || !spec_.eligible(robot, task)) {continue;}
        if (!hold_pose_.count(task.id)) {
          RCLCPP_ERROR(log_, "%s / %s: not planned, its holding pick-place has no trajectory",
                       robot.name.c_str(), task.id.c_str());
          if (autoElig) {
            eligibility_[task.id][robot.name] = "failed: its holding pick-place has no trajectory";
          } else if (selected(robot, task)) {++expected; ++n_missing;}
          continue;
        }
        const bool sel = selected(robot, task);
        if (!sel && !needed.count(task.id)) {continue;}
        planPair(ri, ti, sel);
      }
    }
    // The hold of each pick-place: the longest NEAR span of its held process over the robots
    // that may run it (whichever the solver picks, it must fit), plus the slack. The solver
    // needs e[j] - m0[j] <= m1[i] - h[i] = hold_slots, which the arc span alone would meet --
    // but the torch comes within reach of the part already at its pre-start (`approach`
    // off the seam) and stays until its retreat is over, and before h / after GripOpen the
    // part is the carried one (a sphere in mu, and moving). Sized on the arc alone, every
    // offset of the hold window was forbidden (fabricator v2 probe, 2026-09-28): the torch's
    // pre-start against the part still descending, its retreat against the gripper leaving.
    // So the hold covers the whole near span; the slack takes the end of the flight in.
    hold_slots_.clear();
    for (const auto & [i, j] : spec_.holds) {
      if (!spans.count(j) || spans[j].empty()) {
        RCLCPP_ERROR(
          log_, "hold [%s, %s]: no trajectory of %s planned, so %s's hold cannot be sized",
          i.c_str(), j.c_str(), j.c_str(), i.c_str());
        continue;
      }
      const int slack = static_cast<int>(
        std::ceil(spec_.planning.hold_slack_s / spec_.disc.delta_t - 1e-9));
      // `hold_size: max` (default): the slowest welder fits; `min`: the fastest one does
      const int sp = spec_.planning.hold_size_min ?
        *std::min_element(spans[j].begin(), spans[j].end()) :
        *std::max_element(spans[j].begin(), spans[j].end());
      hold_slots_[i] = std::max(hold_slots_[i], sp + slack);
    }
    for (const auto & [i, h] : hold_slots_) {
      RCLCPP_INFO(
        log_, "hold: %s keeps its part gripped at the place pose for %d slots (%.2f s)",
        i.c_str(), h, h * spec_.disc.delta_t);
    }
    // The holding pick-places, now with their hold: a dwell at the place configuration before
    // GripOpen, re-validated (the configuration is GripOpen's, the part attached as there).
    for (auto & [key, d] : deferred) {
      const RobotCfg & robot = spec_.robots[key.first];
      const TaskDef & task = spec_.tasks[key.second];
      if (!hold_slots_.count(task.id)) {
        RCLCPP_ERROR(log_, "%s / %s: not written, its hold could not be sized",
                     robot.name.c_str(), task.id.c_str());
        if (autoElig) {
          eligibility_[task.id][robot.name] = "failed: no held process planned, hold not sized";
          --expected;
        } else {
          ++n_missing;
        }
        continue;
      }
      insertHoldTail(d.traj, hold_slots_.at(task.id));
      double step = 0.0;
      const int bad = validateSamples(robot, task, d.traj, step);
      if (bad > 0) {
        RCLCPP_ERROR(log_, "%s / %s: %d sample(s) in collision after inserting the hold",
                     robot.name.c_str(), task.id.c_str(), bad);
        if (autoElig) {
          eligibility_[task.id][robot.name] =
            "failed: " + std::to_string(bad) + " sample(s) in collision after inserting the hold";
          --expected;
          continue;
        }
        ++n_invalid;
      }
      worst_step = std::max(worst_step, d.cart_step);
      RCLCPP_INFO(
        log_, "%s / %s: K=%zu slots (%.2f s) with its hold of %d, max link travel per slot %.4f m",
        robot.name.c_str(), task.id.c_str(), d.traj.num_samples,
        d.traj.num_samples * spec_.disc.delta_t, hold_slots_.at(task.id), d.cart_step);
      written[key] = toJson(robot, task, d.traj, &d.timing);
    }
    // pass-0 pairs that failed to plan were counted there; the ones deferred are in `written`

    // Pass 2: everything else.
    for (std::size_t ri = 0; ri < spec_.robots.size(); ++ri) {
      for (std::size_t ti = 0; ti < spec_.tasks.size(); ++ti) {
        const auto & robot = spec_.robots[ri];
        const auto & task = spec_.tasks[ti];
        // An ineligible pair is not planned and not written: the solver reads the robots a
        // task may go to off the (robot, task) durations the seam carries, so absence IS
        // the statement "this robot cannot do this task" (`TaskSpec::eligible`).
        if (!spec_.eligible(robot, task)) {continue;}
        if (!selected(robot, task)) {continue;}
        if (held_tasks.count(task.id) || holders.count(task.id)) {continue;}   // passes 0, 1
        planPair(ri, ti, true);
      }
    }
    std::vector<std::string> artifact;   // one JSON object per ELIGIBLE (robot, task)
    for (const auto & [key, json] : written) {artifact.push_back(json);}   // robot, then task

    // auto eligibility: the matrix, and the one failure that still stops the run -- a task no
    // candidate could take.
    std::vector<std::string> orphans;
    if (autoElig) {
      RCLCPP_INFO(log_, "eligibility (auto): per task, per candidate robot");
      for (const auto & t : spec_.tasks) {
        if (!only_.empty()) {
          bool any_sel = false;
          for (const auto & r : spec_.robots) {any_sel = any_sel || (spec_.eligible(r, t) && selected(r, t));}
          if (!any_sel) {continue;}
        }
        std::string line;
        int n_ok = 0;
        for (const auto & r : spec_.robots) {
          if (!spec_.eligible(r, t)) {continue;}
          const auto it = eligibility_[t.id].find(r.name);
          const std::string st = it == eligibility_[t.id].end() ? std::string("not selected") : it->second;
          n_ok += st == "planned";
          line += (line.empty() ? "" : ", ") + r.name + ": " + st;
        }
        RCLCPP_INFO(log_, "  %-14s %s", t.id.c_str(), line.c_str());
        if (n_ok == 0) {orphans.push_back("task " + t.id + ": 0 trajectories -- " + line);}
      }
      for (const auto & o : orphans) {RCLCPP_ERROR(log_, "%s", o.c_str());}
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
    if (total_retries > 0) {
      RCLCPP_WARN(log_, "plan retries used over the whole run: %d", total_retries);
    }
    if (spec_.planning.max_link_speed > 0.0) {
      RCLCPP_INFO(
        log_, "max_link_speed %.3f m/s: %ld segment(s) re-timed slower to respect it",
        spec_.planning.max_link_speed, capped_segments_);
    }
    const bool all_ok = (n_missing == 0 && n_invalid == 0 && orphans.empty());
    if (all_ok) {
      RCLCPP_INFO(
        log_, "OK: %zu/%zu trajectories written to %s", artifact.size(), expected,
        out_path.c_str());
    } else if (!orphans.empty() && n_missing == 0 && n_invalid == 0) {
      RCLCPP_ERROR(
        log_, "FAILED: %zu task(s) with 0 trajectories (see above); %zu trajectories written to %s",
        orphans.size(), artifact.size(), out_path.c_str());
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
    if (!spec_.holds.empty()) {
      throw std::runtime_error(
        "chained replanning does not support `hold` precedences (the hold tail is sized in "
        "`run` from the held tack's trajectories)");
    }
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
  /// Insert a hold of `slots` samples before the first GripOpen sample of a pick-place: copies
  /// of that (frozen, place-pose) sample, phase Carrying, part attached, support allowance as
  /// on GripOpen. The trajectory is otherwise unchanged -- the hold is a pure dwell.
  static void insertHoldTail(mrct::ResampledTrajectory & traj, int slots)
  {
    std::size_t m1 = traj.num_samples;
    for (std::size_t k = 0; k < traj.num_samples; ++k) {
      if (traj.phase[k] == Phase::GripOpen) {m1 = k; break;}
    }
    if (m1 == traj.num_samples || slots <= 0) {
      throw std::runtime_error("insertHoldTail: no GripOpen sample, or an empty hold");
    }
    const std::size_t n = traj.num_joints;
    std::vector<double> row(traj.sample(m1), traj.sample(m1) + n);
    std::vector<double> block;
    for (int s = 0; s < slots; ++s) {block.insert(block.end(), row.begin(), row.end());}
    traj.positions.insert(traj.positions.begin() + static_cast<long>(m1 * n), block.begin(), block.end());
    traj.phase.insert(traj.phase.begin() + static_cast<long>(m1), slots, Phase::Carrying);
    traj.object_state.insert(
      traj.object_state.begin() + static_cast<long>(m1), slots, ObjectState::Attached);
    traj.support_contact.insert(traj.support_contact.begin() + static_cast<long>(m1), slots, 1);
    traj.num_samples += static_cast<std::size_t>(slots);
  }

  /// Whether `task` is the pick-place of some `hold` precedence (it gets a hold tail).
  bool holdsFor(const std::string & task) const
  {
    return std::any_of(
      spec_.holds.begin(), spec_.holds.end(), [&](const auto & h) {return h.first == task;});
  }

  /// A process trajectory's arc span, in slots: from its first ProcessOn sample to its LAST
  /// ProcessOff sample, both included -- e - m0 in the solver's terms (`process_end_offsets`
  /// minus `pick_offsets`), whatever the number of legs in between.
  /// The NEAR span of a process trajectory, in slots: from the first to the last sample at
  /// which the tool (`ee_link`) is within the weld's `approach` (+1 cm) of any of its seam
  /// waypoints -- the pre-start, the Cartesian approach, every leg and every transfer, the
  /// retreat. What a hold must cover for the torch never to meet the part while it moves.
  int nearSpan(
    const RobotCfg & robot, const TaskDef & task, const mrct::ResampledTrajectory & traj) const
  {
    const auto * jmg = model_->getJointModelGroup(robot.planning_group);
    const auto * link = model_->getLinkModel(robot.ee_link);
    const auto & tool = robot.weld_tool ? *robot.weld_tool : task.tool;
    std::vector<Eigen::Vector3d> pts;
    for (const auto & leg : task.legs) {
      for (const auto & wp : leg) {
        const auto ee = compose(wp, tool);
        pts.emplace_back(ee.position.x, ee.position.y, ee.position.z);
      }
    }
    const double reach = task.approach + 0.01;
    moveit::core::RobotState st = homeState();
    long first = -1, last = -1;
    for (std::size_t k = 0; k < traj.num_samples; ++k) {
      st.setJointGroupPositions(jmg, traj.sample(k));
      st.update();
      const Eigen::Vector3d p = st.getGlobalLinkTransform(link).translation();
      // distance to the seam polyline's waypoints (legs are short: 2-cm spots, 2-cm steps)
      double d = 1e9;
      for (const auto & q : pts) {d = std::min(d, (p - q).norm());}
      // and to the segments between consecutive waypoints of a leg
      std::size_t base = 0;
      for (const auto & leg : task.legs) {
        for (std::size_t w = 1; w < leg.size(); ++w) {
          const Eigen::Vector3d a = pts[base + w - 1], b = pts[base + w];
          const double u = std::clamp((p - a).dot(b - a) / std::max(1e-12, (b - a).squaredNorm()), 0.0, 1.0);
          d = std::min(d, (p - (a + u * (b - a))).norm());
        }
        base += leg.size();
      }
      if (d <= reach) {
        if (first < 0) {first = static_cast<long>(k);}
        last = static_cast<long>(k);
      }
    }
    return first < 0 ? 0 : static_cast<int>(last - first + 1);
  }

  static int processSpan(const mrct::ResampledTrajectory & traj)
  {
    long first = -1, last = -1;
    for (std::size_t k = 0; k < traj.num_samples; ++k) {
      if (first < 0 && traj.phase[k] == Phase::ProcessOn) {first = static_cast<long>(k);}
      if (traj.phase[k] == Phase::ProcessOff) {last = static_cast<long>(k);}
    }
    if (first < 0 || last < first) {
      throw std::runtime_error("a held task has no ProcessOn .. ProcessOff: it must be a weld");
    }
    return static_cast<int>(last - first + 1);
  }

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
    // A HELD process (`hold`) runs while its handler holds the part at the place pose: the
    // handler stands there in this world, so every sample of the process is validated
    // against it on the exact geometry (see `run`).
    const auto hp = hold_pose_.find(task.id);
    const bool held_world = hp != hold_pose_.end() && hp->second.robot != robot.name;
    if (held_world) {
      const auto & handler = robotByName(hp->second.robot);
      state.setJointGroupPositions(
        model_->getJointModelGroup(handler.planning_group), hp->second.q);
      setFingers(state, handler, true);          // holding the part: fingers closed
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
    if (held_world) {
      // The handler's fingers close on the part it holds (its `place__` box here), exactly
      // as its `touch_links` touch the part while carrying it; nothing else is allowed.
      auto & acm = scene->getAllowedCollisionMatrixNonConst();
      for (const auto & l : robotByName(hp->second.robot).touch_links) {
        acm.setEntry(l, "place__" + hp->second.slot, true);
      }
    }
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
    // Fingers (when modelled): closed on the part while it is attached, open otherwise.
    setFingers(scene->getCurrentStateNonConst(), robot, st == ObjectState::Attached);
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
  moveit::core::RobotState stateInScene(
    const planning_scene::PlanningScenePtr & scene, const moveit::core::RobotState & positions) const
  {
    moveit::core::RobotState s(scene->getCurrentState());
    s.setVariablePositions(positions.getVariablePositions());
    // With the fingers modelled, they are the SCENE's (open, or closed while a part is held:
    // `setObjectState`), never whatever the state handed in happened to carry.
    if (spec_.planning.model_fingers) {
      for (const auto & r : spec_.robots) {
        if (!fingersModelled(r)) {continue;}
        const double v = scene->getCurrentState().getVariablePosition(r.gripper_joint);
        s.setJointPositions(r.gripper_joint, &v);
      }
    }
    s.update();
    return s;
  }

  /// Whether `r`'s fingers are modelled (`planning.model_fingers` and a known finger joint).
  bool fingersModelled(const RobotCfg & r) const
  {
    return spec_.planning.model_fingers && r.has_gripper_values && !r.gripper_joint.empty();
  }

  /// Set `r`'s fingers open or closed (mimic joints follow). No-op unless modelled.
  void setFingers(moveit::core::RobotState & st, const RobotCfg & r, bool closed) const
  {
    if (!fingersModelled(r)) {return;}
    const double v = closed ? r.gripper_close : r.gripper_open;
    st.setJointPositions(r.gripper_joint, &v);
    st.update();
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

  /// The weld's counterpart of `supportAcm`: the scene's ACM plus contact between the
  /// robot's `process_links` (torch nozzle, wire) and the parts the weld `touch`es.
  ///
  /// A fillet weld puts the nozzle a few millimetres off two plates at once, and a torch
  /// modelled a little fat touches them -- which FCL reports as a collision on every sample
  /// of the traverse. Same scope as the support allowance, for the same reason: the descent
  /// onto the seam start, the strike, the traverse, the cut and the retreat off the seam end
  /// (`planWeldFrom` flags exactly those segments); the flights out and home are checked in
  /// full. Only the named links against the named parts -- the rest of the torch, the rest of
  /// the arm and every other obstacle are checked as always. A copy, never the scene's.
  collision_detection::AllowedCollisionMatrix processAcm(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const TaskDef & task) const
  {
    collision_detection::AllowedCollisionMatrix acm = scene->getAllowedCollisionMatrix();
    for (const auto & link : robot.process_links) {
      for (const auto & id : task.touch) {acm.setEntry(link, id, true);}
    }
    return acm;
  }

  /// Whether a weld grants any process allowance at all. Without one its segments are not
  /// flagged, so a scene that names no `process_links`/`touch` checks exactly as before.
  static bool hasProcessAllowance(const RobotCfg & robot, const TaskDef & task)
  {
    return task.kind == TaskKind::Weld && !robot.process_links.empty() && !task.touch.empty();
  }

  /// The ACM of a sample taken from a flagged (`support_contact`) segment: the support
  /// allowance of a pick-and-place, the process allowance of a weld.
  collision_detection::AllowedCollisionMatrix contactAcm(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const TaskDef & task) const
  {
    return task.kind == TaskKind::PickPlace ?
           supportAcm(scene, task.object_id) : processAcm(scene, robot, task);
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

  /// The pre-start configurations a weld may start from, nearest `seed` first.
  ///
  /// `tool_axis_free` 0: whatever `ikTo` returns (the named attitude or its half-turn flip),
  /// so a robot that does not set it plans exactly as before. N > 0: the target rotated about
  /// the tool axis by k * 2 pi / N for k = 0..N-1, each solved on its own, every
  /// collision-free solution kept, sorted by joint distance from `seed` (the start, i.e. home
  /// for a stand-alone task) and deduplicated. The caller tries them in that order.
  std::vector<moveit::core::RobotState> ikCandidates(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const geometry_msgs::msg::Pose & target, const moveit::core::RobotState & seed)
  {
    std::vector<moveit::core::RobotState> out;
    if (robot.tool_axis_free <= 0) {
      moveit::core::RobotState s(seed);
      if (ikTo(scene, robot, target, seed, s)) {out.push_back(s);}
      return out;
    }
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
    const moveit::core::RobotState seeded = stateInScene(scene, seed);
    auto valid = [&](moveit::core::RobotState * s, const moveit::core::JointModelGroup * g,
        const double * values) {
        s->setJointGroupPositions(g, values);
        s->update();
        return !scene->isStateColliding(*s, g->getName());
      };
    Eigen::Isometry3d goal;
    tf2::fromMsg(target, goal);

    const int n = robot.tool_axis_free;
    std::vector<std::pair<double, moveit::core::RobotState>> found;
    for (int k = 0; k < n; ++k) {
      const Eigen::Isometry3d g =
        goal * Eigen::AngleAxisd(2.0 * M_PI * k / n, robot.tool_approach_axis);
      moveit::core::RobotState s(seeded);
      if (!s.setFromIK(jmg, g, robot.ee_link, 0.5, valid)) {continue;}
      found.emplace_back(seed.distance(s, jmg), s);
    }
    std::stable_sort(
      found.begin(), found.end(), [](const auto & x, const auto & y) {return x.first < y.first;});
    for (const auto & [d, s] : found) {
      (void)d;
      const bool dup = std::any_of(
        out.begin(), out.end(), [&](const moveit::core::RobotState & o) {
          return o.distance(s, jmg) < 1e-3;
        });
      if (!dup) {out.push_back(s);}
    }
    RCLCPP_INFO(
      log_, "  pre-start: %zu of %d rotations about the tool axis have a collision-free IK",
      found.size(), n);
    return out;
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
    // `support_object` non-empty: this is a pick/place leg and that object may
    // touch the support surface (see `supportAcm`). Empty: the scene's ACM as is.
    return planCartesianStep(
      scene, robot, start, delta, phase, out, end_state, supportAcm(scene, support_object),
      !support_object.empty(), speed);
  }

  /// The same, against an explicit ACM: `contact` says whether `acm` carries an allowance,
  /// and is recorded on the segment so `validateSamples` re-checks its samples with the
  /// same one (`contactAcm`). The weld's descent and retreat come in here.
  bool planCartesianStep(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const moveit::core::RobotState & start, const Eigen::Vector3d & delta, Phase phase,
    Segment & out, moveit::core::RobotState & end_state,
    const collision_detection::AllowedCollisionMatrix & acm, bool contact, double speed = 0.0)
  {
    const auto * full = model_->getJointModelGroup(robot.planning_group);
    std::vector<std::shared_ptr<moveit::core::RobotState>> path;
    if (!cartesianPath(scene, robot, start, delta, std::nullopt, acm, path)) {return false;}

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
    out.support_contact = contact;
    return toSegment(rt, phase, out);
  }

  /// The straight-line interpolation every Cartesian leg is made of: on the 6-DOF arm
  /// group, rail pinned (see `planCartesianStep` for why), every state checked against
  /// `acm`. `path` receives the states, the start state first.
  ///
  /// `rotation` empty: the EE translates by `delta` HOLDING its orientation -- every
  /// approach and retreat, and every straight seam. Given: the EE also turns to that world
  /// orientation, slerped along the line (MoveIt's pose-target interpolation), which is how
  /// a polyline seam changes the torch's attitude from one waypoint to the next.
  ///
  /// False, with the diagnosis in the log, if less than 99 % of the line is achieved.
  bool cartesianPath(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const moveit::core::RobotState & start, const Eigen::Vector3d & delta,
    const std::optional<Eigen::Quaterniond> & rotation,
    const collision_detection::AllowedCollisionMatrix & acm,
    std::vector<std::shared_ptr<moveit::core::RobotState>> & path)
  {
    const auto * arm = model_->getJointModelGroup(robot.arm_group);
    const auto * link = model_->getLinkModel(robot.ee_link);

    auto state = std::make_shared<moveit::core::RobotState>(stateInScene(scene, start));
    path.clear();

    auto valid = [&](moveit::core::RobotState * s, const moveit::core::JointModelGroup * g,
        const double * values) {
        s->setJointGroupPositions(g, values);
        s->update();
        // Check the WHOLE robot, not just the arm: the group being interpolated is
        // the arm, but a collision anywhere (rail carriage, the other robot, the
        // table) still invalidates the state.
        return !collides(scene, *s, robot.planning_group, acm);
      };

    const double wanted_m = delta.norm();
    if (!rotation) {
      // NOTE the return value. The Eigen::Vector3d (translation) overload returns the
      // DISTANCE ACHIEVED IN METRES -- only the Isometry3d (pose-target) overload
      // returns a 0..1 fraction. Comparing this against a fraction silently rejects
      // every successful path (a fully-achieved 12 cm descent "fails" a `< 0.99`
      // test), which looks exactly like a planner problem and is not one.
      const double achieved_m = moveit::core::CartesianInterpolator::computeCartesianPath(
        state.get(), arm, path, link, delta, /*global_reference_frame=*/true,
        moveit::core::MaxEEFStep(spec_.planning.cartesian_step),
        moveit::core::JumpThreshold::disabled(), valid);

      if (achieved_m < 0.99 * wanted_m || path.size() < 2) {
        RCLCPP_WARN(
          log_, "Cartesian %s achieved %.4f m of %.4f m (%.0f%%)", motionName(delta), achieved_m,
          wanted_m, 100.0 * achieved_m / wanted_m);
        diagnoseCartesianFailure(scene, robot, start, delta, std::nullopt, acm);
        return false;
      }
      return true;
    }

    // Pose target: this overload DOES return the fraction. The step bound is the same
    // `cartesian_step` in translation, and MoveIt's 3.5 x that in rotation (1 degree per
    // 5 mm), so a bend of the torch is subdivided as finely as the travel is.
    Eigen::Isometry3d target = state->getGlobalLinkTransform(link);
    target.translation() += delta;
    target.linear() = rotation->normalized().toRotationMatrix();
    const double fraction = moveit::core::CartesianInterpolator::computeCartesianPath(
      state.get(), arm, path, link, target, /*global_reference_frame=*/true,
      moveit::core::MaxEEFStep(spec_.planning.cartesian_step),
      moveit::core::JumpThreshold::disabled(), valid);
    if (fraction < 0.99 || path.size() < 2) {
      RCLCPP_WARN(
        log_, "Cartesian %s (turning the tool) achieved %.0f%% of %.4f m", motionName(delta),
        100.0 * fraction, wanted_m);
      diagnoseCartesianFailure(scene, robot, start, delta, rotation, acm);
      return false;
    }
    return true;
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
    const std::optional<Eigen::Quaterniond> & rotation,
    const collision_detection::AllowedCollisionMatrix & acm)
  {
    const auto * arm = model_->getJointModelGroup(robot.arm_group);
    const auto * link = model_->getLinkModel(robot.ee_link);

    auto state = std::make_shared<moveit::core::RobotState>(stateInScene(scene, start));
    std::vector<std::shared_ptr<moveit::core::RobotState>> path;
    const double wanted_m = delta.norm();
    double geometric_m = 0.0;
    if (!rotation) {
      geometric_m = moveit::core::CartesianInterpolator::computeCartesianPath(
        state.get(), arm, path, link, delta, true,
        moveit::core::MaxEEFStep(spec_.planning.cartesian_step),
        moveit::core::JumpThreshold::disabled());   // no validity callback: IK only
    } else {
      Eigen::Isometry3d target = state->getGlobalLinkTransform(link);
      target.translation() += delta;
      target.linear() = rotation->normalized().toRotationMatrix();
      geometric_m = wanted_m * moveit::core::CartesianInterpolator::computeCartesianPath(
        state.get(), arm, path, link, target, true,
        moveit::core::MaxEEFStep(spec_.planning.cartesian_step),
        moveit::core::JumpThreshold::disabled());
    }

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

    // Cartesian speed cap (`max_link_speed`, off unless the scene sets it). TOTG bounds each
    // joint's speed, not how fast the geometry moves: on a long arm swinging its shoulder
    // with the wrist also saturated, a fingertip covered 0.036 m in one 25 ms slot at
    // vel_scale 0.45 (fabricator, 2026-09-26) -- against 20 mm plates, which is exactly what
    // ADR-0006 forbids. Scaling vel_scale down for everyone would slow every motion to fix
    // the few fast ones. So a segment whose fastest link exceeds the cap is re-timed as a
    // whole, uniformly slower by the ratio: same path, times x f, velocities / f -- the
    // Hermite resampling reproduces the scaled profile exactly. `validateSamples` still
    // measures the resampled result, so the cap is a construction, not the proof.
    const double cap = spec_.planning.max_link_speed;
    if (cap > 0.0) {
      const double peak = linkSpeedPeak(rt);
      if (peak > cap) {
        const double f = peak / cap;
        for (auto & wp : out.waypoints) {
          wp.time_from_start *= f;
          for (auto & v : wp.velocities) {v /= f;}
        }
        ++capped_segments_;
      }
    }
    return true;
  }

  /// The largest Cartesian speed of any link the group moves (with geometry), over a timed
  /// trajectory: at each waypoint from the joint velocities (a finite step of 1 ms along
  /// q-dot), and between waypoints from the chord -- whichever is larger.
  double linkSpeedPeak(const robot_trajectory::RobotTrajectory & rt) const
  {
    const auto * jmg = rt.getGroup();
    const auto & links = jmg->getUpdatedLinkModelsWithGeometry();
    constexpr double h = 1e-3;
    double peak = 0.0;
    std::vector<Eigen::Vector3d> prev;
    for (std::size_t k = 0; k < rt.getWayPointCount(); ++k) {
      moveit::core::RobotState a(rt.getWayPoint(k));
      a.update();
      std::vector<double> q, qd;
      a.copyJointGroupPositions(jmg, q);
      a.copyJointGroupVelocities(jmg, qd);
      moveit::core::RobotState b(a);
      for (std::size_t j = 0; j < q.size() && j < qd.size(); ++j) {q[j] += h * qd[j];}
      b.setJointGroupPositions(jmg, q);
      b.update();
      std::vector<Eigen::Vector3d> now;
      for (const auto * l : links) {
        const Eigen::Vector3d pa = a.getGlobalLinkTransform(l).translation();
        peak = std::max(peak, (b.getGlobalLinkTransform(l).translation() - pa).norm() / h);
        now.push_back(pa);
      }
      const double dt = rt.getWayPointDurationFromPrevious(k);
      if (k > 0 && dt > 1e-9) {
        for (std::size_t i = 0; i < now.size(); ++i) {
          peak = std::max(peak, (now[i] - prev[i]).norm() / dt);
        }
      }
      prev = std::move(now);
    }
    return peak;
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
  ///     home -> pre-start (seam start, raised)          Phase::ToPick
  ///          -> Cartesian descent onto the seam         Phase::ToPick
  ///          -> arc-strike dwell                        Phase::ProcessOn
  ///          -> traverse, waypoint to waypoint          Phase::Processing
  ///          -> arc-out dwell                           Phase::ProcessOff
  ///          -> Cartesian retreat off the seam          Phase::ToHome
  ///          -> home                                    Phase::ToHome
  ///
  /// The seam is a polyline (`path:`; `start`/`end` is its two-point case), run as one
  /// Processing segment at the process speed -- `planProcessLeg`. The milestones are
  /// untouched by it (ADR-0009): the strike dwell opens the process, the cut closes it.
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
    // gives the world EE pose. The robot's own `weld_tool`, when it has one, wins
    // over the seam's `tool` (`loadTaskSpec` guarantees one of the two exists).
    const auto & tool = robot.weld_tool ? *robot.weld_tool : task.tool;
    // One EE polyline per leg (a `path:` seam is one leg); `seam_ee` is the first.
    std::vector<std::vector<geometry_msgs::msg::Pose>> legs_ee;
    for (const auto & leg : task.legs) {
      legs_ee.emplace_back();
      for (const auto & wp : leg) {legs_ee.back().push_back(compose(wp, tool));}
    }
    const auto & seam_ee = legs_ee.front();
    const auto & start_ee = seam_ee.front();
    const double a = task.approach;

    // The process allowance (`processAcm`) covers the descent onto the seam, the process
    // and the retreat off it; `contact` flags those segments for `validateSamples`. With
    // nothing to allow it is the scene's ACM and no segment is flagged -- as before.
    const bool contact = hasProcessAllowance(robot, task);
    const auto process_acm = processAcm(scene, robot, task);

    moveit::core::RobotState home(scene->getCurrentState());   // both robots parked
    moveit::core::RobotState start(from);
    start.update();

    if (scene->isStateColliding(start, robot.planning_group)) {
      RCLCPP_WARN(
        log_, "the start configuration collides in this task's environment -- the previous "
        "task left the arm somewhere %s cannot legally begin from", task.id.c_str());
      return false;
    }

    // Which way the tool comes onto the seam and leaves it: straight down in world z (the
    // default, as for a grasp), or -- `weld_approach: tool` -- along the tool's own axis,
    // which for a torch at 45 degrees into a fillet is the line the wire points along and
    // the only one that does not drag the nozzle sideways across a flange. The axis is taken
    // at the first waypoint for the approach and at the last for the retreat.
    auto toolAxis = [&](const geometry_msgs::msg::Pose & ee) {
        Eigen::Isometry3d t;
        tf2::fromMsg(ee, t);
        return Eigen::Vector3d(t.linear() * robot.tool_approach_axis);
      };
    const Eigen::Vector3d in_dir = robot.weld_approach_along_tool ?
      toolAxis(start_ee) : Eigen::Vector3d(0, 0, -1);
    const Eigen::Vector3d out_dir = robot.weld_approach_along_tool ?
      Eigen::Vector3d(-toolAxis(legs_ee.back().back())) : Eigen::Vector3d(0, 0, 1);
    geometry_msgs::msg::Pose pre_pose = start_ee;
    pre_pose.position.x -= a * in_dir.x();
    pre_pose.position.y -= a * in_dir.y();
    pre_pose.position.z -= a * in_dir.z();

    // How much of the descent and of the retreat the process allowance covers: the last
    // (first) `process_contact_depth` metres next to the seam, not the whole leg -- a nozzle
    // may skim the plate it welds, it may not sink through it on the way down. The leg is then
    // two Cartesian steps, checked in full above the depth and with the allowance below it.
    // Without an allowance, or with a depth reaching the whole approach, it stays ONE step,
    // flagged exactly as before.
    // The split is per leg: `len` is the length of THAT leg (the approach, or a leg's
    // `leg_retreat` between two spots of a tack).
    //
    // One straight leg of `len` metres along `dir`, split at `depth` from the seam end of it
    // (`seam_first`: the seam is where the leg STARTS -- the retreat).
    auto straightLegOf = [&](
      double len, const moveit::core::RobotState & from_state, const Eigen::Vector3d & dir,
      bool seam_first, Phase phase, std::vector<Segment> & out,
      moveit::core::RobotState & to_state) {
        const double depth = contact ? std::min(len, spec_.planning.process_contact_depth) : len;
        const bool split = contact && depth < len - 1e-9;
        if (!split) {
          Segment seg;
          if (!planCartesianStep(
              scene, robot, from_state, len * dir, phase, seg, to_state, process_acm, contact))
          {
            return false;
          }
          out.push_back(seg);
          return true;
        }
        const collision_detection::AllowedCollisionMatrix & plain =
          scene->getAllowedCollisionMatrix();
        const double lens[2] = {seam_first ? depth : len - depth, seam_first ? len - depth : depth};
        const bool near[2] = {seam_first, !seam_first};
        moveit::core::RobotState at(from_state);
        for (int p = 0; p < 2; ++p) {
          Segment seg;
          moveit::core::RobotState next(at);
          if (!planCartesianStep(
              scene, robot, at, lens[p] * dir, phase, seg, next,
              near[p] ? process_acm : plain, near[p]))
          {
            return false;
          }
          out.push_back(seg);
          at = next;
        }
        to_state = at;
        return true;
      };
    auto straightLeg = [&](
      const moveit::core::RobotState & from_state, const Eigen::Vector3d & dir, bool seam_first,
      Phase phase, std::vector<Segment> & out, moveit::core::RobotState & to_state) {
        return straightLegOf(a, from_state, dir, seam_first, phase, out, to_state);
      };

    // Between two legs of a tack (`legs:`), the arc is OFF and the torch moves to the next
    // spot: off the joint along the tool axis by `leg_retreat` (the reverse of how it came
    // on), across to the same stand-off above the next spot -- turning to its attitude, with
    // the rotation about the tool axis the first leg was started with (`held`) -- and back
    // on along the tool axis. All three Cartesian with the rail pinned, as every Cartesian
    // leg is, so the robot's rail stays where the pre-start IK put it for the whole task.
    // Phase::ToPick: hands empty, arc off, on the way to a seam start (the executors strike
    // the arc at each ProcessOn). The near-seam `process_contact_depth` of the retreat and of
    // the re-approach carries the process allowance, exactly like the first approach.
    auto transferTo = [&](
      const moveit::core::RobotState & from_state, const std::vector<geometry_msgs::msg::Pose> & prev_leg,
      const std::vector<geometry_msgs::msg::Pose> & next_leg, const Eigen::Matrix3d & held,
      std::vector<Segment> & out, moveit::core::RobotState & to_state) {
        const double r = task.leg_retreat;
        const Eigen::Vector3d off_dir = robot.weld_approach_along_tool ?
          Eigen::Vector3d(-toolAxis(prev_leg.back())) : Eigen::Vector3d(0, 0, 1);
        const Eigen::Vector3d on_dir = robot.weld_approach_along_tool ?
          toolAxis(next_leg.front()) : Eigen::Vector3d(0, 0, -1);
        moveit::core::RobotState lifted(from_state);
        if (!straightLegOf(r, from_state, off_dir, true, Phase::ToPick, out, lifted)) {return false;}

        const auto * link = model_->getLinkModel(robot.ee_link);
        Eigen::Isometry3d next_start;
        tf2::fromMsg(next_leg.front(), next_start);
        const Eigen::Vector3d here = stateInScene(scene, lifted).getGlobalLinkTransform(link).translation();
        const Eigen::Vector3d delta = (next_start.translation() - r * on_dir) - here;
        const Eigen::Matrix3d now_R = stateInScene(scene, lifted).getGlobalLinkTransform(link).linear();
        const Eigen::Matrix3d want_R = next_start.linear() * held;
        std::optional<Eigen::Quaterniond> turn;
        if (Eigen::AngleAxisd(now_R.transpose() * want_R).angle() > 1e-6) {
          turn = Eigen::Quaterniond(want_R);
        }
        moveit::core::RobotState over(lifted);
        if (delta.norm() > 1e-6 || turn) {
          std::vector<std::shared_ptr<moveit::core::RobotState>> path;
          if (!cartesianPath(scene, robot, lifted, delta, turn, scene->getAllowedCollisionMatrix(), path)) {
            RCLCPP_WARN(log_, "  ... on the arc-off transfer between two legs");
            return false;
          }
          robot_trajectory::RobotTrajectory rt(model_, model_->getJointModelGroup(robot.planning_group));
          for (const auto & st : path) {rt.addSuffixWayPoint(*st, 0.0);}
          if (!timeParameterise(rt)) {return false;}
          Segment seg;
          if (!toSegment(rt, Phase::ToPick, seg)) {return false;}
          out.push_back(seg);
          over = *path.back();
        }
        return straightLegOf(r, over, on_dir, false, Phase::ToPick, out, to_state);
      };

    // --- fly out to the seam start, raised ----------------------------------- #
    // The pre-start candidates: one for a robot whose tool attitude is fixed (`ikTo`, as
    // always), or every collision-free rotation about the tool axis for one that leaves it
    // free (`tool_axis_free`), nearest the start configuration first. A candidate whose
    // flight, descent, seam or retreat cannot be planned hands over to the next: which
    // rotation lets the swan neck clear a flange is only known once the seam has been run.
    const auto candidates = ikCandidates(scene, robot, pre_pose, start);
    if (candidates.empty()) {
      RCLCPP_WARN(log_, "no collision-free IK for the pre-start pose");
      return false;
    }

    for (std::size_t c = 0; c < candidates.size(); ++c) {
      if (timedOut()) {
        RCLCPP_WARN(log_, "  pair time cap reached: pre-start candidates %zu..%zu not tried", c + 1,
                    candidates.size());
        return false;
      }
      scene->getCurrentStateNonConst() = home;       // undo a failed candidate's retreat state
      std::vector<Segment> cand;
      const moveit::core::RobotState & pre_start = candidates[c];
      if (candidates.size() > 1) {
        RCLCPP_INFO(
          log_, "  pre-start candidate %zu of %zu (joint distance %.3f from the start)", c + 1,
          candidates.size(), start.distance(pre_start, jmg));
      }

      Segment s1;
      if (!planJoint(scene, robot, start, pre_start, Phase::ToPick, s1)) {continue;}
      cand.push_back(s1);

      moveit::core::RobotState at_start(pre_start);
      if (!straightLeg(pre_start, in_dir, false, Phase::ToPick, cand, at_start)) {continue;}

      // --- strike the arc: arm frozen, slots consumed ------------------------ #
      std::vector<double> start_q;
      at_start.copyJointGroupPositions(jmg, start_q);
      cand.push_back(
        mrct::makeDwell(
          start_q, Phase::ProcessOn, spec_.disc.process_dwell_slots, spec_.disc.delta_t));
      cand.back().support_contact = contact;

      // --- run the seam, waypoint to waypoint, at the process speed ---------- #
      Segment s3;
      moveit::core::RobotState at_end(at_start);
      if (!planProcessLeg(
          scene, robot, at_start, seam_ee, task.speed, process_acm, s3, at_end))
      {
        continue;
      }
      s3.support_contact = contact;
      cand.push_back(s3);

      // --- cut the arc ------------------------------------------------------- #
      std::vector<double> end_q;
      at_end.copyJointGroupPositions(jmg, end_q);
      cand.push_back(
        mrct::makeDwell(
          end_q, Phase::ProcessOff, spec_.disc.process_dwell_slots, spec_.disc.delta_t));
      cand.back().support_contact = contact;

      // --- further legs (`legs:`): arc off, across, strike, run, cut, per leg -- #
      // The rotation about the tool axis the first leg was started with, held for every leg.
      Eigen::Isometry3d first_nominal;
      tf2::fromMsg(seam_ee.front(), first_nominal);
      const Eigen::Matrix3d held = first_nominal.linear().transpose() *
        stateInScene(scene, at_start).getGlobalLinkTransform(
        model_->getLinkModel(robot.ee_link)).linear();
      bool legs_ok = true;
      for (std::size_t k = 1; k < legs_ee.size() && legs_ok; ++k) {
        moveit::core::RobotState on_leg(at_end);
        if (!transferTo(at_end, legs_ee[k - 1], legs_ee[k], held, cand, on_leg)) {
          legs_ok = false;
          break;
        }
        std::vector<double> q;
        on_leg.copyJointGroupPositions(jmg, q);
        cand.push_back(
          mrct::makeDwell(q, Phase::ProcessOn, spec_.disc.process_dwell_slots, spec_.disc.delta_t));
        cand.back().support_contact = contact;
        Segment leg;
        moveit::core::RobotState leg_end(on_leg);
        if (!planProcessLeg(
            scene, robot, on_leg, legs_ee[k], task.speed, process_acm, leg, leg_end))
        {
          legs_ok = false;
          break;
        }
        leg.support_contact = contact;
        cand.push_back(leg);
        leg_end.copyJointGroupPositions(jmg, q);
        cand.push_back(
          mrct::makeDwell(q, Phase::ProcessOff, spec_.disc.process_dwell_slots, spec_.disc.delta_t));
        cand.back().support_contact = contact;
        at_end = leg_end;
      }
      if (!legs_ok) {
        RCLCPP_WARN(log_, "  a later leg of %s could not be planned from this candidate",
                    task.id.c_str());
        continue;
      }

      // --- retreat and (maybe) go home --------------------------------------- #
      scene->getCurrentStateNonConst() = at_end;

      moveit::core::RobotState retreated(at_end);
      if (!straightLeg(at_end, out_dir, true, Phase::ToHome, cand, retreated)) {continue;}

      if (to_home) {
        Segment s5;
        if (!planJoint(scene, robot, retreated, home, Phase::ToHome, s5)) {continue;}
        cand.push_back(s5);
        end = home;
      } else {
        end = retreated;
      }
      segs.insert(segs.end(), cand.begin(), cand.end());
      return true;
    }
    return false;
  }

  /// The process leg of a weld: from the seam's first waypoint through every other one, as
  /// ONE Phase::Processing segment timed at the process speed.
  ///
  /// One Cartesian line per polyline segment (`cartesianPath`, rail pinned as for every
  /// Cartesian leg), the EE orientation slerped from each waypoint's to the next's; then the
  /// concatenated path is timed by arc length at `speed` in one go
  /// (`timeParameteriseAtSpeed`), so the tool keeps the process speed THROUGH the corners
  /// and the leg lasts (polyline length) / `speed`. A two-point seam is one line and exactly
  /// the traverse this function replaced.
  ///
  /// The targets are the nominal waypoints `seam_ee` moved by where the leg actually
  /// starts: in position by the offset between `start` and `seam_ee[0]` (IK tolerance), in
  /// orientation by the half-turn `ikTo` may have chosen about the tool axis (the grasp
  /// symmetry, a free rotation about the wire for a torch too). Expressing waypoint k as
  /// `R_k R_0^T R_start` keeps that choice for the whole leg instead of spinning the wrist
  /// half a turn back mid-seam. Positions are absolute, so a segment that stops within the
  /// 1 % the interpolator is allowed does not carry its shortfall into the next.
  bool planProcessLeg(
    const planning_scene::PlanningScenePtr & scene, const RobotCfg & robot,
    const moveit::core::RobotState & start,
    const std::vector<geometry_msgs::msg::Pose> & seam_ee, double speed,
    const collision_detection::AllowedCollisionMatrix & acm, Segment & out,
    moveit::core::RobotState & end_state)
  {
    const auto * full = model_->getJointModelGroup(robot.planning_group);
    const auto * link = model_->getLinkModel(robot.ee_link);

    auto nominal = [](const geometry_msgs::msg::Pose & p) {
        Eigen::Isometry3d t;
        tf2::fromMsg(p, t);
        return t;
      };
    const Eigen::Isometry3d at_start = stateInScene(scene, start).getGlobalLinkTransform(link);
    const Eigen::Isometry3d first = nominal(seam_ee.front());
    // The rotation that takes the nominal first attitude to the one actually held.
    const Eigen::Matrix3d held = first.linear().transpose() * at_start.linear();

    std::vector<std::shared_ptr<moveit::core::RobotState>> leg;
    moveit::core::RobotState from(start);
    for (std::size_t k = 1; k < seam_ee.size(); ++k) {
      const Eigen::Isometry3d prev = nominal(seam_ee[k - 1]);
      const Eigen::Isometry3d next = nominal(seam_ee[k]);
      const Eigen::Vector3d here =
        stateInScene(scene, from).getGlobalLinkTransform(link).translation();
      // = next + (at_start - first) - here, grouped so the first line is EXACTLY
      // `next - first` (the second bracket is 0.0): the two-point traverse reproduces the
      // old delta bit for bit.
      const Eigen::Vector3d delta =
        (next.translation() - first.translation()) - (here - at_start.translation());

      // Hold the attitude when the two waypoints share it -- always so for a two-point seam
      // written as `start`/`end`, whose points carry no rotation: that is the translation
      // interpolation every scene before polylines ran, unchanged. Otherwise turn to it.
      std::optional<Eigen::Quaterniond> turn;
      if (Eigen::AngleAxisd(prev.linear().transpose() * next.linear()).angle() > 1e-9) {
        turn = Eigen::Quaterniond(next.linear() * held);
      }

      std::vector<std::shared_ptr<moveit::core::RobotState>> path;
      if (!cartesianPath(scene, robot, from, delta, turn, acm, path)) {
        if (seam_ee.size() > 2) {
          RCLCPP_WARN(
            log_, "  ... on segment %zu of %zu of the seam", k, seam_ee.size() - 1);
        }
        return false;
      }
      // Each line starts with its own start state, which is the previous line's last.
      leg.insert(leg.end(), leg.empty() ? path.begin() : std::next(path.begin()), path.end());
      from = *path.back();
    }

    robot_trajectory::RobotTrajectory rt(model_, full);
    for (const auto & s : leg) {rt.addSuffixWayPoint(*s, 0.0);}
    if (!timeParameteriseAtSpeed(rt, robot, speed)) {return false;}

    end_state = *leg.back();
    return toSegment(rt, Phase::Processing, out);
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
      if (timedOut()) {return false;}
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
    const double a = task.approach;   // `planning.approach` unless the task sets its own

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

    std::vector<double> place_q;
    placed.copyJointGroupPositions(jmg, place_q);

    // (A pick-place that HOLDS its part for a `hold` precedence gets its hold -- frozen
    // Carrying samples before GripOpen -- inserted into the resampled trajectory by `run`,
    // `insertHoldTail`, once the held process is planned: see `run`. No new Phase, so the
    // executors, the gripper and the resampler see an ordinary carry.)

    // --- open the gripper ---------------------------------------------------- #
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

  // ---- the eligibility gate (`planning.eligibility: auto`) ------------------- #

  /// Can `robot` reach `task`'s key poses at all, in the task's own world? Collision-aware
  /// IK, before any planning, so a hopeless pair costs a few IK calls instead of every
  /// planning attempt. `why` names the first pose that failed. Reuses the planner's own IK
  /// (`ikTo`, `ikCandidates`) and its pose conventions; it is a NECESSARY condition only --
  /// planning stays the judge -- and so it is optimistic where it must guess (the part is
  /// left out of the world at the grasp and the place, where the gripper holds it).
  bool gatePair(const RobotCfg & robot, const TaskDef & task, std::string & why)
  {
    auto scene = sceneFor(robot, task);
    moveit::core::RobotState home(scene->getCurrentState());
    if (task.kind == TaskKind::PickPlace) {
      const ObjectDef & obj = spec_.object(task.object_id);
      setObjectState(scene, robot, task, ObjectState::AtSpawn);
      const auto pick_ee = compose(obj.spawn, obj.grasp_in_obj);
      const auto place_ee = compose(task.place, obj.grasp_in_obj);
      const double a = task.approach;
      moveit::core::RobotState q(home);
      if (!ikTo(scene, robot, raised(pick_ee, a), home, q)) {why = "IK pre-pick"; return false;}
      // the part itself out of the world: at the grasp the fingers straddle it, and at the
      // pre-place and the place it is in the hand
      auto bare = scene->diff();
      moveit_msgs::msg::CollisionObject rm;
      rm.id = obj.id;
      rm.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      bare->processCollisionObjectMsg(rm);
      moveit::core::RobotState g(q);
      if (!ikTo(bare, robot, pick_ee, q, g)) {why = "IK pick (grasp)"; return false;}
      moveit::core::RobotState pp(g);
      if (!ikTo(bare, robot, raised(place_ee, a), g, pp)) {why = "IK pre-place"; return false;}
      moveit::core::RobotState pl(pp);
      if (!ikTo(bare, robot, place_ee, pp, pl)) {why = "IK place"; return false;}
      return true;
    }
    // A weld: the pre-start (every rotation about the tool axis a torch may use), then every
    // waypoint of every leg with the rail PINNED where that pre-start put it -- as the weld
    // runs -- the rotation held along the legs, each IK seeded from the previous waypoint.
    const auto & tool = robot.weld_tool ? *robot.weld_tool : task.tool;
    std::vector<std::vector<geometry_msgs::msg::Pose>> legs_ee;
    for (const auto & leg : task.legs) {
      legs_ee.emplace_back();
      for (const auto & wp : leg) {legs_ee.back().push_back(compose(wp, tool));}
    }
    const auto & start_ee = legs_ee.front().front();
    Eigen::Isometry3d first;
    tf2::fromMsg(start_ee, first);
    const Eigen::Vector3d in_dir = robot.weld_approach_along_tool ?
      Eigen::Vector3d(first.linear() * robot.tool_approach_axis) : Eigen::Vector3d(0, 0, -1);
    geometry_msgs::msg::Pose pre_pose = start_ee;
    pre_pose.position.x -= task.approach * in_dir.x();
    pre_pose.position.y -= task.approach * in_dir.y();
    pre_pose.position.z -= task.approach * in_dir.z();
    const auto candidates = ikCandidates(scene, robot, pre_pose, home);
    if (candidates.empty()) {why = "IK pre-start"; return false;}

    const auto * arm = model_->getJointModelGroup(robot.arm_group);
    const auto * link = model_->getLinkModel(robot.ee_link);
    const auto acm = processAcm(scene, robot, task);
    auto valid = [&](moveit::core::RobotState * st, const moveit::core::JointModelGroup * g,
        const double * values) {
        st->setJointGroupPositions(g, values);
        st->update();
        return !collides(scene, *st, robot.planning_group, acm);
      };
    std::string furthest = "IK leg 1 waypoint 1";
    std::size_t best = 0;
    for (const auto & pre : candidates) {
      moveit::core::RobotState st = stateInScene(scene, pre);
      const Eigen::Matrix3d held =
        first.linear().transpose() * st.getGlobalLinkTransform(link).linear();
      bool ok = true;
      std::size_t n_done = 0;
      for (std::size_t L = 0; L < legs_ee.size() && ok; ++L) {
        for (std::size_t w = 0; w < legs_ee[L].size(); ++w) {
          Eigen::Isometry3d target;
          tf2::fromMsg(legs_ee[L][w], target);
          target.linear() = target.linear() * held;
          if (!st.setFromIK(arm, target, robot.ee_link, 0.5, valid)) {
            ok = false;
            if (n_done >= best) {
              best = n_done;
              furthest = "IK leg " + std::to_string(L + 1) + " waypoint " + std::to_string(w + 1) +
                " (rail pinned, " + std::to_string(candidates.size()) + " pre-start candidate(s))";
            }
            break;
          }
          ++n_done;
        }
      }
      if (ok) {return true;}
    }
    why = furthest;
    return false;
  }

  /// Past the current pair's wall-time cap (`planning.pair_timeout_s`, auto mode)?
  bool timedOut() const
  {
    return pair_deadline_ && std::chrono::steady_clock::now() > *pair_deadline_;
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

      // The contact allowance follows the segment the sample was taken from: the support
      // allowance of a pick-and-place, the process allowance of a weld (`contactAcm`).
      const auto acm = traj.support_contact[k] ?
        contactAcm(scene, robot, task) : supportAcm(scene, std::string{});   // copies
      // Fingers (when modelled): on the GripClose and GripOpen dwells they are moving between
      // open and closed, so the sample must be clear along the sweep: open, half-way and
      // closed (the pads follow a short arc -- sagitta ~2 mm over the 0.5 rad -- so its
      // midpoint is checked too, not only the ends).
      bool finger_hit = false;
      if (fingersModelled(robot) &&
        (traj.phase[k] == Phase::GripClose || traj.phase[k] == Phase::GripOpen))
      {
        for (const double f : {0.0, 0.5, 1.0}) {
          moveit::core::RobotState other(state);
          const double v = robot.gripper_open + f * (robot.gripper_close - robot.gripper_open);
          other.setJointPositions(robot.gripper_joint, &v);
          other.update();
          if (collides(scene, other, robot.planning_group, acm)) {
            finger_hit = true;
            if (bad == 0) {
              RCLCPP_WARN(log_, "  sample %zu: in collision with the fingers at %.3f", k, v);
            }
            break;
          }
        }
      }
      if (finger_hit || collides(scene, state, robot.planning_group, acm)) {
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
          const double step = (now[i] - prev[i]).norm();
          if (step > max_cartesian_step) {
            max_cartesian_step = step;
            // Which link, where: the number alone does not say what to slow down.
            fastest_link_ = links[i]->getName();
            fastest_sample_ = k;
            fastest_phase_ = mrct::phaseName(traj.phase[k]);
          }
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
    // The hold tail of a pick-place in a `hold` precedence: its last `hold_slots` samples
    // before GripOpen hold the part at the place pose. Only when there is one, so every
    // artifact without a hold is written as before.
    if (task.kind == TaskKind::PickPlace) {
      if (const auto h = hold_slots_.find(task.id); h != hold_slots_.end() && h->second > 0) {
        o << "      \"hold_slots\": " << h->second << ",\n";
      }
    }
    // A HELD process: planned and validated against its handler standing at the hold
    // configuration (`sceneFor`). The collision stages rely on it to exempt the hold from mu
    // for this pair: the clearance is certified here, on the exact geometry.
    if (const auto hp = hold_pose_.find(task.id);
      hp != hold_pose_.end() && hp->second.robot != robot.name)
    {
      o << "      \"held_by\": \"" << hp->second.robot << "|" << hp->second.task << "\",\n";
    }
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

    // auto eligibility: what became of every candidate pair (planned, or why it was dropped).
    // Only in auto mode, so every strict artifact is byte-identical to before.
    if (spec_.planning.eligibility_auto && chains_.empty()) {
      f << "  \"eligibility\": {\n";
      bool first_t = true;
      for (const auto & t : spec_.tasks) {
        const auto it = eligibility_.find(t.id);
        if (it == eligibility_.end() || it->second.empty()) {continue;}
        f << (first_t ? "" : ",\n") << "    \"" << t.id << "\": {";
        bool first_r = true;
        for (const auto & [r, st] : it->second) {
          f << (first_r ? "" : ", ") << "\"" << r << "\": \"" << st << "\"";
          first_r = false;
        }
        f << "}";
        first_t = false;
      }
      f << "\n  },\n";
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
  std::vector<std::string> only_;                            // `only_pairs`, empty: all
  // Pick-place id -> slots of its hold tail (`run` sizes it; empty without a `hold`).
  std::map<std::string, int> hold_slots_;
  // Held process id -> the handler that holds its part, where: the world it is planned in.
  struct HoldPose
  {
    std::string robot;               ///< the holding robot
    std::string slot;                ///< the holding pick-place's slot (its `place__` box)
    std::vector<double> q;           ///< its planning-group configuration at the place pose
    std::string task;                ///< the holding pick-place
  };
  std::map<std::string, HoldPose> hold_pose_;
  // auto eligibility: task -> robot -> planned | gate: <why> | failed: <why> (the artifact's
  // `eligibility` block), and the current pair's wall-time deadline.
  std::map<std::string, std::map<std::string, std::string>> eligibility_;
  std::optional<std::chrono::steady_clock::time_point> pair_deadline_;
  // Where the last `validateSamples` measured its largest per-slot link travel (log only).
  std::string fastest_link_, fastest_phase_;
  std::size_t fastest_sample_{0};
  // Segments `toSegment` slowed down to `max_link_speed` (log only).
  long capped_segments_{0};
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
    std::string only_pairs;
    node->get_parameter_or("only_pairs", only_pairs, std::string{});
    gen.setOnly(only_pairs);
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
