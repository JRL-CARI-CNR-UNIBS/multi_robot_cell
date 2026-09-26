// Offline inter-robot collision matrices -> forbidden start-offset sets.
//
// Reads the trajectory artifact, computes for every pair of trajectories belonging
// to DIFFERENT robots the boolean matrix
//
//     mu[k,l] = 1  iff  robot r at sample k of task i collides with
//                       robot s at sample l of task j
//
// and reduces it -- exactly, not conservatively -- to the forbidden relative start
// offsets D_(ri)(sj) = { k - l : mu[k,l] = 1 } (ADR-0001). The output is the
// geometry-free SchedulingProblem: durations, forbidden offsets, precedences. No
// robot model, no poses, no meshes cross this line.
//
// WHAT mu MUST CONTAIN, AND WHAT IT MUST NOT
//
// It must contain ONLY robot-vs-robot. Robot-vs-environment and robot self-collision
// were already settled in the trajectory stage: every trajectory was planned and then
// re-validated against its own environment E_i. Re-checking them here would be waste,
// and worse, it would make mu[k,l] true for reasons that have nothing to do with the
// two robots' relative timing -- which is precisely what the offsets encode.
//
// But it MUST contain the CARRIED OBJECT. While robot r ferries its box across the
// table, that box is part of r's geometry and can hit robot s. Nothing else covers
// this: s's trajectory was planned against a STATIC environment in which the box sits
// at its spawn or place pose, never at the intermediate poses it actually occupies
// while being carried. Two carried objects can also hit each other. Miss this and the
// schedule is unsound in exactly the way the whole method claims it is not.
//
// The ACM is therefore masked so that the ONLY pairs checked are
//     (robot r's links + its attached object) x (robot s's links + its attached object)
// and everything else -- self pairs, world pairs -- is allowed.
//
// COST
//
// K ~ 750, 4 tasks, 2 robots => 16 trajectory pairs x ~570k config pairs ~ 9.2M checks.
// A full FCL check on each would take ~15-20 min. Almost all of them are trivially
// far apart, so a bounding-sphere broad phase (one sphere per robot per sample,
// computed once from FK) rejects the vast majority for the cost of a distance
// comparison, and FCL only runs on the survivors.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>

#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <multi_robot_cell_tamp/resample.hpp>

// JSON is a subset of YAML, and yaml-cpp is already a dependency -- so the
// trajectory artifact is parsed with it rather than dragging in nlohmann/json
// (which is not installed and would need root to add).
#include <yaml-cpp/yaml.h>

namespace
{

constexpr int AT_SPAWN = 0;
constexpr int ATTACHED = 1;
constexpr int AT_PLACE = 2;

// Fixed-point scale for the seam's `transit_distances` -- MUST match
// tamp_scheduler/model.py's `TRANSIT_SCALE` and collision_generator_vamp.py's own
// copy exactly, and MIRRORS collision_generator_vamp.py's `TRANSIT_SCALE`.
constexpr double TRANSIT_SCALE = 1000.0;

struct Traj
{
  std::string robot;
  std::string task;
  std::string object;
  std::vector<std::string> joint_names;
  std::vector<std::vector<double>> q;      // K x 7
  std::vector<int> object_state;           // K
  std::vector<int> phase;                  // K, values of mrct::Phase
  std::size_t K() const {return q.size();}
};

/// One robot's whole geometry at one sample, reduced to a bounding sphere.
/// Cheap to build (FK we need anyway) and it prunes ~all of the 9.2M pairs.
struct Bound
{
  Eigen::Vector3d centre;
  double radius;
};

/// A graspable object as mu sees it: its bounding box (`size`; a `mesh:` object is
/// modelled by this box too, which over-covers it) and where it rides while carried.
///
/// `ee_T_obj` is the object's pose in the robot's `ee_link` frame, `grasp^-1`, from the
/// task YAML's `objects[].grasp` (translation AND rotation). It is exactly the transform
/// trajectory_generator.cpp's `setObjectState(Attached)` fixes: that attaches the object
/// at its world pose while the EE sits at `object (x) grasp` (the IK target), so the
/// object's pose in the EE frame is `grasp^-1`. Before 2026-09-21 this was a fixed
/// {0,0,0.10} with identity orientation, disjoint from the real box (ADR-0005 addendum).
struct ObjectGeom
{
  std::array<double, 3> size;
  double radius;                            // half-diagonal: bounding sphere about the centre
  Eigen::Isometry3d ee_T_obj{Eigen::Isometry3d::Identity()};
};

/// A YAML pose {x,y,z,roll,pitch,yaw} with trajectory_generator.cpp's `poseFromYaml`
/// semantics (tf2 `setRPY`: fixed-axis roll, pitch, yaw, i.e. R = Rz * Ry * Rx).
Eigen::Isometry3d isoFromYaml(const YAML::Node & n)
{
  const auto get = [&](const char * k) {return n[k] ? n[k].as<double>() : 0.0;};
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.translation() = Eigen::Vector3d(n["x"].as<double>(), n["y"].as<double>(), n["z"].as<double>());
  t.linear() = (Eigen::AngleAxisd(get("yaw"), Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(get("pitch"), Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(get("roll"), Eigen::Vector3d::UnitX())).toRotationMatrix();
  return t;
}

// --------------------------------------------------------------------------- #
class CollisionGenerator
{
public:
  CollisionGenerator(
    const rclcpp::Node::SharedPtr & node, const YAML::Node & artifact,
    const std::string & task_yaml)
  : node_(node), log_(node->get_logger()), art_(artifact)
  {
    robot_model_loader::RobotModelLoader loader(node_, "robot_description");
    model_ = loader.getModel();
    if (!model_) {throw std::runtime_error("could not load `robot_description`");}

    scene_ = std::make_shared<planning_scene::PlanningScene>(model_);
    delta_t_ = art_["delta_t"].as<double>();
    for (const auto & r : art_["robots"]) {robots_.push_back(r.as<std::string>());}
    for (const auto & t : art_["tasks"]) {tasks_.push_back(t.as<std::string>());}
    if (robots_.size() != 2) {
      throw std::runtime_error("this stage assumes exactly two robots");
    }

    for (const auto & t : art_["trajectories"]) {
      Traj tr;
      tr.robot = t["robot"].as<std::string>();
      tr.task = t["task"].as<std::string>();
      tr.object = t["object"].as<std::string>();
      for (const auto & n : t["joint_names"]) {tr.joint_names.push_back(n.as<std::string>());}
      for (const auto & row : t["positions"]) {
        tr.q.push_back(row.as<std::vector<double>>());
      }
      for (const auto & o : t["object_state"]) {tr.object_state.push_back(o.as<int>());}
      for (const auto & p : t["phase"]) {tr.phase.push_back(p.as<int>());}
      trajs_[{tr.robot, tr.task}] = std::move(tr);
    }

    // Per-robot home configuration (7 DOF: rail + 6 joints), written unconditionally
    // by trajectory_generator.cpp. Needed for `transit_distances` below -- the seam's
    // own field, not geometry that crosses the ADR-0002 line (it is a scalar cost per
    // (robot, task), same as `durations`).
    for (const auto & kv : art_["homes"]) {
      homes_[kv.first.as<std::string>()] = kv.second.as<std::vector<double>>();
    }

    // Interchangeable slots, when the trajectory artifact carries them (only a
    // scene with a `slots:` block does). Read here for one reason -- deciding
    // which task pairs can never co-occur -- and otherwise passed through to the
    // seam verbatim, exactly like `precedence_modes`.
    if (art_["slot_of"]) {
      for (const auto & kv : art_["slot_of"]) {
        slot_of_[kv.first.as<std::string>()] = kv.second.as<std::string>();
      }
    }
    if (art_["object_of"]) {
      for (const auto & kv : art_["object_of"]) {
        object_of_[kv.first.as<std::string>()] = kv.second.as<std::string>();
      }
    }

    loadObjectGeometry(task_yaml);
    computeAttachFrames();
    buildLinkSets();
    buildAcm();
  }

  void run(const std::string & out_path)
  {
    // FK once per sample, for every trajectory. Everything downstream reads these.
    std::map<std::pair<std::string, std::string>, std::vector<Bound>> bounds;
    for (auto & [key, tr] : trajs_) {
      bounds[key] = computeBounds(tr);
    }

    const std::string & r = robots_[0];
    const std::string & s = robots_[1];

    std::map<std::string, std::vector<int>> forbidden;
    std::size_t total_pairs = 0, broad_survivors = 0, colliding_cells = 0;

    for (const auto & i : tasks_) {
      for (const auto & j : tasks_) {
        // Mutually exclusive candidates never run together, so there is no pair of
        // simultaneous samples to forbid an offset between (ADR-0003 addendum):
        // the solver runs exactly one candidate per slot and consumes each object
        // at most once. Skipping them is not just a saving -- attaching one object
        // id to BOTH robots at once is not something a MoveIt planning scene can
        // represent, and the second attach would silently move the first, making
        // mu under-report. No scene without `slots:` can enter this branch (the
        // maps are empty unless the trajectory artifact carries them), so every
        // existing seam is byte-identical.
        if (mutuallyExclusive(i, j)) {continue;}
        const Traj & ti = trajs_.at({r, i});
        const Traj & tj = trajs_.at({s, j});
        const auto & bi = bounds.at({r, i});
        const auto & bj = bounds.at({s, j});

        std::set<int> offsets;
        std::size_t survivors = 0, hits = 0;

        // The (k,l) grid partitions into blocks of constant attachment (each
        // object_state is one contiguous run). Configure the scene once per block
        // instead of re-attaching per cell -- 9.2M attach/detach cycles would dwarf
        // the collision checks themselves.
        for (const auto & [si, ki0, ki1] : runs(ti.object_state)) {
          for (const auto & [sj, kj0, kj1] : runs(tj.object_state)) {
            configureAttachments(r, ti, si, s, tj, sj);

            for (std::size_t k = ki0; k < ki1; ++k) {
              for (std::size_t l = kj0; l < kj1; ++l) {
                ++total_pairs;
                // Broad phase: two bounding spheres that do not touch cannot collide.
                const double d = (bi[k].centre - bj[l].centre).norm();
                if (d > bi[k].radius + bj[l].radius) {continue;}
                ++survivors;

                if (inCollision(r, ti, k, s, tj, l)) {
                  ++hits;
                  offsets.insert(static_cast<int>(k) - static_cast<int>(l));
                }
              }
            }
          }
        }

        broad_survivors += survivors;
        colliding_cells += hits;
        if (!offsets.empty()) {
          forbidden[r + "|" + i + "|" + s + "|" + j] =
            std::vector<int>(offsets.begin(), offsets.end());
        }
        RCLCPP_INFO(
          log_, "(%s,%s) x (%s,%s): %zu/%zu cells survived broad phase, %zu collide -> "
          "%zu forbidden offsets", r.c_str(), i.c_str(), s.c_str(), j.c_str(),
          survivors, ti.K() * tj.K(), hits, offsets.size());
      }
    }

    writeProblem(out_path, forbidden);

    RCLCPP_INFO(
      log_,
      "done: %zu config-pairs, %zu survived the broad phase (%.2f%%), %zu in collision. "
      "Written to %s",
      total_pairs, broad_survivors, 100.0 * static_cast<double>(broad_survivors) /
      static_cast<double>(std::max<std::size_t>(total_pairs, 1)), colliding_cells,
      out_path.c_str());
  }

  /// Test hook (`dump_object_centres:=<file>`): the carried object's WORLD pose as this
  /// engine's own planning scene holds it -- attached by `setAttached`, read back with
  /// its collision body's global transform -- at every 10th ATTACHED sample of every trajectory. The pose
  /// test in vamp_fcl_object_pose_test.py compares it against VAMP and the generator.
  void dumpObjectCentres(const std::string & path)
  {
    std::ofstream f(path);
    if (!f) {throw std::runtime_error("cannot write " + path);}
    f.precision(17);
    f << "[";
    bool first = true;
    for (const auto & [key, tr] : trajs_) {
      if (tr.object.empty()) {continue;}
      setAttached(tr.robot, tr.object, true);
      for (std::size_t k = 0; k < tr.K(); ++k) {
        if (tr.object_state[k] != ATTACHED || k % 10 != 0) {continue;}
        moveit::core::RobotState & state = scene_->getCurrentStateNonConst();
        setRobot(state, tr.robot, tr, k);
        state.update();
        // The box primitive's own world pose (the body's frame is the attach link;
        // the primitive pose inside it is what setAttached sets).
        const auto * body = state.getAttachedBody(tr.object);
        if (body == nullptr) {throw std::runtime_error("object not attached: " + tr.object);}
        const Eigen::Isometry3d w = body->getGlobalCollisionBodyTransforms().at(0);
        const Eigen::Quaterniond q(w.linear());
        f << (first ? "\n" : ",\n") << "{\"robot\": \"" << tr.robot << "\", \"task\": \""
          << tr.task << "\", \"k\": " << k << ", \"centre\": [" << w.translation().x() << ", "
          << w.translation().y() << ", " << w.translation().z() << "], \"quat_xyzw\": ["
          << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "]}";
        first = false;
      }
      setAttached(tr.robot, tr.object, false);
    }
    f << "\n]\n";
    RCLCPP_INFO(log_, "object centres written to %s", path.c_str());
  }

private:
  /// Can these two tasks ever both be in one plan?
  ///
  /// Empty maps (every scene without `slots:`) make this always false, which is
  /// exactly today's behaviour.
  bool mutuallyExclusive(const std::string & i, const std::string & j) const
  {
    const auto si = slot_of_.find(i), sj = slot_of_.find(j);
    if (si != slot_of_.end() && sj != slot_of_.end() && si->second == sj->second) {
      return true;
    }
    const auto oi = object_of_.find(i), oj = object_of_.find(j);
    return oi != object_of_.end() && oj != object_of_.end() && oi->second == oj->second;
  }

  /// The geometry-free SchedulingProblem. This file is the seam: durations,
  /// forbidden offsets, precedences -- and not one pose, mesh, or joint value.
  void writeProblem(const std::string & path, const std::map<std::string, std::vector<int>> & forbidden)
  {
    std::ofstream f(path);
    if (!f) {throw std::runtime_error("cannot write " + path);}
    f.precision(17);

    f << "{\n  \"delta_t\": " << delta_t_ << ",\n";

    f << "  \"robots\": [";
    for (std::size_t i = 0; i < robots_.size(); ++i) {
      f << (i ? ", " : "") << "\"" << robots_[i] << "\"";
    }
    f << "],\n  \"tasks\": [";
    for (std::size_t i = 0; i < tasks_.size(); ++i) {
      f << (i ? ", " : "") << "\"" << tasks_[i] << "\"";
    }
    f << "],\n  \"precedences\": [";
    bool first = true;
    for (const auto & p : art_["precedences"]) {
      f << (first ? "" : ", ") << "[\"" << p[0].as<std::string>() << "\", \""
        << p[1].as<std::string>() << "\"]";
      first = false;
    }
    f << "],\n";
    // Passed through verbatim, and only when the trajectory artifact has it: this
    // stage attaches no meaning to the modes, and an artifact from before they
    // existed must produce the same seam it always did.
    if (art_["precedence_modes"]) {
      f << "  \"precedence_modes\": [";
      first = true;
      for (const auto & m : art_["precedence_modes"]) {
        f << (first ? "" : ", ") << "\"" << m.as<std::string>() << "\"";
        first = false;
      }
      f << "],\n";
    }
    // Interchangeable slots, passed through verbatim and only when the trajectory
    // artifact has them. Four id-to-id relations, no geometry: which candidates
    // compete for one place in the plan, which physical object each consumes, and
    // how the slots are ordered. `slot_of`'s mere PRESENCE is what tells the
    // solver this is a slot problem, so an ordinary scene must not carry it.
    for (const char * key : {"slot_of", "object_of"}) {
      if (!art_[key]) {continue;}
      f << "  \"" << key << "\": {";
      first = true;
      for (const auto & kv : art_[key]) {
        f << (first ? "" : ", ") << "\"" << kv.first.as<std::string>() << "\": \""
          << kv.second.as<std::string>() << "\"";
        first = false;
      }
      f << "},\n";
    }
    if (art_["slot_precedences"]) {
      f << "  \"slot_precedences\": [";
      first = true;
      for (const auto & p : art_["slot_precedences"]) {
        f << (first ? "" : ", ") << "[\"" << p[0].as<std::string>() << "\", \""
          << p[1].as<std::string>() << "\"]";
        first = false;
      }
      f << "],\n";
    }
    if (art_["slot_precedence_modes"]) {
      f << "  \"slot_precedence_modes\": [";
      first = true;
      for (const auto & m : art_["slot_precedence_modes"]) {
        f << (first ? "" : ", ") << "\"" << m.as<std::string>() << "\"";
        first = false;
      }
      f << "],\n";
    }
    f << "  \"durations\": {\n";
    first = true;
    for (const auto & [key, tr] : trajs_) {
      f << (first ? "" : ",\n") << "    \"" << key.first << "|" << key.second << "\": "
        << tr.K();
      first = false;
    }
    f << "\n  },\n  \"pick_offsets\": {\n";
    first = true;
    for (const auto & [key, tr] : trajs_) {
      f << (first ? "" : ",\n") << "    \"" << key.first << "|" << key.second << "\": "
        << pickOffset(key, tr);
      first = false;
    }
    f << "\n  },\n  \"place_offsets\": {\n";
    first = true;
    for (const auto & [key, tr] : trajs_) {
      f << (first ? "" : ",\n") << "    \"" << key.first << "|" << key.second << "\": "
        << placeOffset(key, tr, pickOffset(key, tr));
      first = false;
    }
    f << "\n  },\n  \"transit_distances\": {\n";
    first = true;
    for (const auto & [key, tr] : trajs_) {
      const int pick = pickOffset(key, tr);
      const int place = placeOffset(key, tr, pick);
      f << (first ? "" : ",\n") << "    \"" << key.first << "|" << key.second << "\": "
        << transitDistance(key, tr, pick, place);
      first = false;
    }
    f << "\n  },\n  \"forbidden_offsets\": {\n";
    first = true;
    for (const auto & [key, offs] : forbidden) {
      f << (first ? "" : ",\n") << "    \"" << key << "\": [";
      for (std::size_t i = 0; i < offs.size(); ++i) {f << (i ? ", " : "") << offs[i];}
      f << "]";
      first = false;
    }
    f << "\n  }\n}\n";
  }

  /// The ACQUIRE milestone m0: index of the first sample whose phase is GripClose
  /// (a pick) or ProcessOn (a weld strikes its arc), as a 0-based offset into the
  /// task's own resampled sequence (same convention as `durations`).
  ///
  /// The JSON key stays `pick_offsets` so every archived seam keeps loading; for a
  /// pick-and-place the value is unchanged, because a pick-and-place trajectory has
  /// no process phase. MIRRORED in collision_generator_vamp.py `pick_offset` -- the
  /// two must stay identical. A trajectory with neither phase is malformed: fail loud.
  static int pickOffset(
    const std::pair<std::string, std::string> & key, const Traj & tr)
  {
    constexpr int grip_close = static_cast<int>(multi_robot_cell_tamp::Phase::GripClose);
    constexpr int process_on = static_cast<int>(multi_robot_cell_tamp::Phase::ProcessOn);
    for (std::size_t k = 0; k < tr.phase.size(); ++k) {
      if (tr.phase[k] == grip_close || tr.phase[k] == process_on) {return static_cast<int>(k);}
    }
    throw std::runtime_error(
      "trajectory " + key.first + "|" + key.second +
      " has no GripClose or ProcessOn sample -- no acquire milestone (m0)");
  }

  /// The RELEASE milestone m1: first sample at or after m0 whose phase is GripOpen
  /// (a place) or ProcessOff (the arc goes out). Key `place_offsets`, same reasons.
  static int placeOffset(
    const std::pair<std::string, std::string> & key, const Traj & tr, int pick)
  {
    constexpr int grip_open = static_cast<int>(multi_robot_cell_tamp::Phase::GripOpen);
    constexpr int process_off = static_cast<int>(multi_robot_cell_tamp::Phase::ProcessOff);
    for (std::size_t k = static_cast<std::size_t>(pick); k < tr.phase.size(); ++k) {
      if (tr.phase[k] == grip_open || tr.phase[k] == process_off) {return static_cast<int>(k);}
    }
    throw std::runtime_error(
      "trajectory " + key.first + "|" + key.second +
      " has no GripOpen or ProcessOff sample at or after its m0 -- no release milestone (m1)");
  }

  /// `SchedulingProblem.transit_distances`'s value for one (robot, task): the seam's
  /// own transit cost, NOT APEX-MR's formula (see model.py's `transit_distances`
  /// docstring for the full rationale -- summarised: two legs anchored at home->pick
  /// and pick->place, matching the topology of the home->pick->place->home
  /// trajectory this work actually plans, rather than APEX-MR's home->pick +
  /// home->place). L1 (sum of absolute per-DOF differences) over all 7 DOF (rail +
  /// 6 joints), scaled by TRANSIT_SCALE and rounded, same fixed-point convention
  /// `durations`/`delta_t` already use. MIRRORS collision_generator_vamp.py's
  /// `transit_distance` -- keep the two identical.
  long long transitDistance(
    const std::pair<std::string, std::string> & key, const Traj & tr, int pick, int place) const
  {
    const auto & home = homes_.at(key.first);
    const auto & pick_cfg = tr.q.at(static_cast<std::size_t>(pick));
    const auto & place_cfg = tr.q.at(static_cast<std::size_t>(place));
    double c = 0.0;
    for (std::size_t d = 0; d < home.size(); ++d) {c += std::abs(home[d] - pick_cfg[d]);}
    for (std::size_t d = 0; d < pick_cfg.size(); ++d) {c += std::abs(pick_cfg[d] - place_cfg[d]);}
    return std::llround(c * TRANSIT_SCALE);
  }

  /// Contiguous runs of equal value: [(value, begin, end), ...].
  static std::vector<std::tuple<int, std::size_t, std::size_t>> runs(const std::vector<int> & v)
  {
    std::vector<std::tuple<int, std::size_t, std::size_t>> out;
    std::size_t start = 0;
    for (std::size_t k = 1; k <= v.size(); ++k) {
      if (k == v.size() || v[k] != v[start]) {
        out.emplace_back(v[start], start, k);
        start = k;
      }
    }
    return out;
  }

  void loadObjectGeometry(const std::string & path)
  {
    YAML::Node root = YAML::LoadFile(path);
    for (const auto & n : root["objects"]) {
      ObjectGeom g;
      g.size = {n["size"][0].as<double>(), n["size"][1].as<double>(), n["size"][2].as<double>()};
      g.radius = 0.5 * std::sqrt(
        g.size[0] * g.size[0] + g.size[1] * g.size[1] + g.size[2] * g.size[2]);
      g.ee_T_obj = isoFromYaml(n["grasp"]).inverse();
      objects_[n["id"].as<std::string>()] = g;
    }
    for (const auto & kv : root["robots"]) {
      const std::string name = kv.first.as<std::string>();
      attach_link_[name] = kv.second["attach_link"].as<std::string>();
      groups_[name] = kv.second["planning_group"].as<std::string>();
      for (const auto & l : kv.second["touch_links"]) {
        touch_links_[name].push_back(l.as<std::string>());
      }
      // Link-name prefixes for this robot; default "<name>_" matches every
      // existing UR scene. See trajectory_generator.cpp's RobotCfg for the same
      // field -- the two generators must agree on which links belong to whom.
      if (kv.second["link_prefixes"]) {
        for (const auto & lp : kv.second["link_prefixes"]) {
          link_prefixes_[name].push_back(lp.as<std::string>());
        }
      } else {
        link_prefixes_[name].push_back(name + "_");
      }
      // The carried object's pose comes from the object's `grasp` and the robot's
      // `ee_link` (see ObjectGeom). The old `attach_offset` key is gone: fail loud
      // rather than silently ignore a scene that still sets it.
      if (kv.second["attach_offset"]) {
        throw std::runtime_error(
          "robot '" + name + "': `attach_offset` is no longer read -- the carried "
          "object's pose is grasp^-1 in `ee_link` (ADR-0005 addendum 2026-09-21)");
      }
      ee_link_[name] = kv.second["ee_link"].as<std::string>();
      approach_axis_[name] = Eigen::Vector3d::UnitZ();
      if (kv.second["tool_approach_axis"]) {
        const auto a = kv.second["tool_approach_axis"].as<std::string>();
        approach_axis_[name] = a == "x" ? Eigen::Vector3d::UnitX() :
          a == "y" ? Eigen::Vector3d::UnitY() : Eigen::Vector3d::UnitZ();
      }
    }
    checkGraspSymmetry();
  }

  /// The trajectory generator's IK accepts the grasp OR the grasp turned by pi about
  /// the tool approach axis (`ikTo`'s flip), whichever lands nearer the seed, and the
  /// artifact does not record which. mu models `grasp^-1` for both, which is exact
  /// only when the flip leaves the object's volume where it was: the object centre
  /// lies ON the approach axis and the box is symmetric under the half-turn (the axis
  /// is parallel to one of the box's own axes). Every current scene grasps a box from
  /// above along its z axis, which satisfies both; a scene that does not must fail here.
  void checkGraspSymmetry() const
  {
    for (const auto & [robot, axis] : approach_axis_) {
      for (const auto & [id, g] : objects_) {
        const Eigen::Vector3d t = g.ee_T_obj.translation();
        const double off_axis = (t - t.dot(axis) * axis).norm();
        const Eigen::Vector3d a_obj = g.ee_T_obj.linear().transpose() * axis;
        const double align = a_obj.cwiseAbs().maxCoeff();
        if (off_axis > 1e-6 || align < 1.0 - 1e-6) {
          throw std::runtime_error(
            "object '" + id + "' for robot '" + robot + "': its grasp is not symmetric "
            "under the IK's half-turn flip about the tool approach axis, so the carried "
            "pose is ambiguous -- mu cannot model it from `grasp` alone");
        }
      }
    }
  }

  /// `ee_link` in `attach_link`'s frame, per robot: the fixed transform that carries
  /// `ee_T_obj` into the frame the object is attached to. Identity on the UR cell
  /// (robotiq_85_base_link sits exactly on tool0). Checked to be configuration-
  /// independent: a movable joint between the two would make it meaningless.
  void computeAttachFrames()
  {
    moveit::core::RobotState a(model_), b(model_);
    a.setToDefaultValues();
    b.setToRandomPositions();
    a.update();
    b.update();
    for (const auto & r : robots_) {
      const auto rel = [&](const moveit::core::RobotState & s) {
          return Eigen::Isometry3d(
            s.getGlobalLinkTransform(attach_link_.at(r)).inverse() *
            s.getGlobalLinkTransform(ee_link_.at(r)));
        };
      const Eigen::Isometry3d ta = rel(a), tb = rel(b);
      if ((ta.matrix() - tb.matrix()).cwiseAbs().maxCoeff() > 1e-9) {
        throw std::runtime_error(
          "robot '" + r + "': ee_link -> attach_link is not a fixed transform");
      }
      attach_T_ee_[r] = ta;
      for (const auto & [id, g] : objects_) {
        const Eigen::Vector3d c = (ta * g.ee_T_obj).translation();
        RCLCPP_INFO(
          log_, "%s: carried '%s' centred at (%.4f, %.4f, %.4f) m in %s", r.c_str(), id.c_str(),
          c.x(), c.y(), c.z(), attach_link_.at(r).c_str());
      }
    }
  }

  /// The links belonging to each robot that actually have collision geometry.
  static bool matchesAnyPrefix(
    const std::string & link_name, const std::vector<std::string> & prefixes)
  {
    for (const auto & p : prefixes) {
      if (link_name.rfind(p, 0) == 0) {return true;}
    }
    return false;
  }

  void buildLinkSets()
  {
    for (const auto & r : robots_) {
      const auto & prefixes = link_prefixes_.at(r);
      for (const auto * lm : model_->getLinkModels()) {
        if (!lm->getShapes().empty() && matchesAnyPrefix(lm->getName(), prefixes)) {
          links_[r].push_back(lm);
        }
      }
      RCLCPP_INFO(log_, "%s: %zu collision links", r.c_str(), links_[r].size());
      if (links_[r].empty()) {
        throw std::runtime_error(
          "robot '" + r + "' matched zero collision links under its link_prefixes "
          "-- check the scene YAML's link_prefixes against the URDF link names");
      }
    }
  }

  /// Mask the ACM so ONLY robot-vs-robot pairs are checked.
  ///
  /// Everything is allowed by default; then exactly the cross-robot link pairs (and
  /// the two carried objects, which are attached bodies and therefore appear in the
  /// ACM under their object ids) are DISallowed, i.e. checked. Self-collision and
  /// robot-vs-world are deliberately allowed here: the trajectory stage already
  /// settled them, and a hit for one of those reasons would corrupt mu with a
  /// collision that has nothing to do with the robots' relative timing.
  void buildAcm()
  {
    acm_ = std::make_shared<collision_detection::AllowedCollisionMatrix>(
      scene_->getAllowedCollisionMatrix());

    std::vector<std::string> all;
    for (const auto * lm : model_->getLinkModels()) {all.push_back(lm->getName());}
    for (const auto & [id, g] : objects_) {(void)g; all.push_back(id);}
    for (const auto & a : all) {
      for (const auto & b : all) {acm_->setEntry(a, b, true);}   // allow (= do not check)
    }

    auto names = [&](const std::string & robot) {
        std::vector<std::string> v;
        for (const auto * lm : links_[robot]) {v.push_back(lm->getName());}
        for (const auto & [id, g] : objects_) {(void)g; v.push_back(id);}
        return v;
      };
    // Object ids are shared across robots (only one robot holds a given object at a
    // time), so adding every object id to both sides is safe: the pair (obj, obj) is
    // never both-attached, and an object attached to r is checked against s's links.
    std::size_t checked = 0;
    for (const auto & a : names(robots_[0])) {
      for (const auto & b : names(robots_[1])) {
        if (a == b) {continue;}
        acm_->setEntry(a, b, false);                              // DISallow (= check)
        ++checked;
      }
    }
    RCLCPP_INFO(log_, "ACM masked: %zu cross-robot pairs are checked, all else allowed", checked);
  }

  void configureAttachments(
    const std::string & r, const Traj & ti, int state_i,
    const std::string & s, const Traj & tj, int state_j)
  {
    setAttached(r, ti.object, state_i == ATTACHED);
    setAttached(s, tj.object, state_j == ATTACHED);
  }

  void setAttached(const std::string & robot, const std::string & object, bool attach)
  {
    // A process task carries nothing ("object": ""). Returning here is not only
    // tidier: a REMOVE message with an EMPTY id means "remove everything" to MoveIt,
    // attached bodies on the link and world objects alike.
    if (object.empty()) {
      if (attach) {
        throw std::runtime_error("an objectless trajectory has an ATTACHED sample");
      }
      return;
    }
    moveit_msgs::msg::AttachedCollisionObject aco;
    aco.link_name = attach_link_.at(robot);
    aco.touch_links = touch_links_.at(robot);
    aco.object.id = object;
    aco.object.header.frame_id = attach_link_.at(robot);

    if (!attach) {
      aco.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      scene_->processAttachedCollisionObjectMsg(aco);
      moveit_msgs::msg::CollisionObject rm;
      rm.id = object;
      rm.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      scene_->processCollisionObjectMsg(rm);
      return;
    }

    const auto & g = objects_.at(object);
    shape_msgs::msg::SolidPrimitive prim;
    prim.type = prim.BOX;
    prim.dimensions = {g.size[0], g.size[1], g.size[2]};
    aco.object.primitives.push_back(prim);
    // The TRUE carried pose, translation and rotation: grasp^-1 in ee_link, carried
    // into attach_link (see ObjectGeom). Not second-order: the fixed 0.10 m offset
    // this replaced put the modelled box 1-3 cm clear of the real one.
    const Eigen::Isometry3d t = attach_T_ee_.at(robot) * g.ee_T_obj;
    const Eigen::Quaterniond qr(t.linear());
    geometry_msgs::msg::Pose p;
    p.position.x = t.translation().x();
    p.position.y = t.translation().y();
    p.position.z = t.translation().z();
    p.orientation.x = qr.x();
    p.orientation.y = qr.y();
    p.orientation.z = qr.z();
    p.orientation.w = qr.w();
    aco.object.primitive_poses.push_back(p);
    aco.object.operation = moveit_msgs::msg::CollisionObject::ADD;
    scene_->processAttachedCollisionObjectMsg(aco);
  }

  void setRobot(moveit::core::RobotState & state, const std::string & robot, const Traj & t,
    std::size_t k)
  {
    const auto * jmg = model_->getJointModelGroup(groups_.at(robot));
    state.setJointGroupPositions(jmg, t.q[k]);
  }

  bool inCollision(
    const std::string & r, const Traj & ti, std::size_t k,
    const std::string & s, const Traj & tj, std::size_t l)
  {
    moveit::core::RobotState & state = scene_->getCurrentStateNonConst();
    setRobot(state, r, ti, k);
    setRobot(state, s, tj, l);
    state.update();

    collision_detection::CollisionRequest req;
    req.contacts = false;
    collision_detection::CollisionResult res;
    scene_->checkCollision(req, res, state, *acm_);
    return res.collision;
  }

  /// Bounding sphere over all of a robot's links (plus its carried object, when it
  /// has one) at each sample. Built from the FK we have to do anyway.
  std::vector<Bound> computeBounds(const Traj & t)
  {
    moveit::core::RobotState state(model_);
    state.setToDefaultValues();

    std::vector<Bound> out(t.K());
    // Objectless (weld) trajectories have no object slot: never attached.
    const ObjectGeom * obj = t.object.empty() ? nullptr : &objects_.at(t.object);
    // 1 mm of slack on every bounding sphere: the broad phase may only ever let MORE
    // pairs through to FCL, never fewer.
    constexpr double kSlack = 1e-3;

    for (std::size_t k = 0; k < t.K(); ++k) {
      setRobot(state, t.robot, t, k);
      state.update();

      std::vector<std::pair<Eigen::Vector3d, double>> spheres;
      for (const auto * lm : links_.at(t.robot)) {
        // The link's collision AABB is `extents` wide and centred at
        // `centered_bounding_box_offset` in the LINK frame -- not at the link origin
        // (a UR upper-arm mesh starts at the shoulder and runs 0.6 m out). Centring
        // the sphere at the origin, as this did before 2026-09-21, could reject a
        // pair whose far end actually touches.
        const Eigen::Vector3d c =
          state.getGlobalLinkTransform(lm) * lm->getCenteredBoundingBoxOffset();
        const Eigen::Vector3d ext = lm->getShapeExtentsAtOrigin();
        spheres.emplace_back(c, 0.5 * ext.norm() + kSlack);
      }
      if (obj != nullptr && t.object_state[k] == ATTACHED) {
        // The carried box, bounded about its TRUE centre by its half-diagonal.
        const auto * al = model_->getLinkModel(attach_link_.at(t.robot));
        const Eigen::Vector3d c = state.getGlobalLinkTransform(al) *
          (attach_T_ee_.at(t.robot) * obj->ee_T_obj).translation();
        spheres.emplace_back(c, obj->radius + kSlack);
      }

      Eigen::Vector3d centre = Eigen::Vector3d::Zero();
      for (const auto & [c, rad] : spheres) {centre += c;}
      centre /= static_cast<double>(spheres.size());

      double radius = 0.0;
      for (const auto & [c, rad] : spheres) {
        radius = std::max(radius, (c - centre).norm() + rad);
      }
      out[k] = Bound{centre, radius};
    }
    return out;
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger log_;
  YAML::Node art_;
  moveit::core::RobotModelPtr model_;
  planning_scene::PlanningScenePtr scene_;
  collision_detection::AllowedCollisionMatrixPtr acm_;

  double delta_t_{0.0};
  std::vector<std::string> robots_, tasks_;
  std::map<std::pair<std::string, std::string>, Traj> trajs_;
  std::map<std::string, ObjectGeom> objects_;
  std::map<std::string, std::string> attach_link_, groups_;
  std::map<std::string, std::vector<std::string>> touch_links_;
  std::map<std::string, std::vector<const moveit::core::LinkModel *>> links_;
  std::map<std::string, std::vector<std::string>> link_prefixes_;
  std::map<std::string, std::string> ee_link_;
  std::map<std::string, Eigen::Vector3d> approach_axis_;
  std::map<std::string, Eigen::Isometry3d> attach_T_ee_;
  // Empty unless the scene declares interchangeable slots.
  std::map<std::string, std::string> slot_of_, object_of_;
  // Per-robot home configuration, for `transit_distances`.
  std::map<std::string, std::vector<double>> homes_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
    "collision_generator",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  const std::string traj_file = node->get_parameter("traj_file").as_string();
  const std::string task_file = node->get_parameter("task_file").as_string();
  const std::string out_file = node->get_parameter("out_file").as_string();

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::thread spinner([&exec]() {exec.spin();});

  int rc = 0;
  try {
    // JSON parses as YAML.
    const YAML::Node art = YAML::LoadFile(traj_file);
    CollisionGenerator gen(node, art, task_file);
    std::string dump;
    node->get_parameter_or("dump_object_centres", dump, std::string{});
    if (dump.empty()) {
      gen.run(out_file);
    } else {
      gen.dumpObjectCentres(dump);
    }
  } catch (const std::exception & e) {
    RCLCPP_FATAL(node->get_logger(), "%s", e.what());
    rc = 1;
  }

  exec.cancel();
  spinner.join();
  rclcpp::shutdown();
  return rc;
}
