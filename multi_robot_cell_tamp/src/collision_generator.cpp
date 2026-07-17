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

// JSON is a subset of YAML, and yaml-cpp is already a dependency -- so the
// trajectory artifact is parsed with it rather than dragging in nlohmann/json
// (which is not installed and would need root to add).
#include <yaml-cpp/yaml.h>

namespace
{

constexpr int AT_SPAWN = 0;
constexpr int ATTACHED = 1;
constexpr int AT_PLACE = 2;

struct Traj
{
  std::string robot;
  std::string task;
  std::string object;
  std::vector<std::string> joint_names;
  std::vector<std::vector<double>> q;      // K x 7
  std::vector<int> object_state;           // K
  std::size_t K() const {return q.size();}
};

/// One robot's whole geometry at one sample, reduced to a bounding sphere.
/// Cheap to build (FK we need anyway) and it prunes ~all of the 9.2M pairs.
struct Bound
{
  Eigen::Vector3d centre;
  double radius;
};

struct ObjectGeom
{
  std::array<double, 3> size;
  double radius;                            // bounding sphere of the box
};

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
      trajs_[{tr.robot, tr.task}] = std::move(tr);
    }

    loadObjectGeometry(task_yaml);
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

private:
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
    f << "],\n  \"durations\": {\n";
    first = true;
    for (const auto & [key, tr] : trajs_) {
      f << (first ? "" : ",\n") << "    \"" << key.first << "|" << key.second << "\": "
        << tr.K();
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
      objects_[n["id"].as<std::string>()] = g;
    }
    for (const auto & kv : root["robots"]) {
      const std::string name = kv.first.as<std::string>();
      attach_link_[name] = kv.second["attach_link"].as<std::string>();
      groups_[name] = kv.second["planning_group"].as<std::string>();
      for (const auto & l : kv.second["touch_links"]) {
        touch_links_[name].push_back(l.as<std::string>());
      }
    }
  }

  /// The links belonging to each robot that actually have collision geometry.
  void buildLinkSets()
  {
    for (const auto & r : robots_) {
      for (const auto * lm : model_->getLinkModels()) {
        if (lm->getName().rfind(r + "_", 0) == 0 && !lm->getShapes().empty()) {
          links_[r].push_back(lm);
        }
      }
      RCLCPP_INFO(log_, "%s: %zu collision links", r.c_str(), links_[r].size());
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
    geometry_msgs::msg::Pose p;
    p.orientation.w = 1.0;
    // Attached at the gripper's grasp offset. The exact offset within the gripper is
    // second-order for an inter-robot check -- what matters is that the object's
    // VOLUME travels with the arm, which a naive "robot links only" mu would miss.
    p.position.z = 0.10;
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
    const double obj_r = objects_.at(t.object).radius;

    for (std::size_t k = 0; k < t.K(); ++k) {
      setRobot(state, t.robot, t, k);
      state.update();

      std::vector<std::pair<Eigen::Vector3d, double>> spheres;
      for (const auto * lm : links_.at(t.robot)) {
        const Eigen::Vector3d c = state.getGlobalLinkTransform(lm).translation();
        const Eigen::Vector3d ext = lm->getShapeExtentsAtOrigin();
        spheres.emplace_back(c, 0.5 * ext.norm());
      }
      if (t.object_state[k] == ATTACHED) {
        // The carried object rides the gripper; bound it at the attach link.
        const auto * al = model_->getLinkModel(attach_link_.at(t.robot));
        spheres.emplace_back(
          state.getGlobalLinkTransform(al).translation(), obj_r + 0.15);
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
    gen.run(out_file);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(node->get_logger(), "%s", e.what());
    rc = 1;
  }

  exec.cancel();
  spinner.join();
  rclcpp::shutdown();
  return rc;
}
