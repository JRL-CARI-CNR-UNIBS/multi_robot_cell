// plan_oracle_check -- the geometric back end of the independent plan oracle.
//
//     plan_oracle_check <replay.txt> <contacts.txt>
//
// It is driven by `scripts/plan_oracle.py`, which turns a FINAL plan (trajectories +
// schedule + plan graph + scene YAML) into a tick-by-tick REPLAY: both robots' joint
// values and every object's realised state (at spawn, riding a gripper, at its place
// pose). This program replays that on the REAL geometry -- the cell's URDF collision
// meshes, the SRDF's allowed-collision matrix, the scene's fixtures and the objects' own
// boxes or meshes -- with MoveIt's FCL checker, and writes every contact it finds.
//
// Why it exists (ADR-0007, addendum 2026-09-21): `simulate_tpg.py`'s oracle reads the
// same sphere model and the same mu that produced the plan, so a modelling error shared by
// the planner and the checker (e.g. the carried object at the wrong pose, which both mu
// engines had until 2026-09-21) is invisible to it. Nothing here reads mu, the spheres or
// the seam's forbidden offsets. It shares only the robot model with the planner.
//
// What is NOT reported, and why (and nothing else is excluded):
//   * pairs the SRDF disables (a robot's adjacent / never-colliding links): MoveIt's own
//     notion of a legitimate self-contact;
//   * an attached object against the carrying robot's `touch_links` (the gripper fingers),
//     and only while it is attached -- MoveIt's attach semantics;
//   * an object against the support surface (`SUPPORT`, the table top) while it stands
//     still at a spawn or place pose; while it is carried only a touching contact
//     (depth <= SUPPORT_TOL) is excused -- the lift-off and set-down;
//   * a carried object against another object or a fixture with depth <= FITUP_GAP, only
//     when the scene declares a fit-up gap (FITUP_GAP > 0). With FITUP_GAP = 0 (default)
//     every such contact is reported.
// World-world pairs (two static objects) are not checked by MoveIt; the set-down contact
// is caught while the object is still attached, at its release dwell.
//
// Replay format: one directive per line, whitespace-separated. See plan_oracle.py.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shapes.h>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>

namespace
{

std::string slurp(const std::string & path)
{
  std::ifstream f(path);
  if (!f) {throw std::runtime_error("cannot read " + path);}
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

Eigen::Isometry3d readPose(std::istream & in)
{
  double x, y, z, qx, qy, qz, qw;
  in >> x >> y >> z >> qx >> qy >> qz >> qw;
  if (!in) {throw std::runtime_error("malformed pose");}
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.translation() = Eigen::Vector3d(x, y, z);
  t.linear() = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
  return t;
}

shapes::ShapeConstPtr readShape(std::istream & in)
{
  std::string kind;
  in >> kind;
  if (kind == "BOX") {
    double sx, sy, sz;
    in >> sx >> sy >> sz;
    return std::make_shared<shapes::Box>(sx, sy, sz);
  }
  if (kind == "MESH") {
    std::string path;
    double scale;
    in >> path >> scale;
    shapes::ShapeConstPtr m(shapes::createMeshFromResource(
        "file://" + path, Eigen::Vector3d(scale, scale, scale)));
    if (!m) {throw std::runtime_error("cannot load mesh " + path);}
    return m;
  }
  throw std::runtime_error("unknown shape kind '" + kind + "'");
}

const char * typeName(collision_detection::BodyType t)
{
  switch (t) {
    case collision_detection::BodyTypes::ROBOT_LINK: return "robot_link";
    case collision_detection::BodyTypes::ROBOT_ATTACHED: return "attached_object";
    case collision_detection::BodyTypes::WORLD_OBJECT: return "world_object";
  }
  return "unknown";
}

struct RobotInfo
{
  std::string attach_link;
  std::set<std::string> touch_links;
  std::vector<std::string> joints;
};

enum class ObjState { None, Static, Attached };

struct ObjInfo
{
  shapes::ShapeConstPtr shape;
  ObjState state{ObjState::None};
  std::string kind;              // "spawn" / "place" while static
  std::string robot;             // carrier while attached
  Eigen::Isometry3d world{Eigen::Isometry3d::Identity()};
};

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::cerr << "usage: plan_oracle_check <replay.txt> <contacts.txt>\n";
    return 2;
  }
  rclcpp::init(1, argv);
  const auto t0 = std::chrono::steady_clock::now();
  std::ifstream in(argv[1]);
  std::ofstream out(argv[2]);
  if (!in || !out) {std::cerr << "cannot open replay or output file\n"; return 2;}

  std::string urdf_path, srdf_path, support = "table_top";
  double support_tol = 2e-3, fitup_gap = 0.0;
  std::map<std::string, RobotInfo> robots;
  std::map<std::string, ObjInfo> objects;
  std::vector<std::pair<std::string, std::pair<shapes::ShapeConstPtr, Eigen::Isometry3d>>> fixtures;

  moveit::core::RobotModelPtr model;
  std::shared_ptr<planning_scene::PlanningScene> scene;
  std::unique_ptr<moveit::core::RobotState> state;
  collision_detection::AllowedCollisionMatrix acm;

  long n_checks = 0, n_ticks = 0, n_contact_ticks = 0;
  long excused_support = 0, excused_fitup = 0;
  std::string line;
  int lineno = 0;
  try {
    while (std::getline(in, line)) {
      ++lineno;
      std::istringstream ls(line);
      std::string cmd;
      if (!(ls >> cmd) || cmd[0] == '#') {continue;}
      if (cmd == "URDF") {ls >> urdf_path;}
      else if (cmd == "SRDF") {ls >> srdf_path;}
      else if (cmd == "SUPPORT") {ls >> support;}
      else if (cmd == "SUPPORT_TOL") {ls >> support_tol;}
      else if (cmd == "FITUP_GAP") {ls >> fitup_gap;}
      else if (cmd == "ROBOT") {
        std::string name; int n;
        ls >> name;
        RobotInfo & r = robots[name];
        ls >> r.attach_link >> n;
        for (int i = 0; i < n; ++i) {std::string l; ls >> l; r.touch_links.insert(l);}
      } else if (cmd == "JOINTS") {
        std::string name; int n;
        ls >> name >> n;
        for (int i = 0; i < n; ++i) {std::string j; ls >> j; robots[name].joints.push_back(j);}
      } else if (cmd == "SHAPE") {
        std::string id;
        ls >> id;
        objects[id].shape = readShape(ls);
      } else if (cmd == "FIXTURE") {
        std::string id;
        ls >> id;
        auto shape = readShape(ls);
        fixtures.push_back({id, {shape, readPose(ls)}});
      } else if (cmd == "BEGIN") {
        auto urdf = urdf::parseURDF(slurp(urdf_path));
        if (!urdf) {throw std::runtime_error("cannot parse URDF " + urdf_path);}
        auto srdf = std::make_shared<srdf::Model>();
        if (!srdf->initString(*urdf, slurp(srdf_path))) {
          throw std::runtime_error("cannot parse SRDF " + srdf_path);
        }
        model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
        if (!model->hasLinkModel(support)) {
          throw std::runtime_error("support surface '" + support + "' is not a link of the model");
        }
        scene = std::make_shared<planning_scene::PlanningScene>(model);
        state = std::make_unique<moveit::core::RobotState>(model);
        state->setToDefaultValues();   // gripper joints at 0 = open, as MoveIt plans them
        state->update();
        acm = scene->getAllowedCollisionMatrix();   // the SRDF's, and only that
        for (const auto & [id, sp] : fixtures) {
          scene->getWorldNonConst()->addToObject(id, sp.second, sp.first, Eigen::Isometry3d::Identity());
        }
        out << "FIXTURES " << fixtures.size() << "\n";
      } else if (cmd == "Q") {
        std::string name;
        ls >> name;
        const auto & r = robots.at(name);
        for (const auto & j : r.joints) {
          double v;
          ls >> v;
          state->setJointPositions(j, &v);
        }
        if (!ls) {throw std::runtime_error("short Q line");}
        state->update();
      } else if (cmd == "SPAWN" || cmd == "PLACE") {
        std::string id;
        ls >> id;
        const Eigen::Isometry3d pose = readPose(ls);
        ObjInfo & o = objects.at(id);
        if (o.state == ObjState::Attached) {
          // Release: how far the carried object is from where the plan says it lands --
          // a direct check of the carried-pose model (grasp^-1 from the realised grasp).
          const auto * ab = state->getAttachedBody(id);
          const Eigen::Isometry3d carried = ab->getGlobalPose();
          const double d = (carried.translation() - pose.translation()).norm();
          const double a = Eigen::AngleAxisd(carried.linear().transpose() * pose.linear()).angle();
          out << "RELEASE " << id << " " << o.robot << " " << d << " " << a << "\n";
          state->clearAttachedBody(id);
        } else if (o.state == ObjState::Static) {
          scene->getWorldNonConst()->removeObject(id);
        }
        scene->getWorldNonConst()->addToObject(id, pose, o.shape, Eigen::Isometry3d::Identity());
        acm.setEntry(id, support, true);     // a standing object rests on the table
        o.state = ObjState::Static;
        o.kind = cmd == "SPAWN" ? "spawn" : "place";
        o.world = pose;
      } else if (cmd == "ATTACH") {
        std::string id, robot;
        ls >> id >> robot;
        ObjInfo & o = objects.at(id);
        if (o.state != ObjState::Static) {
          throw std::runtime_error("ATTACH of '" + id + "', which is not standing anywhere");
        }
        const auto & r = robots.at(robot);
        // The object is fixed to the link WHERE IT STANDS at the grasp configuration --
        // the realised grasp, exactly what the trajectory generator's attach does. It does
        // not assume grasp^-1 (nor the IK's possible half-turn flip); RELEASE then reports
        // how far that carried pose lands from the planned place pose.
        const Eigen::Isometry3d link_T_obj =
          state->getGlobalLinkTransform(r.attach_link).inverse() * o.world;
        scene->getWorldNonConst()->removeObject(id);
        acm.setEntry(id, support, false);    // a carried object may only TOUCH the table
        state->attachBody(id, link_T_obj, {o.shape}, {Eigen::Isometry3d::Identity()},
                          r.touch_links, r.attach_link);
        state->update();
        const Eigen::Vector3d c = link_T_obj.translation();
        out << "ATTACH " << id << " " << robot << " " << c.x() << " " << c.y() << " " << c.z()
            << "\n";
        o.state = ObjState::Attached;
        o.robot = robot;
      } else if (cmd == "CHECK") {
        long first, last;
        ls >> first >> last;
        collision_detection::CollisionRequest req;
        req.contacts = true;
        req.max_contacts = 100000;
        req.max_contacts_per_pair = 4;
        collision_detection::CollisionResult res;
        state->updateCollisionBodyTransforms();
        scene->checkCollision(req, res, *state, acm);
        ++n_checks;
        n_ticks += last - first + 1;
        bool any = false;
        for (const auto & [pair, cs] : res.contacts) {
          double depth = 0.0;
          for (const auto & c : cs) {depth = std::max(depth, c.depth);}
          const auto & c = cs.front();
          const bool a1 = c.body_type_1 == collision_detection::BodyTypes::ROBOT_ATTACHED;
          const bool a2 = c.body_type_2 == collision_detection::BodyTypes::ROBOT_ATTACHED;
          const bool w1 = c.body_type_1 == collision_detection::BodyTypes::WORLD_OBJECT;
          const bool w2 = c.body_type_2 == collision_detection::BodyTypes::WORLD_OBJECT;
          // Carried object touching the table (lift-off / set-down).
          if ((a1 && c.body_name_2 == support) || (a2 && c.body_name_1 == support)) {
            if (depth <= support_tol) {++excused_support; continue;}
          }
          // Carried object seated on a declared fit-up gap.
          if (fitup_gap > 0.0 && ((a1 && w2) || (a2 && w1)) && depth <= fitup_gap) {
            ++excused_fitup;
            continue;
          }
          any = true;
          out << "CONTACT " << first << " " << last << " " << c.body_name_1 << " "
              << typeName(c.body_type_1) << " " << c.body_name_2 << " "
              << typeName(c.body_type_2) << " " << depth << "\n";
        }
        if (any) {n_contact_ticks += last - first + 1;}
      } else if (cmd == "END") {
        break;
      } else {
        throw std::runtime_error("unknown directive '" + cmd + "'");
      }
    }
  } catch (const std::exception & e) {
    std::cerr << "plan_oracle_check: line " << lineno << ": " << e.what() << "\n";
    out << "ERROR " << lineno << " " << e.what() << "\n";
    rclcpp::shutdown();
    return 3;
  }
  const double secs = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - t0).count();
  out << "STATS " << n_checks << " " << n_ticks << " " << n_contact_ticks << " " << secs << " "
      << excused_support << " " << excused_fitup << "\n";
  rclcpp::shutdown();
  return 0;
}
