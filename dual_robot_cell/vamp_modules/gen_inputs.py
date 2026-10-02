#!/usr/bin/env python3
"""Generate cricket codegen inputs for vamp.ur10e_rail:

  - ur10e_rail_spherized.urdf : shipped CoMMALab spherized UR10e+2F85 gripper
                                (already sphere-decomposed by foam upstream) with the
                                cell's prismatic linear-guide (rail) inserted at the base,
                                so the movable-joint order pinocchio sees is
                                [linear_guide_joint(prismatic), shoulder_pan, shoulder_lift,
                                 elbow, wrist_1, wrist_2, wrist_3]  == the artifact order.
  - ur10e_rail.srdf           : robot1_* disable_collisions from the cell moveit SRDF,
                                de-prefixed to the single-robot link names (self-collision
                                info; does NOT affect fk()/mu, only lets cricket run fast).
  - ur10e_rail.json           : cricket config.

Rail geometry (scope locked with user: rail + arm + gripper) is added as conservative
sphere rows over the support/rail/carriage boxes from the cell macro. These structural
links are geometrically separated from the *other* robot, so they contribute nothing to
mu (only broad-phase cost) -- but they are included to honor the locked scope and to match
FCL's robot geometry.  Values come straight from ur10e_2f85_linear_robot_macro.xacro +
multi_robot_cell.urdf.xacro:
    guide_length = table_length+0.6 = 2.8   guide_travel = table_length = 2.2  -> rail +-1.1
    support_height = guide_origin_z = 0.70   rail_height = 0.05   carriage_height = 0.03
    support_width = 0.25   rail_width = 0.20   carriage_length = 0.35
    world->support origin z = guide_origin_z = 0.70 (robot placed at cell height, y=0, yaw=0
    -> robot-agnostic canonical base; per-robot y/yaw belong to the engine base_transform).
"""
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
SHIPPED = "/home/riccardo_dolci/ompl_icra_ws/cricket_src/resources/ur10e/ur_10e_spherized.urdf"
MOVEIT_SRDF = "/home/riccardo_dolci/ws_thesis/src/multi_robot_cell/dual_robot_cell/multi_robot_moveit_config/config/multi_robot_cell.srdf"
OUT = os.environ.get("VAMP_CODEGEN_INPUTS", HERE)
os.makedirs(OUT, exist_ok=True)

# `--tool torch` (fabricator4 welders): the Robotiq 2F-85 and its 58 spheres are replaced by the
# parametric MIG torch of thesis_material_tamp/tools/make_torch.py -- its sphere chain (tool0
# frame) on a `torch_link` fixed to tool0, and a `torch_tcp` frame (the wire tip) as the end
# effector. Module `ur10e_rail_torch`. The rail rows stay FIRST (the engine drops the first 18
# spheres as structural, CELL_N_STRUCTURAL), then the 21 arm spheres, then the torch.
ap = argparse.ArgumentParser()
ap.add_argument("--tool", choices=("gripper", "torch"), default="gripper")
ap.add_argument("--torch-spheres", default=os.path.join(
    HERE, "..", "multi_robot_cell_description", "meshes", "torch", "spheres.yaml"))
ARGS = ap.parse_args()
TORCH = ARGS.tool == "torch"
NAME = "ur10e_rail_torch" if TORCH else "ur10e_rail"


def load_torch(path):
    """(tcp_xyz, tcp_rpy, [(x, y, z, r), ...]) from make_torch.py's spheres.yaml (no yaml dep)."""
    tcp, spheres = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith("tcp:"):
            xyz = line.split("xyz: [")[1].split("]")[0]
            rpy = line.split("rpy: [")[1].split("]")[0]
            tcp = ([float(v) for v in xyz.split(",")], [float(v) for v in rpy.split(",")])
        elif line.startswith("- ["):
            spheres.append([float(v) for v in line[3:-1].split(",")])
    return tcp, spheres

# ---- cell rail dimensions (metres) --------------------------------------- #
GUIDE_LEN   = 2.8     # table_length + 0.6
GUIDE_TRAV  = 2.2     # table_length  -> prismatic limits +-1.1
SUP_H       = 0.70    # support_height = guide_origin_z
SUP_W       = 0.25
RAIL_H      = 0.05
RAIL_W      = 0.20
CARR_H      = 0.03
CARR_L      = 0.35
GUIDE_ORIGIN_Z = 0.70 # world -> support link frame

def sphere_collision(x, y, z, r):
    c = ET.Element("collision")
    g = ET.SubElement(c, "geometry")
    s = ET.SubElement(g, "sphere"); s.set("radius", f"{r:.6f}")
    o = ET.SubElement(c, "origin"); o.set("xyz", f"{x:.6f} {y:.6f} {z:.6f}"); o.set("rpy", "0 0 0")
    return c

def box_sphere_row(link, half_len_x, z_center, r, n):
    """Row of n spheres along local x at height z_center, radius r."""
    xs = np.linspace(-half_len_x, half_len_x, n)
    for x in xs:
        link.append(sphere_collision(float(x), 0.0, z_center, r))

def make_link(name):
    l = ET.Element("link"); l.set("name", name); return l

def make_joint(name, jtype, parent, child, xyz, rpy="0 0 0", axis=None, limit=None):
    j = ET.Element("joint"); j.set("name", name); j.set("type", jtype)
    p = ET.SubElement(j, "parent"); p.set("link", parent)
    c = ET.SubElement(j, "child"); c.set("link", child)
    o = ET.SubElement(j, "origin"); o.set("xyz", xyz); o.set("rpy", rpy)
    if axis is not None:
        a = ET.SubElement(j, "axis"); a.set("xyz", axis)
    if limit is not None:
        lo, hi = limit
        lm = ET.SubElement(j, "limit")
        lm.set("lower", f"{lo}"); lm.set("upper", f"{hi}")
        lm.set("effort", "5000"); lm.set("velocity", "1.0")
    return j

# ------------------------------------------------------------------- URDF -- #
tree = ET.parse(SHIPPED)
root = tree.getroot()
root.set("name", NAME)
if "path" in root.attrib:
    del root.attrib["path"]

# strip visuals (drop mesh dependency) and inertials (unneeded for fk/collision)
for link in root.findall("link"):
    for tag in ("visual", "inertial"):
        for e in link.findall(tag):
            link.remove(e)

if TORCH:   # drop the gripper: every robotiq_* link and every joint touching one
    for j in list(root.findall("joint")):
        if any(e.get("link", "").startswith("robotiq_") for e in (j.find("parent"), j.find("child"))):
            root.remove(j)
    for l in list(root.findall("link")):
        if l.get("name").startswith("robotiq_"):
            root.remove(l)
    (tcp_xyz, tcp_rpy), torch_spheres = load_torch(ARGS.torch_spheres)
    torch = make_link("torch_link")
    for x, y, z, r in torch_spheres:
        torch.append(sphere_collision(x, y, z, r))
    root.append(torch)
    root.append(make_link("torch_tcp"))
    root.append(make_joint("tool0-torch", "fixed", "tool0", "torch_link", xyz="0 0 0"))
    root.append(make_joint("torch-tcp", "fixed", "torch_link", "torch_tcp",
                           xyz=" ".join(f"{v:.6f}" for v in tcp_xyz),
                           rpy=" ".join(f"{v:.6f}" for v in tcp_rpy)))

# remove the old fixed world->base_link joint; we splice the rail chain in
for j in root.findall("joint"):
    if j.get("name") == "base_joint":
        root.remove(j)

# --- rail links with conservative sphere cover --------------------------- #
support = make_link("linear_guide_support_link")
# support box 2.8 x 0.25 x 0.70, box centre at z=-SUP_H/2 in support frame
box_sphere_row(support, GUIDE_LEN/2, -SUP_H/2, r=0.40, n=8)

rail = make_link("linear_guide_rail_link")
# rail box 2.8 x 0.20 x 0.05, box centre at z=+RAIL_H/2 in rail frame
box_sphere_row(rail, GUIDE_LEN/2, RAIL_H/2, r=0.12, n=8)

carriage = make_link("linear_guide_carriage_link")
# carriage box 0.35 x 0.20 x 0.03, box centre at z=+CARR_H/2 in carriage frame
carriage.append(sphere_collision(-CARR_L/3, 0.0, CARR_H/2, 0.12))
carriage.append(sphere_collision( CARR_L/3, 0.0, CARR_H/2, 0.12))

# insert the three links right after <link name="world">
world_idx = list(root).index(root.find("link"))  # first link == world
for off, l in enumerate((support, rail, carriage), start=1):
    root.insert(world_idx + off, l)

# --- rail chain joints (mirrors the cell macro, robot at y=0, yaw=0) ------ #
joints = [
    make_joint("world_to_support", "fixed", "world", "linear_guide_support_link",
               xyz=f"0 0 {GUIDE_ORIGIN_Z}"),
    make_joint("linear_guide_rail_joint", "fixed", "linear_guide_support_link",
               "linear_guide_rail_link", xyz="0 0 0"),
    make_joint("linear_guide_joint", "prismatic", "linear_guide_rail_link",
               "linear_guide_carriage_link", xyz=f"0 0 {RAIL_H}", axis="1 0 0",
               limit=(-GUIDE_TRAV/2, GUIDE_TRAV/2)),
    make_joint("carriage_to_base", "fixed", "linear_guide_carriage_link", "base_link",
               xyz=f"0 0 {CARR_H}", rpy="0 0 0"),
]
# append near the top (after links); order among joints does not matter to pinocchio
for j in reversed(joints):
    root.insert(world_idx + 4, j)

urdf_path = os.path.join(OUT, f"{NAME}_spherized.urdf")
ET.indent(tree, space="  ")
tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
print("wrote", urdf_path)

# ------------------------------------------------------------------- SRDF -- #
msrdf = ET.parse(MOVEIT_SRDF).getroot()
def deprefix(name):
    return name[len("robot1_"):] if name.startswith("robot1_") else name

pairs = []
for dc in msrdf.findall("disable_collisions"):
    l1, l2 = dc.get("link1"), dc.get("link2")
    if l1.startswith("robot1_") and l2.startswith("robot1_"):
        if TORCH and ("robotiq" in l1 or "robotiq" in l2):
            continue
        pairs.append((deprefix(l1), deprefix(l2), dc.get("reason", "Never")))
if TORCH:
    pairs += [("wrist_3_link", "torch_link", "Adjacent"), ("wrist_2_link", "torch_link", "Never")]

srobot = ET.Element("robot"); srobot.set("name", NAME)
grp = ET.SubElement(srobot, "group"); grp.set("name", NAME)
for jn in ("linear_guide_joint", "shoulder_pan_joint", "shoulder_lift_joint",
           "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"):
    ET.SubElement(grp, "joint").set("name", jn)
for l1, l2, reason in pairs:
    dc = ET.SubElement(srobot, "disable_collisions")
    dc.set("link1", l1); dc.set("link2", l2); dc.set("reason", reason)
# rail adjacencies (new links) -- keep cricket from flagging them
for l1, l2 in (("linear_guide_support_link", "linear_guide_rail_link"),
               ("linear_guide_rail_link", "linear_guide_carriage_link"),
               ("linear_guide_carriage_link", "base_link"),
               ("linear_guide_carriage_link", "base_link_inertia")):
    dc = ET.SubElement(srobot, "disable_collisions")
    dc.set("link1", l1); dc.set("link2", l2); dc.set("reason", "Adjacent")

srdf_tree = ET.ElementTree(srobot)
ET.indent(srdf_tree, space="  ")
srdf_path = os.path.join(OUT, f"{NAME}.srdf")
srdf_tree.write(srdf_path, encoding="utf-8", xml_declaration=True)
print("wrote", srdf_path, "with", len(pairs), "de-prefixed disable pairs")

# ------------------------------------------------------------------- JSON -- #
# NOTE: cricket sets the struct's internal `name` constant to lower(name), and VAMP's
# python binding names the submodule from that constant (robot_helper.hh def_submodule
# + __init__.py getattr(_core, robots()[i])). It therefore MUST equal the
# VAMP_ROBOT_MODULES entry `ur10e_rail`. lower("UR10eRail") == "ur10erail" (no underscore)
# would mismatch and break import/stub-gen -- so we name it "ur10e_rail" (a valid C++
# identifier), making struct == internal name == module name all "ur10e_rail".
cfg = {
    "name": NAME,
    "urdf": f"{NAME}_spherized.urdf",
    "srdf": f"{NAME}.srdf",
    "end_effector": "torch_tcp" if TORCH else "robotiq_85_base_link",
    "resolution": 32,
    "template": "templates/fk_template.hh",
    "subtemplates": [{"name": "ccfk", "template": "templates/ccfk_template.hh"}],
    "output": f"{NAME}.hh",
}
json_path = os.path.join(OUT, f"{NAME}.json")
with open(json_path, "w") as f:
    json.dump(cfg, f, indent=4); f.write("\n")
print("wrote", json_path)
