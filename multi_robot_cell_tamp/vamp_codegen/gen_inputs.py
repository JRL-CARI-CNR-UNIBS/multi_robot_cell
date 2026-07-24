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
import xml.etree.ElementTree as ET
import numpy as np
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
SHIPPED = "/home/riccardo_dolci/ompl_icra_ws/cricket_src/resources/ur10e/ur_10e_spherized.urdf"
MOVEIT_SRDF = "/home/riccardo_dolci/ws_thesis/src/multi_robot_cell/multi_robot_moveit_config/config/multi_robot_cell.srdf"
OUT = os.environ.get("VAMP_CODEGEN_INPUTS", HERE)
os.makedirs(OUT, exist_ok=True)

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
root.set("name", "ur10e_rail")
if "path" in root.attrib:
    del root.attrib["path"]

# strip visuals (drop mesh dependency) and inertials (unneeded for fk/collision)
for link in root.findall("link"):
    for tag in ("visual", "inertial"):
        for e in link.findall(tag):
            link.remove(e)

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

urdf_path = os.path.join(OUT, "ur10e_rail_spherized.urdf")
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
        pairs.append((deprefix(l1), deprefix(l2), dc.get("reason", "Never")))

srobot = ET.Element("robot"); srobot.set("name", "ur10e_rail")
grp = ET.SubElement(srobot, "group"); grp.set("name", "ur10e_rail")
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
srdf_path = os.path.join(OUT, "ur10e_rail.srdf")
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
    "name": "ur10e_rail",
    "urdf": "ur10e_rail_spherized.urdf",
    "srdf": "ur10e_rail.srdf",
    "end_effector": "robotiq_85_base_link",
    "resolution": 32,
    "template": "templates/fk_template.hh",
    "subtemplates": [{"name": "ccfk", "template": "templates/ccfk_template.hh"}],
    "output": "ur10e_rail.hh",
}
json_path = os.path.join(OUT, "ur10e_rail.json")
with open(json_path, "w") as f:
    json.dump(cfg, f, indent=4); f.write("\n")
print("wrote", json_path)
