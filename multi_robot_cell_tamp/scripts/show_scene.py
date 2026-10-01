#!/usr/bin/env python3
"""Show a scene YAML in RViz without running any pipeline stage.

    ros2 run multi_robot_cell_tamp show_scene.py numbers
    ros2 run multi_robot_cell_tamp show_scene.py /path/to/tamp_task_x.yaml

Needs the cell up (``ros2 launch multi_robot_cell_bringup start.launch.py``). The argument
is either a path or a scene name (``numbers`` -> ``config/tamp_task_numbers.yaml``; the
default ``nominal`` is ``tamp_task.yaml``).

What appears, both in the scene's own ``base_frame``:

* in the PLANNING SCENE (RViz "Planning Scene" display): fixtures and every object at its
  SPAWN pose -- the same call the executors make (``SceneVisualizer.publish_static``), which
  first clears whatever an earlier scene left behind;
* as MARKERS on ``/scene_preview`` (add a "MarkerArray" display for that topic): the PLACE
  poses as translucent orange shapes -- one per task, or one per SLOT -- plus text labels for
  every object id and every place/slot id, and the WELD points (see below).

Meshes (``mesh: {file, scale}`` on a fixture or object, origin = bbox centre): fixtures and
objects go to the planning scene as ``CollisionObject.meshes`` instead of a box. A PLACE
marker for a task whose object has a mesh is a translucent-orange ``MESH_RESOURCE``
(``file://`` the resolved STL, ``scale`` = the YAML's ``scale``, i.e. 1 for a metre-unit
file). A SLOT gets the mesh only when every candidate has the same one (file and scale);
otherwise the candidates differ, and a slot marker was always one box sized to the largest
candidate, which is what it stays. The box is the fallback for the same reason it is the
generator's: it is the one shape that stands for "whichever candidate lands here".

Weld points (``welds:``), namespaces ``weld_point`` / ``weld_label`` / ``weld_seam`` /
``weld_tool``: a small sphere at each ``start`` (a ``path:`` seam: its first waypoint) with its id
as a label; for a seam longer than 1 cm also a thin line to ``end`` (through every ``path:``
waypoint); and a short arrow at the tool (EE) pose the generator would
command at ``start`` -- from ``start + tool.position`` along the tool z axis (approach
direction), which is where to check the tilt of a spot weld. Hide any of them by namespace.

The node stays up so the markers persist; Ctrl-C leaves the planning-scene objects where
they are (the next run of any executor clears them).
"""

import argparse
import math
import os
import sys

import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_visualizer import SceneVisualizer  # noqa: E402


def resolve_scene(arg: str) -> str:
    if os.path.isfile(arg):
        return os.path.abspath(arg)
    share = os.path.join(get_package_share_directory("multi_robot_cell_tamp"), "config")
    name = "tamp_task.yaml" if arg == "nominal" else f"tamp_task_{arg}.yaml"
    path = os.path.join(share, name)
    if not os.path.isfile(path):
        have = sorted(f[len("tamp_task_"):-len(".yaml")] for f in os.listdir(share)
                      if f.startswith("tamp_task_") and f.endswith(".yaml"))
        raise SystemExit(f"no scene {arg!r}: not a file and {path} does not exist. "
                         f"Installed scenes: nominal, {', '.join(have)}")
    return path


def place_boxes(spec: dict, meshes: dict | None = None) -> list:
    """[(label, size, pose_dict, mesh_or_None)] for every destination the scene defines.

    ``meshes`` maps object id -> ``MeshAsset`` (``SceneVisualizer.object_mesh``); a task whose
    object is in it, or a slot whose candidates ALL share one mesh, carries that mesh, else
    None (draw the box).
    """
    meshes = meshes or {}
    sizes = {o["id"]: [float(v) for v in o["size"]] for o in spec.get("objects") or []}
    out = []
    for t in spec.get("tasks") or []:
        if "object" in t and t["object"] in sizes:
            out.append((t["id"], sizes[t["object"]], dict(t["place"]),
                        meshes.get(t["object"])))
    for s in spec.get("slots") or []:
        cids = [c for c in s.get("candidates") or [] if c in sizes]
        if cids:
            size = [max(sizes[c][i] for c in cids) for i in range(3)]
            shared = {(meshes[c].file, meshes[c].scale) if c in meshes else None for c in cids}
            mesh = meshes[cids[0]] if len(shared) == 1 and None not in shared else None
            out.append((s["id"], size, dict(s["place"]), mesh))
    return out


# A seam this short is a spot weld: nothing to draw between start and end.
SPOT_MAX_LENGTH = 0.01
WELD_POINT_DIAMETER = 0.010
WELD_TOOL_ARROW = 0.06


def weld_points(spec: dict, pose_from) -> list:
    """[(id, start_xyz, points_xyz, tool_position_xyz, tool_z_axis_xyz)] from ``welds:``.

    ``points_xyz`` is the seam polyline: the ``path:`` waypoints, or ``[start, end]``.

    The generator composes the EE pose as ``seam_point (x) tool`` with the seam point carrying
    no rotation, so the EE origin sits at ``start + tool.position`` and its z axis is the tool
    z rotated by ``tool``'s roll/pitch/yaw.
    """
    out = []
    for w in spec.get("welds") or []:
        raw = ([q for leg in w["legs"] for q in leg] if w.get("legs")      # a tack of spots
               else (w.get("path") or [w["start"], w["end"]]))
        pts = [tuple(float(p.get(k, 0.0)) for k in "xyz") for p in raw]
        st = pts[0]
        q = pose_from(w.get("tool") or {}).orientation
        ee = tuple(float((w.get("tool") or {}).get(k, 0.0)) for k in "xyz")
        # third column of the rotation matrix of q = the rotated z axis
        z = (2 * (q.x * q.z + q.w * q.y), 2 * (q.y * q.z - q.w * q.x),
             1 - 2 * (q.x * q.x + q.y * q.y))
        out.append((w["id"], st, pts, ee, z))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scene", nargs="?", default="nominal",
                    help="scene name (config/tamp_task_<name>.yaml) or a path")
    ap.add_argument("--no-markers", action="store_true", help="planning scene only")
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    path = resolve_scene(args.scene)

    rclpy.init()
    node = Node("show_scene")
    with open(path) as f:
        spec = yaml.safe_load(f)

    viz = SceneVisualizer(node, path)
    node.get_logger().info(f"scene: {path}")
    viz.publish_static()

    if not args.no_markers:
        qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        pub = node.create_publisher(MarkerArray, "/scene_preview", qos)
        frame = spec.get("base_frame", "world")
        markers = MarkerArray()

        def add(ns, ident, mtype, pose, scale, rgba, text=""):
            m = Marker()
            m.header.frame_id = frame
            m.ns, m.id, m.type, m.action = ns, ident, mtype, Marker.ADD
            m.pose, m.scale = pose, scale
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.text = text
            markers.markers.append(m)

        from geometry_msgs.msg import Point, Vector3
        i = 0
        for label, size, pose_d, mesh in place_boxes(spec, viz.object_mesh):
            pose = viz.pose_from(pose_d)
            if mesh is None:
                add("place", i, Marker.CUBE, pose, Vector3(x=size[0], y=size[1], z=size[2]),
                    (1.0, 0.55, 0.0, 0.35))
            else:
                add("place", i, Marker.MESH_RESOURCE, pose,
                    Vector3(x=mesh.scale, y=mesh.scale, z=mesh.scale), (1.0, 0.55, 0.0, 0.35))
                m = markers.markers[-1]
                m.mesh_resource = mesh.resource
                m.mesh_use_embedded_materials = False
            lab = viz.pose_from(pose_d)
            lab.position.z += size[2] / 2 + 0.05
            add("place_label", i, Marker.TEXT_VIEW_FACING, lab, Vector3(z=0.03),
                (1.0, 0.85, 0.4, 1.0), label)
            i += 1
        for j, (oid, (size, spawn_d, _g)) in enumerate(viz.objects.items()):
            lab = viz.pose_from(spawn_d)
            lab.position.z += size[2] / 2 + 0.05
            add("object_label", j, Marker.TEXT_VIEW_FACING, lab, Vector3(z=0.03),
                (0.6, 0.9, 1.0, 1.0), oid)

        n_welds = 0
        for k, (wid, st, pts, ee, z) in enumerate(weld_points(spec, viz.pose_from)):
            n_welds += 1
            d = WELD_POINT_DIAMETER
            add("weld_point", k, Marker.SPHERE, viz.pose_from(dict(zip("xyz", st))),
                Vector3(x=d, y=d, z=d), (1.0, 0.2, 0.2, 1.0))
            lab = viz.pose_from(dict(zip("xyz", st)))
            lab.position.z += 0.03
            add("weld_label", k, Marker.TEXT_VIEW_FACING, lab, Vector3(z=0.02),
                (1.0, 0.6, 0.6, 1.0), wid)
            if sum(math.dist(a, b) for a, b in zip(pts, pts[1:])) > SPOT_MAX_LENGTH:
                add("weld_seam", k, Marker.LINE_STRIP, viz.pose_from({}), Vector3(x=0.004),
                    (1.0, 0.2, 0.2, 1.0))
                markers.markers[-1].points = [Point(x=p[0], y=p[1], z=p[2]) for p in pts]
            tail = tuple(a + b for a, b in zip(st, ee))
            head = tuple(a + WELD_TOOL_ARROW * b for a, b in zip(tail, z))
            add("weld_tool", k, Marker.ARROW, viz.pose_from({}),
                Vector3(x=0.004, y=0.008, z=0.012), (1.0, 0.5, 0.2, 0.8))
            markers.markers[-1].points = [Point(x=p[0], y=p[1], z=p[2]) for p in (tail, head)]

        for _ in range(3):
            pub.publish(markers)
            rclpy.spin_once(node, timeout_sec=0.2)
        node.get_logger().info(
            f"scene: {len(viz.fixtures)} fixture(s), {len(viz.objects)} object(s) at spawn "
            f"(planning scene, {len(viz.fixture_mesh)} + {len(viz.object_mesh)} as meshes); "
            f"{i} place pose(s) and {n_welds} weld point(s) as markers on /scene_preview. "
            "Ctrl-C to quit.")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
