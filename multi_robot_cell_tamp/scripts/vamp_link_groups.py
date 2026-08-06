"""Per-link sphere groups for the VAMP engine's hierarchical broad phase.

WHY THIS EXISTS
---------------
The engine's original broad phase used ONE bounding sphere per sample over the whole
robot. Measured on the tower scene: it pruned **0.6 %** of the 7.66 M config-pair grid
-- worthless, exactly as ADR-0005 suspected. The reason is geometric: a UR10e's
arm+gripper spans ~0.9 m, and the two rails sit 1.6 m apart, so the two bounding
spheres overlap at essentially every pose.

Grouping the spheres **per link** fixes it. Each link's spheres stay bunched (link
bound radii here are 0.05-0.45 m), so a 17x17 group-pair test separates poses the
single sphere never could: it kills **60.9 %** of config pairs outright, and drops the
surviving narrow work by 9.2x in operation count.

WHERE THE GROUPS COME FROM
--------------------------
``cricket`` emits the robot's spheres in the spherized URDF's link order, one entry per
``<collision><geometry><sphere>``. So the URDF *is* the index table -- we parse it
rather than hard-coding indices, which keeps the engine robot-agnostic (a future robot
needs no code change, just its own spherized URDF).

The first ``n_structural`` spheres (rail/support/carriage, 18 for ur10e_rail) are
excluded from mu upstream, so groups are re-indexed against the post-exclusion array.

CHOICE OF GRANULARITY
---------------------
Per-link, not finer. Sub-splitting the two long links (upper_arm/forearm, 8 spheres
each) prunes more narrow work but costs more in the G^2 group test, and that trade is
net-negative -- measured total sphere-tests per config pair:

    per-link (G=17):  289 group + 755 narrow = 1044   <- best
    max 4/group (27): 729 group + 532 narrow = 1261
    max 2/group (45): 2025 group + 347 narrow = 2372

against 6400 for the unpruned check.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

# (link name, start, end) index ranges into the robot's sphere array.
Groups = List[Tuple[str, int, int]]

DEFAULT_SPHERIZED_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vamp_codegen", "inputs", "ur10e_rail_spherized.urdf")


def link_groups(urdf_path: str, n_structural: int = 0) -> Groups:
    """Sphere index ranges per link, re-indexed past the excluded structural prefix.

    Ranges are half-open ``[start, end)`` into the robot's sphere array *after* the
    first ``n_structural`` spheres have been dropped. Links wholly inside the
    structural prefix are omitted; the prefix is asserted to end on a link boundary,
    since a group straddling the cut would silently mis-index every later link.
    """
    root = ET.parse(urdf_path).getroot()

    groups: Groups = []
    cursor = 0
    for link in root.findall("link"):
        n = sum(1 for c in link.findall("collision") if c.find("geometry/sphere") is not None)
        if n:
            groups.append((link.get("name"), cursor, cursor + n))
            cursor += n

    if n_structural and not any(a == n_structural for _, a, _ in groups):
        raise ValueError(
            f"n_structural={n_structural} does not fall on a link boundary in {urdf_path}; "
            f"link starts are {[a for _, a, _ in groups]}")

    return [(name, a - n_structural, b - n_structural)
            for name, a, b in groups if a >= n_structural]


def n_spheres(urdf_path: str) -> int:
    """Total spheres in the spherized URDF -- cross-check against ``robot.n_spheres()``."""
    root = ET.parse(urdf_path).getroot()
    return sum(1 for link in root.findall("link")
               for c in link.findall("collision") if c.find("geometry/sphere") is not None)
