#!/usr/bin/env python3
"""Free-space trajectory quality of a trajectory artifact: the yardstick of motion quality.

    python3 scripts/traj_quality.py A.json                 # summary + worst legs
    python3 scripts/traj_quality.py A.json B.json          # the same, side by side, and the legs
                                                           # whose path changed most
    options: --top N (worst legs listed, default 10), --elbow (UR elbow-branch flag), --json

A LEG is a maximal run of a free-space phase -- ToPick (0), Carrying (2), ToHome (4) -- of a
trajectory's resampled samples, taken from the sample before the run (where the previous phase
ended) to its last sample. So a pick-place has three legs (each an OMPL leg plus the Cartesian
approach or retreat that shares its phase), a weld one per commute. Per leg:

  path        joint-space path length, sum over steps of |dq| (rail metres and radians mixed, as
              the joint-space planner sees them)
  direct      |q_last - q_first|, the straight line between the leg's endpoints
  ratio       path / direct (1 = straight in joint space; undefined when direct is ~0)
  tv / net    per joint: total variation sum |dq_j| vs net change |q_last_j - q_first_j|
  max_rev_net the largest |net| of a revolute joint -- above pi, a wrist (or any joint) turned
              more than half a turn where the other way round was shorter
  rail        travel of the prismatic joints (m)
  moving      seconds with any joint moving (frozen samples, e.g. a hold, excluded)

The trajectory generator writes the same figures into each trajectory's `quality` block (since
2026-10-11); this script recomputes them from the positions, so it reads ANY artifact, old ones
included, and checks the block when there is one. Revolute vs prismatic: the block's `revolute`
mask, else the joint name (linear / rail / guide / prismatic / torso_lift = prismatic).

--elbow flags an elbow-branch change: a joint named *elbow* whose value changes sign over the
leg with |net| > 2 rad (meaningful on UR arms; off by default, the generic metrics are not).

System python3 + numpy; no ROS.
"""
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

FREE = {0: "to_pick", 2: "carrying", 4: "to_home"}
PRISMATIC_HINTS = ("linear", "rail", "guide", "prismatic", "torso_lift")


def revolute_mask(t):
    q = t.get("quality")
    if q and "revolute" in q:
        return np.array(q["revolute"], dtype=bool)
    return np.array([not any(h in n for h in PRISMATIC_HINTS) for n in t["joint_names"]])


def legs_of(t, dt, elbow=False):
    q = np.asarray(t["positions"], dtype=float)
    ph = list(t["phase"])
    rev = revolute_mask(t)
    i_elbow = [j for j, n in enumerate(t["joint_names"]) if "elbow" in n]
    out, i = [], 0
    for p, g in itertools.groupby(ph):
        n = len(list(g))
        first, last = max(i - 1, 0), i + n - 1
        i += n
        if p not in FREE:
            continue
        seg = q[first:last + 1]
        d = np.diff(seg, axis=0)
        step = np.sqrt((d * d).sum(1)) if len(d) else np.zeros(0)
        net = np.abs(seg[-1] - seg[0])
        tv = np.abs(d).sum(0) if len(d) else np.zeros(q.shape[1])
        direct = float(np.sqrt(((seg[-1] - seg[0]) ** 2).sum()))
        path = float(step.sum())
        leg = {
            "robot": t["robot"], "task": t["task"], "phase": FREE[p], "first": first, "last": last,
            "moving_s": int((step > 1e-9).sum()) * dt, "path": path, "direct": direct,
            "ratio": path / direct if direct > 1e-9 else None,
            "max_revolute_net": float(net[rev].max()) if rev.any() else 0.0,
            "rail_travel": float(tv[~rev].sum()), "tv": tv.tolist(), "net": net.tolist(),
        }
        if elbow:
            leg["elbow_branch"] = any(
                np.sign(seg[0, j]) != np.sign(seg[-1, j]) and net[j] > 2.0 for j in i_elbow)
        out.append(leg)
    return out


def check_block(t, legs):
    """Largest disagreement between the generator's `quality` block and the recomputation."""
    q = t.get("quality")
    if not q:
        return None
    if len(q["legs"]) != len(legs):
        return float("inf")
    worst = 0.0
    for a, b in zip(q["legs"], legs):
        if (a["first"], a["last"]) != (b["first"], b["last"]):
            return float("inf")
        for k in ("path", "direct", "max_revolute_net", "rail_travel", "moving_s"):
            worst = max(worst, abs(a[k] - b[k]) / max(1.0, abs(b[k])))
    return worst


def analyse(path, elbow):
    a = json.load(open(path))
    dt = a["delta_t"]
    legs, checks = [], []
    for t in a["trajectories"]:
        ls = legs_of(t, dt, elbow)
        legs += ls
        c = check_block(t, ls)
        if c is not None:
            checks.append(c)
    ratios = np.array([l["ratio"] for l in legs if l["ratio"] is not None])
    s = {
        "file": path, "trajectories": len(a["trajectories"]), "legs": len(legs),
        "ratio_median": float(np.median(ratios)) if len(ratios) else None,
        "ratio_p90": float(np.percentile(ratios, 90)) if len(ratios) else None,
        "ratio_max": float(ratios.max()) if len(ratios) else None,
        "legs_rev_net_gt_pi": sum(l["max_revolute_net"] > math.pi for l in legs),
        "path_total": sum(l["path"] for l in legs),
        "moving_s_total": sum(l["moving_s"] for l in legs),
        "rail_total": sum(l["rail_travel"] for l in legs),
        "quality_block": (f"checked on {len(checks)} trajectories, max rel. diff {max(checks):.1e}"
                          if checks else "absent"),
    }
    if elbow:
        s["legs_elbow_branch"] = sum(l.get("elbow_branch", False) for l in legs)
    return s, legs


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


ROWS = ["trajectories", "legs", "ratio_median", "ratio_p90", "ratio_max", "legs_rev_net_gt_pi",
        "legs_elbow_branch", "path_total", "moving_s_total", "rail_total", "quality_block"]


def worst_table(legs, top):
    ranked = sorted(legs, key=lambda l: -(l["ratio"] or 0.0))[:top]
    print(f"  worst {len(ranked)} legs by ratio:")
    print(f"    {'robot':8s}{'task':22s}{'leg':10s}{'ratio':>7s}{'path':>7s}{'direct':>8s}"
          f"{'maxrev':>8s}{'rail':>7s}{'moving':>8s}")
    for l in ranked:
        flag = " >pi" if l["max_revolute_net"] > math.pi else ""
        flag += " elbow" if l.get("elbow_branch") else ""
        print(f"    {l['robot']:8s}{l['task'][:21]:22s}{l['phase']:10s}{fmt(l['ratio']):>7s}"
              f"{l['path']:7.2f}{l['direct']:8.2f}{l['max_revolute_net']:8.2f}"
              f"{l['rail_travel']:7.2f}{l['moving_s']:8.2f}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("artifacts", nargs="+", help="one artifact, or two to compare (A then B)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--elbow", action="store_true", help="flag UR elbow-branch changes")
    ap.add_argument("--json", action="store_true", help="print the summaries as JSON")
    args = ap.parse_args()
    if len(args.artifacts) > 2:
        ap.error("one artifact, or two to compare")
    res = [analyse(p, args.elbow) for p in args.artifacts]
    if args.json:
        print(json.dumps([s for s, _ in res], indent=1))
        return 0
    rows = [r for r in ROWS if any(r in s for s, _ in res)]
    if len(res) == 1:
        s, legs = res[0]
        print(s["file"])
        for r in rows:
            print(f"  {r:20s} {fmt(s[r])}")
        worst_table(legs, args.top)
        return 0
    (sa, la), (sb, lb) = res
    print(f"A: {sa['file']}\nB: {sb['file']}")
    print(f"  {'':20s} {'A':>12s} {'B':>12s}")
    for r in rows:
        if r == "quality_block":
            continue
        va, vb = sa.get(r), sb.get(r)
        delta = (f"{vb - va:+.3f}" if isinstance(va, (int, float)) and isinstance(vb, (int, float))
                 and va is not None and vb is not None else "")
        print(f"  {r:20s} {fmt(va):>12s} {fmt(vb):>12s}  {delta}")
    # legs matched by (robot, task, phase, ordinal within the trajectory)
    def key(legs):
        seen, out = {}, {}
        for l in legs:
            k = (l["robot"], l["task"], l["phase"])
            seen[k] = seen.get(k, 0) + 1
            out[k + (seen[k],)] = l
        return out
    ka, kb = key(la), key(lb)
    common = sorted(ka.keys() & kb.keys())
    print(f"  legs matched by (robot, task, phase): {len(common)} "
          f"(A only {len(ka.keys() - kb.keys())}, B only {len(kb.keys() - ka.keys())})")
    if common:
        ch = sorted(common, key=lambda k: -abs(kb[k]["path"] - ka[k]["path"]))[:args.top]
        print(f"  largest path changes (A -> B):")
        for k in ch:
            a, b = ka[k], kb[k]
            print(f"    {k[0]:8s}{k[1][:21]:22s}{k[2]:10s} path {a['path']:6.2f} -> {b['path']:6.2f}"
                  f"  ratio {fmt(a['ratio'])} -> {fmt(b['ratio'])}"
                  f"  maxrev {a['max_revolute_net']:.2f} -> {b['max_revolute_net']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
