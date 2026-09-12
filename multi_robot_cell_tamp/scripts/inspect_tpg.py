#!/usr/bin/env python3
"""Read a temporal plan graph, and draw a piece of it.

    python3 scripts/inspect_tpg.py                                  # summary
    python3 scripts/inspect_tpg.py --node robot1:1023               # decode one node
    python3 scripts/inspect_tpg.py --svg tpg.svg                    # draw a window

``tpg.json`` is a partial order, not a schedule (ADR-0007), and its whole content is one
integer per node -- which makes it compact to store and hard to picture. This renders a
window of it in APEX-MR's terms:

* **type-1 edges** run along one robot's own node sequence: `n -> n+1`, "finish this
  configuration before starting the next". They are not stored anywhere, because the node
  ordering *is* them.
* **type-2 edges** cross between robots and are what ``deps`` holds: robot r may enter node
  `n` only once the other robot has reached `deps[r][n]`. The stored index is the colliding
  node's SUCCESSOR, so the figure marks the colliding configuration separately -- the gap
  between the two is the whole safety argument.

WHY THE FIGURE COMPRESSES ONE LANE
----------------------------------
The two lanes advance at unrelated rates. On the swap scene a 24-node window on robot1
refers to targets spread over 220 nodes of robot2, in two clusters, so drawing both lanes
at true scale is impossible on a page. The waiting robot's window is therefore contiguous
and literal, while the other lane draws only the nodes actually referenced (plus each
colliding predecessor) and marks each elided run with how many nodes it skipped. Type-1
edges spanning a break are dashed, so nothing pretends to be adjacent that is not.

The output is a standalone SVG: no plotting library (there is none in ``.venv_vamp``), and
vector output drops into the thesis as-is.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

FREE = -1

# Light-background palette: this is a figure for a document, not a dark-mode UI.
BG = "#ffffff"
LANE = {"a": "#1f4e79", "b": "#7a3e00"}          # robot A / robot B accents
INK = "#1a1a1a"
MUTED = "#6b7280"
TYPE1 = "#9ca3af"
TYPE2 = "#c2410c"
COLLIDE = "#dc2626"


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def locate(tpg: dict, robot: str, node: int) -> Tuple[str, int]:
    for seg in tpg["segments"][robot]:
        if seg["start_node"] <= node < seg["start_node"] + seg["n"]:
            return seg["task"], node - seg["start_node"]
    return "(past the end)", node


def summarise(tpg: dict) -> None:
    dt = float(tpg["delta_t"])
    print(f"robots            : {tpg['robots']}")
    print(f"delta_t           : {dt} s per node")
    mk = int(tpg["nominal_makespan_slots"])
    print(f"nominal makespan  : {mk} slots = {mk * dt:.2f} s   (reference only, not a contract)")
    print(f"edges             : {tpg['n_edges']} "
          f"({tpg.get('n_precedence_edges', 0)} from task precedences, the rest geometric)")
    margin = int(tpg.get("rigid_delay_margin_slots", -1))
    if margin >= 0:
        print(f"rigid delay margin: {margin} slots = {margin * dt:.2f} s   "
              f"(what the OLD executor could absorb, by luck)")
    print()
    for robot in tpg["robots"]:
        dep = tpg["deps"][robot]
        held = [n for n, v in enumerate(dep) if v != FREE]
        print(f"{robot}: {len(dep)} nodes, {len(held)} with a type-2 edge")
        for seg in tpg["segments"][robot]:
            lo, hi = seg["start_node"], seg["start_node"] + seg["n"] - 1
            n_held = sum(1 for n in held if lo <= n <= hi)
            print(f"    {seg['task']:10s} nodes {lo:5d}..{hi:<5d} ({seg['n']:4d})  "
                  f"{n_held:4d} constrained")
        for lo, hi in windows(held):
            other = tpg["robots"][1] if robot == tpg["robots"][0] else tpg["robots"][0]
            tgt = [dep[n] for n in range(lo, hi + 1) if dep[n] != FREE]
            print(f"    interaction window: nodes {lo}..{hi}  waits on {other} "
                  f"nodes {min(tgt)}..{max(tgt)}")
        print()


def windows(held: List[int], gap: int = 50) -> List[Tuple[int, int]]:
    """Group constrained nodes into runs, so the interaction regions are visible."""
    if not held:
        return []
    out, lo, prev = [], held[0], held[0]
    for n in held[1:]:
        if n - prev > gap:
            out.append((lo, prev))
            lo = n
        prev = n
    out.append((lo, prev))
    return out


def compress(needed: List[int], join: int = 2) -> List[Tuple[str, int]]:
    """Lay out a sparse node set: ``('node', i)`` items with ``('break', skipped)`` between.

    Runs closer than ``join`` are filled in rather than broken, so the figure does not
    fragment over a one-node hole.
    """
    items: List[Tuple[str, int]] = []
    prev = None
    for n in sorted(set(needed)):
        if prev is not None:
            hole = n - prev - 1
            if 0 < hole <= join:
                items += [("node", prev + 1 + k) for k in range(hole)]
            elif hole > join:
                items.append(("break", hole))
        items.append(("node", n))
        prev = n
    return items


def densest(dep: List[int], count: int):
    """Start of the most informative window: most type-2 edges, then narrowest.

    The first constrained node is a poor default -- on the tower scene it is an isolated
    single edge, which draws a figure that shows nothing. Prefer the window that carries the
    most edges, breaking ties toward the one whose targets sit closest together so the other
    lane stays compact.
    """
    best, best_key = None, None
    for start in range(0, max(1, len(dep) - count + 1)):
        tgt = [dep[n] for n in range(start, min(start + count, len(dep))) if dep[n] != FREE]
        if not tgt:
            continue
        key = (-len(tgt), abs((max(tgt) - min(tgt)) - count))
        if best_key is None or key < best_key:
            best, best_key = start, key
    return best


def draw(tpg: dict, robot: str, first: int, count: int, out_path: str,
         join: int = 2) -> None:
    robots = tpg["robots"]
    other = robots[1] if robot == robots[0] else robots[0]
    dep = tpg["deps"][robot]
    last = min(first + count - 1, len(dep) - 1)

    lane_a = [("node", n) for n in range(first, last + 1)]
    edges = [(n, int(dep[n])) for n in range(first, last + 1) if dep[n] != FREE]
    if not edges:
        print(f"warning: no type-2 edges on {robot} in nodes {first}..{last}; "
              f"the figure will show type-1 edges only")

    # The other lane shows each target and the colliding node it implies, nothing else.
    needed = sorted({v for _, v in edges} | {v - 1 for _, v in edges if v > 0})
    lane_b = compress(needed, join) if needed else []

    # ---- geometry --------------------------------------------------------- #
    # Each lane is spread across the SAME usable width, whatever its node count, so the
    # figure fills the frame instead of trailing off. The two lanes therefore have
    # different spacings -- which is honest: the x axis was never a shared time axis (one
    # lane is elided), only an ordering.
    #
    # The legend names the lanes (colour -> robot -> tasks), so the lanes need no label
    # gutter and the margins are equal by construction: lanes, legend and title all sit on
    # the page's vertical axis. The legend block is measured first because a long task list
    # can be what forces the canvas wider.
    r, y_a, y_b, height = 19, 150, 410, 580

    def tw(t: str, size: float, bold: bool = False) -> float:
        """Rough advance width of a string in the sans stack: enough to centre a block."""
        return len(t) * size * (0.58 if bold else 0.52)

    lanes = []
    for name, items in ((robot, lane_a), (other, lane_b)):
        nodes = [v for k, v in items if k == "node"]
        tasks = " / ".join(sorted({locate(tpg, name, n)[0] for n in nodes})) if nodes else ""
        lanes.append((name, f"{name} \u2014 {tasks}" if tasks else name, nodes))

    sw, gap, col_gap = 34, 8, 48
    col_a = [("lane_a", lanes[0][1]),
             ("line", "type-1: same robot, next configuration"),
             ("node", "node n = that robot's configuration at trajectory sample n")]
    col_b = [("lane_b", lanes[1][1]),
             ("curve", "type-2: wait for the other robot"),
             ("bad", "colliding sample; the edge points at its successor")]
    w_a = max(sw + gap + tw(t, 11) for _, t in col_a)
    w_b = max(sw + gap + tw(t, 11) for _, t in col_b)
    leg_w = w_a + col_gap + w_b

    pad = 48
    n_max = max(len(lane_a), len(lane_b), 2)
    width = max(940, int(leg_w) + 96, 2 * pad + (2 * r + 14) * (n_max - 1))
    span = width - 2 * pad

    def place(items):
        n = len(items)
        if n == 1:
            return {items[0][1]: width / 2} if items[0][0] == "node" else {}, {0: width / 2}
        step = span / (n - 1)
        nodes = {v: pad + i * step for i, (k, v) in enumerate(items) if k == "node"}
        gaps = {i: pad + i * step for i, (k, _) in enumerate(items) if k == "break"}
        return nodes, gaps

    xa, _ = place(lane_a)
    xb, gaps_b = place(lane_b) if lane_b else ({}, {})

    s: List[str] = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="Inter, Helvetica, Arial, sans-serif">')
    s.append(f'<rect width="{width}" height="{height}" fill="{BG}"/>')
    s.append('<defs>'
             f'<marker id="a1" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">'
             f'<path d="M0,0 L6,3 L0,6 z" fill="{TYPE1}"/></marker>'
             f'<marker id="a2" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">'
             f'<path d="M0,0 L7,3.5 L0,7 z" fill="{TYPE2}"/></marker>'
             '</defs>')

    def text(x, y, t, size=12, fill=INK, anchor="middle", weight="400"):
        s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
                 f'text-anchor="{anchor}" font-weight="{weight}">{t}</text>')

    def node(x, y, idx, fill, stroke):
        s.append(f'<circle cx="{x:.1f}" cy="{y}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="2"/>')
        text(x, y + 4, str(idx), 11 if len(str(idx)) < 4 else 9.5, INK)

    # ---- title ------------------------------------------------------------- #
    text(width / 2, 54, "TPG", 20, INK, "middle", "600")

    # ---- type-1 edges ------------------------------------------------------ #
    def type1(items, xmap, y):
        prev = None
        for kind, val in items:
            if kind == "break":
                continue
            if prev is not None:
                x0, x1 = xmap[prev], xmap[val]
                gap = val - prev
                dash = ' stroke-dasharray="5 4"' if gap > 1 else ""
                s.append(f'<line x1="{x0 + r:.1f}" y1="{y}" x2="{x1 - r - 5:.1f}" y2="{y}" '
                         f'stroke="{TYPE1}" stroke-width="2"{dash} marker-end="url(#a1)"/>')
                if gap > 1:
                    text((x0 + x1) / 2, y - r - 8, f"+{gap - 1}", 10, MUTED)
            prev = val

    type1(lane_a, xa, y_a)
    type1(lane_b, xb, y_b)

    # ---- type-2 edges (the only thing actually stored) --------------------- #
    targets = {v for _, v in edges}
    colliding = {v - 1 for _, v in edges if v > 0}
    for n, v in edges:
        if v not in xb or n not in xa:
            continue
        x0, x1 = xb[v], xa[n]
        cy = (y_a + y_b) / 2
        s.append(f'<path d="M{x0:.1f},{y_b - r} C{x0:.1f},{cy} {x1:.1f},{cy} '
                 f'{x1:.1f},{y_a + r + 6}" fill="none" stroke="{TYPE2}" stroke-width="1.8" '
                 f'opacity="0.75" marker-end="url(#a2)"/>')

    # ---- nodes ------------------------------------------------------------- #
    for _, n in lane_a:
        node(xa[n], y_a, n, "#eef2ff", LANE["a"])
    for kind, val in lane_b:
        if kind == "break":
            continue
        fill = "#fee2e2" if val in colliding else ("#fff7ed" if val in targets else "#f8fafc")
        node(xb[val], y_b, val, fill, COLLIDE if val in colliding else LANE["b"])
    for i, hole in [(i, v) for i, (k, v) in enumerate(lane_b) if k == "break"]:
        x = gaps_b[i]
        text(x, y_b + 5, "···", 17, MUTED)
        text(x, y_b + r + 16, f"{hole} skipped", 9.5, MUTED)

    # ---- legend (two columns, the block centred on the lanes' axis) -------- #
    ly = height - 90
    x_a = (width - leg_w) / 2
    x_b = x_a + w_a + col_gap
    swatch = {"lane_a": ("#eef2ff", LANE["a"], ""),
              "lane_b": ("#fff7ed", LANE["b"], ""),
              "node": ("#f1f5f9", MUTED, "n"),
              "bad": ("#fee2e2", COLLIDE, "")}

    def legend(x, rows):
        for k, (kind, label) in enumerate(rows):
            y = ly + 28 * k
            if kind == "line":
                s.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x + 30:.1f}" y2="{y}" '
                         f'stroke="{TYPE1}" stroke-width="2" marker-end="url(#a1)"/>')
            elif kind == "curve":
                s.append(f'<path d="M{x:.1f},{y + 6} C{x + 10:.1f},{y - 8} '
                         f'{x + 22:.1f},{y - 8} {x + 32:.1f},{y}" fill="none" '
                         f'stroke="{TYPE2}" stroke-width="1.8" marker-end="url(#a2)"/>')
            else:
                fill, stroke, inner = swatch[kind]
                s.append(f'<circle cx="{x + 13:.1f}" cy="{y - 4}" r="12" fill="{fill}" '
                         f'stroke="{stroke}" stroke-width="2"/>')
                if inner:
                    text(x + 13, y, inner, 11, INK)
            text(x + sw + gap, y + (4 if kind in ("line", "curve") else 0),
                 label, 11, INK, "start")

    legend(x_a, col_a)
    legend(x_b, col_b)
    s.append("</svg>")

    with open(out_path, "w") as f:
        f.write("\n".join(s) + "\n")
    print(f"wrote {out_path}: {robot} nodes {first}..{last}, {len(edges)} type-2 edge(s), "
          f"{len([1 for k, _ in lane_b if k == 'node'])} {other} nodes drawn, "
          f"{width}x{height} px")


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpg", default=os.path.join(pkg, "artifacts", "vamp", "tpg_refined.json"))
    p.add_argument("--node", help="decode one node, as robot:index")
    p.add_argument("--svg", help="write an SVG of a window of the graph")
    p.add_argument("--robot", help="which robot's nodes the window follows (default: the "
                                   "one with the most type-2 edges)")
    p.add_argument("--from", dest="first", type=int, default=None,
                   help="first node of the window (default: its first constrained node)")
    p.add_argument("--count", type=int, default=6, help="nodes in the window (default 6)")
    p.add_argument("--join", type=int, default=2,
                   help="on the other lane, fill gaps of at most this many nodes "
                        "instead of marking them as skipped (default 2)")
    args = p.parse_args(argv)

    tpg = load(args.tpg)

    if args.node:
        robot, _, idx = args.node.partition(":")
        n = int(idx)
        task, k = locate(tpg, robot, n)
        v = int(tpg["deps"][robot][n])
        other = tpg["robots"][1] if robot == tpg["robots"][0] else tpg["robots"][0]
        print(f"{robot} node {n} = {task} sample {k}")
        if v == FREE:
            print("  no type-2 edge: it may be entered as soon as the previous node is done")
        else:
            ot, ok = locate(tpg, other, v)
            ct, ck = locate(tpg, other, v - 1)
            print(f"  type-2 edge: needs {other} to have reached node {v} ({ot} sample {ok})")
            print(f"  i.e. {other} must have LEFT node {v - 1} ({ct} sample {ck}) -- "
                  f"the configuration that collides with this one")
        return 0

    if args.svg:
        robot = args.robot or max(tpg["robots"],
                                  key=lambda r: sum(1 for v in tpg["deps"][r] if v != FREE))
        first = args.first
        if first is None:
            first = densest(tpg["deps"][robot], args.count)
            if first is None:
                print(f"{robot} has no type-2 edges; pass --from explicitly")
                return 1
        draw(tpg, robot, first, args.count, args.svg, args.join)
        return 0

    summarise(tpg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
