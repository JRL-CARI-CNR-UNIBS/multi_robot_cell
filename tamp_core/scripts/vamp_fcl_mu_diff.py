#!/usr/bin/env python3
"""Diff two geometry-free seams (``tamp_problem.json``): an FCL reference vs a VAMP one.

Compares ``forbidden_offsets`` per trajectory-pair key and classifies the difference:

* EXTRA   (in VAMP, not in FCL) -- VAMP false positives. EXPECTED: sphere sets
  over-approximate the true robot volume, so VAMP forbids a superset. These cost
  makespan (the scheduler avoids offsets it needn't) but never make a plan unsound.
* MISSING (in FCL, not in VAMP) -- VAMP false negatives. These are UNSOUND: an offset
  FCL proved unsafe that VAMP declared safe. On a robot whose sphere model MATCHES the
  FCL collision model this count must be ZERO; a nonzero count is a hard failure.

PHASE-A NOTE: a real 1:1 diff needs the Phase-B ``vamp.ur10e_rail``. On the shipped
ur5 the geometry does not match the cell, so extras/missings here are geometry
artefacts -- this run only exercises the diff plumbing, not soundness.

    .venv_vamp/bin/python scripts/vamp_fcl_mu_diff.py FCL.json VAMP.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Set


def load_offsets(path: str) -> Dict[str, Set[int]]:
    with open(path) as f:
        art = json.load(f)
    return {k: set(int(o) for o in v) for k, v in art.get("forbidden_offsets", {}).items()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fcl", help="reference seam (C++ FCL collision_generator)")
    p.add_argument("vamp", help="candidate seam (collision_generator_vamp.py)")
    p.add_argument("--quiet", action="store_true", help="summary only, no per-key lines")
    args = p.parse_args(argv)

    fcl = load_offsets(args.fcl)
    vamp = load_offsets(args.vamp)
    keys = sorted(set(fcl) | set(vamp))

    total_extra = total_missing = total_agree = 0
    keys_only_fcl: List[str] = []
    keys_only_vamp: List[str] = []
    unsound_keys: List[str] = []

    for key in keys:
        f = fcl.get(key, set())
        v = vamp.get(key, set())
        extra = v - f       # VAMP false positives
        missing = f - v     # VAMP false negatives (unsound)
        total_extra += len(extra)
        total_missing += len(missing)
        total_agree += len(f & v)
        if key not in fcl:
            keys_only_vamp.append(key)
        if key not in vamp:
            keys_only_fcl.append(key)
        if missing:
            unsound_keys.append(key)
        if not args.quiet and (extra or missing):
            tag = "  UNSOUND" if missing else ""
            print(f"{key}: +{len(extra)} extra, -{len(missing)} missing{tag}")
            if missing:
                print(f"    missing (FCL forbids, VAMP allows): {sorted(missing)}")

    print("\n=== summary ===")
    print(f"keys: {len(keys)} total  ({len(fcl)} in FCL, {len(vamp)} in VAMP)")
    print(f"offsets agreeing (in both): {total_agree}")
    print(f"EXTRA   (VAMP false positives, expected): {total_extra}")
    print(f"MISSING (VAMP false negatives, UNSOUND):  {total_missing}")
    if keys_only_fcl:
        print(f"keys only in FCL ({len(keys_only_fcl)}): {keys_only_fcl}")
    if keys_only_vamp:
        print(f"keys only in VAMP ({len(keys_only_vamp)}): {keys_only_vamp}")

    if total_missing:
        print(f"\nFAIL: {total_missing} unsound missing offset(s) across "
              f"{len(unsound_keys)} key(s) -- VAMP allowed an offset FCL forbids.")
        return 1
    print("\nOK: no missing offsets (VAMP over-approximates FCL, as required).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
