#!/usr/bin/env python3
"""Sanity checks for the codegen'd vamp.ur10e_rail module (ADR-0005 Phase B).

Run with the vamp venv:
    .venv_vamp/bin/python vamp_codegen/sanity_check.py
"""
import numpy as np
import vamp

R = vamp.ur10e_rail
print("robots in module:", vamp.robots)
print("dimension() =", R.dimension())
print("n_spheres() =", R.n_spheres())
assert R.dimension() == 7, "expected 7 DOF [rail, pan, lift, elbow, w1, w2, w3]"

# robot1 home from the trajectory artifact: [rail, pan, lift, elbow, w1, w2, w3]
home = [0.0, 0.1041, -2.3258, 1.9613, -1.215, -1.4927, -0.1041]

def spheres(q):
    s = R.fk(q)
    c = np.array([[p.x, p.y, p.z] for p in s])
    r = np.array([p.r for p in s])
    return c, r

c, r = spheres(home)
print(f"\nfk(home): {len(c)} spheres")
print(f"  x range [{c[:,0].min():.3f}, {c[:,0].max():.3f}]")
print(f"  y range [{c[:,1].min():.3f}, {c[:,1].max():.3f}]")
print(f"  z range [{c[:,2].min():.3f}, {c[:,2].max():.3f}]   (arm base baked at cell z~0.78)")
print(f"  radii   [{r.min():.4f}, {r.max():.4f}]")
assert c[:,2].max() > 0.9, "arm should reach above the ~0.78 m base height"
assert r.max() <= 0.401, "max radius should be the ~0.40 rail-cover sphere"

# --- rail-joint correctness: config[0] is a prismatic translation along +x ----
d = 1.0
c2, _ = spheres([home[0] + d] + home[1:])
shift = c2 - c
# every sphere must translate by +d in x, 0 in y,z (arm pose unchanged)
max_err = np.abs(shift - np.array([d, 0.0, 0.0])).max()
print(f"\nrail +{d} m: max per-sphere deviation from pure +x translation = {max_err:.2e} m")
assert max_err < 1e-4, "rail (config[0]) must be a pure +x prismatic shift of all spheres"

# --- eefk sane -----------------------------------------------------------------
ee = np.asarray(R.eefk(home))
print(f"\neefk(home) translation = {ee[:3,3].round(3).tolist()}  (end_effector={R.end_effector()})")

print("\nALL SANITY CHECKS PASSED")
