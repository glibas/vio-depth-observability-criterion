"""Controlled failure cases for the admission chain.

Feeds MSCKF.triangulate() with synthetic tracks whose geometry is known and
reports where each track leaves the chain: apparent-parallax screen, the
depth / Gauss-Newton guards, the depth-observability gate (b/d or b_perp/d),
the 2-px reprojection limit, or admitted. Pixel noise sigma_px = 1 px.

Usage: python experiments/failure_cases.py [--trials 500] [--out failure_cases.csv]
"""
import argparse
import csv
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msckf.config import Config
from msckf.filter import MSCKF, Clone
from msckf.geometry import so3_exp

F = 460.0          # focal length [px] (EuRoC-like)
N_VIEWS = 6
DEPTH = 5.0        # landmark distance [m]


def scenario(name, rng):
    """Camera centres and rotations (world <- camera) and the landmark, per case."""
    R0 = np.eye(3)                       # camera looks along world +z
    s = np.linspace(0, 1, N_VIEWS)
    off = np.radians(3.0)                # landmark bearing 3 deg off the optical axis
    L = DEPTH * np.array([np.sin(off), 0.0, np.cos(off)])
    Rs = [R0] * N_VIEWS
    outlier = False
    if name == "lateral 0.30 m (b/d = 0.06)":
        ps = [np.array([0.30 * k, 0, 0]) for k in s]
    elif name == "lateral 0.01 m (b/d = 0.002)":
        ps = [np.array([0.01 * k, 0, 0]) for k in s]
    elif name == "axial 0.50 m along the ray":
        u = L / np.linalg.norm(L)
        ps = [0.50 * k * u for k in s]
    elif name == "rotation only (5 deg yaw)":
        ps = [np.zeros(3)] * N_VIEWS
        Rs = [so3_exp(np.array([0, np.radians(5.0) * k, 0])) for k in s]
    elif name == "standstill (1 mm jitter)":
        ps = [rng.normal(0, 0.001, 3) for _ in s]
    elif name == "outlier track (one view +3 px)":
        ps = [np.array([0.30 * k, 0, 0]) for k in s]
        outlier = True
    elif name == "far landmark 80 m, lateral 0.30 m":
        ps = [np.array([0.30 * k, 0, 0]) for k in s]
        L = 80.0 * L / np.linalg.norm(L)
    else:
        raise KeyError(name)
    return ps, Rs, L, outlier


CASES = ["lateral 0.30 m (b/d = 0.06)", "lateral 0.01 m (b/d = 0.002)",
         "axial 0.50 m along the ray", "rotation only (5 deg yaw)",
         "standstill (1 mm jitter)", "outlier track (one view +3 px)",
         "far landmark 80 m, lateral 0.30 m"]
OUTCOMES = ["admitted", "parallax screen", "depth/GN guards", "depth gate", "reprojection limit"]


def one_trial(kf, name, rng, sigma_px=1.0):
    ps, Rs, L, outlier = scenario(name, rng)
    kf.clones = OrderedDict((i, Clone(R, p)) for i, (R, p) in enumerate(zip(Rs, ps)))
    obs = {}
    for i, (R, p) in enumerate(zip(Rs, ps)):
        pc = R.T @ (L - p)
        z = pc[:2] / pc[2] + rng.normal(0, sigma_px / F, 2)
        if outlier and i == N_VIEWS // 2:
            z = z + np.array([3.0 / F, 0.0])
        obs[i] = tuple(z)
    p_W = kf.triangulate(obs)
    if p_W is None:
        r = kf.last_reason
        if r == "gate":
            return "depth gate", None
        if r == "rej_parallax":
            return "parallax screen", None
        return "depth/GN guards", None
    if kf._last_reproj * F > kf.cfg.max_reproj_err_px:
        return "reprojection limit", None
    d_true = np.linalg.norm(L - ps[0])
    return "admitted", abs(np.linalg.norm(p_W - ps[0]) - d_true) / d_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--k-sigma", type=float, default=2.5, help="k * sigma_px [px]")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    theta = a.k_sigma / F
    rows = []
    for gate in ("none", "ratio", "perp"):
        cfg = Config(gate=gate, gate_value=theta, min_parallax_deg=0.3)
        kf = MSCKF(cfg, np.eye(3), np.zeros(3))
        kf.focal = F
        for name in CASES:
            rng = np.random.default_rng(0)
            cnt = dict.fromkeys(OUTCOMES, 0)
            errs = []
            for _ in range(a.trials):
                o, e = one_trial(kf, name, rng)
                cnt[o] += 1
                if e is not None:
                    errs.append(e)
            rows.append({"gate": gate, "case": name,
                         **{k: round(v / a.trials, 3) for k, v in cnt.items()},
                         "median_rel_depth_err_admitted": round(float(np.median(errs)), 3) if errs else None})
    w = max(len(c) for c in CASES)
    print(f"theta = {theta:.4f} rad (k*sigma_px = {a.k_sigma} px, f = {F:.0f} px), {a.trials} trials per case\n")
    print(f"{'gate':6} {'case':{w}} " + " ".join(f"{o[:12]:>12}" for o in OUTCOMES) + "  med.rel.err")
    for r in rows:
        print(f"{r['gate']:6} {r['case']:{w}} " + " ".join(f"{r[o]:12.3f}" for o in OUTCOMES)
              + f"  {r['median_rel_depth_err_admitted']}")
    if a.out:
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
