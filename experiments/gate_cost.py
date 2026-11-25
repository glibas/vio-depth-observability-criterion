"""Cost of the gate arithmetic in isolation.

Times (i) the b/d test exactly as in MSCKF.triangulate (norms of the camera
displacements, max, one division, one comparison), (ii) the transverse
b_perp/d variant, and (iii) one full triangulation (linear init + Gauss-Newton)
of the same track for scale. The in-run numbers (gate_us_per_call,
frame_ms_mean) are reported by run_sequence.py.

Usage: python experiments/gate_cost.py [--views 11] [--repeat 20000]
"""
import argparse
import os
import sys
import timeit
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msckf.config import Config
from msckf.filter import MSCKF, Clone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=11)
    ap.add_argument("--repeat", type=int, default=20000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    ps = np.cumsum(rng.normal(0, 0.02, (a.views, 3)), axis=0)
    depth, theta = 4.0, 0.0055
    ray = np.array([0.05, -0.02, 1.0]); ray /= np.linalg.norm(ray)

    def gate_ratio():
        disp = ps - ps[0]
        return np.sqrt((disp * disp).sum(axis=1)).max() / depth < theta

    def gate_perp():
        disp = ps - ps[0]
        perp = disp - np.outer(disp @ ray, ray)
        return np.sqrt((perp * perp).sum(axis=1)).max() / depth < theta

    cfg = Config(gate="none", min_parallax_deg=0.0, log_gate=False)
    kf = MSCKF(cfg, np.eye(3), np.zeros(3))
    kf.clones = OrderedDict((i, Clone(np.eye(3), np.array([0.03 * i, 0, 0]))) for i in range(a.views))
    L = np.array([0.2, 0.1, depth])
    obs = {i: tuple((L - c.p)[:2] / (L - c.p)[2]) for i, c in kf.clones.items()}

    def tri():
        return kf.triangulate(obs)

    res = {}
    for name, fn, n in (("b/d gate", gate_ratio, a.repeat), ("b_perp/d gate", gate_perp, a.repeat),
                        ("full triangulation", tri, max(200, a.repeat // 20))):
        t = min(timeit.repeat(fn, number=n, repeat=5)) / n
        res[name] = t * 1e6
        print(f"{name:20s} {t * 1e6:9.2f} us per call ({a.views} views, Python/NumPy)")
    print(f"gate / triangulation cost ratio: {res['b/d gate'] / res['full triangulation']:.3f}")
    return res


if __name__ == "__main__":
    main()
