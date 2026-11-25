"""The two figures of the paper, written to results/figures/ (PDF + PNG).

Fig. 1  (fig2_mh03)  EuRoC MH_03: position error over time, loose screen
                     without / with the gate, plus gate rejections per frame
                     with standstill blocks shaded.  Needs the two MH_03
                     trajectory logs and the EuRoC ground truth (--data).
Fig. 2  (fig3_suite) ATE on all 17 sequences for the three configurations.

Usage: python charts/make_figures.py --data <dir with euroc/> [--results results]
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
sys.path.insert(0, ROOT)

C_STD, C_AGG, C_GATE = "#64748B", "#C2410C", "#0284C7"
SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
        "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
        "V2_01_easy", "V2_02_medium", "V2_03_difficult",
        "room1", "room2", "room3", "room4", "room5", "room6"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def savefig(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def err_series(traj_file, seq, euroc_dir):
    from msckf.dataset import EurocDataset
    from msckf.evaluate import associate, align_umeyama
    ds = EurocDataset(os.path.join(euroc_dir, seq))
    tr = np.loadtxt(traj_file, delimiter=",", skiprows=1)
    ie, ig = associate(tr[:, 0], ds.gt_t)
    pe, pg = tr[ie, 1:4], ds.gt_p[ig]
    R, t = align_umeyama(pe, pg, yaw_only=True)
    return tr[ie, 0] - tr[0, 0], np.linalg.norm(pg - (pe @ R.T + t), axis=1), tr


def fig_mh03(traj_dir, euroc_dir):
    f0 = os.path.join(traj_dir, "agg_none__MH_03_medium.traj.csv")
    f1 = os.path.join(traj_dir, "agg_ratio__MH_03_medium.traj.csv")
    t0, e0, _ = err_series(f0, "MH_03_medium", euroc_dir)
    t1, e1, tr = err_series(f1, "MH_03_medium", euroc_dir)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3]})
    ax.plot(t0, e0, color=C_AGG, lw=1, label="aggressive admission, no gate")
    ax.plot(t1, e1, color=C_GATE, lw=1, label="with depth-observability gate")
    ax.set_yscale("log")
    ax.set_ylabel("position error [m]")
    ax.legend(fontsize=6.5, frameon=False)
    ax.annotate("motion onset\nafter standstill", xy=(10, max(e0.max() * 0.5, 1)),
                fontsize=6.5, color="#16202B")
    tt = tr[:, 0] - tr[0, 0]
    noupd, rej = tr[:, 4] < 0, tr[:, 5]
    start = None
    for i in range(len(tt)):
        if noupd[i] and start is None:
            start = i
        elif (not noupd[i] or i == len(tt) - 1) and start is not None:
            if i - start >= 20:
                ax2.axvspan(tt[start], tt[i - 1], color=C_STD, alpha=0.18, lw=0)
            start = None
    ax2.bar(tt, rej, width=0.3, color=C_GATE)
    ax2.set_ylabel("gate\nrejections")
    ax2.set_xlabel("time [s]")
    savefig(fig, "fig2_mh03")


def fig_suite(summary):
    rows = list(csv.DictReader(open(summary)))
    ate = {(r["config"], r["sequence"]): float(r["ate_se3_m"]) for r in rows}
    labels = [s.split("_")[0] + "_" + s.split("_")[1] if s[0] in "MV" else s
              for s in SEQS]
    x, w = np.arange(len(SEQS)), 0.27
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    for off, c, color, lab in [(-w, "std_none", C_STD, "standard (1.0°)"),
                               (0, "agg_none", C_AGG, "aggressive (0.3°), no gate"),
                               (w, "agg_ratio", C_GATE, "aggressive + gate")]:
        ax.bar(x + off, [ate[(c, s)] for s in SEQS], w, color=color, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("ATE RMSE [m]")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    savefig(fig, "fig3_suite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with euroc/ (ground truth for Fig. 1)")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"),
                    help="dir with summary.csv and trajectories/ (default: shipped results)")
    ap.add_argument("--traj-dir", default=None,
                    help="dir with agg_*__MH_03_medium.traj.csv (default: <results>/trajectories, or out/)")
    a = ap.parse_args()
    traj_dir = a.traj_dir or os.path.join(a.results, "trajectories")
    if not os.path.exists(os.path.join(traj_dir, "agg_ratio__MH_03_medium.traj.csv")):
        traj_dir = os.path.join(ROOT, "out")
    fig_mh03(traj_dir, os.path.join(a.data, "euroc"))
    fig_suite(os.path.join(a.results, "summary.csv"))


if __name__ == "__main__":
    main()
