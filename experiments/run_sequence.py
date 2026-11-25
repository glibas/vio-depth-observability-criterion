"""Run one sequence with one gate setting.

Usage:
  python -m experiments.run_sequence <seq_dir> --gate {none,ratio,cond}
      [--gate-value V] [--out res.json] [--traj traj.csv] [--max-time S]

TUM-VI sequences (detected by dso/camchain.yaml) automatically use the
dataset's calibrated IMU noise.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msckf.config import Config, TUMVI_IMU
from msckf.dataset import load_dataset
from msckf.tracker import FeatureTracker
from msckf.filter import MSCKF
from msckf.evaluate import ate_rmse, rpe_drift


def run(seq_dir, cfg, max_time=None):
    if os.environ.get("DG_SINGLE_THREAD"):   # parallel batch runs: one core per run
        cv2.setNumThreads(1)
    ds = load_dataset(seq_dir)
    if ds.fisheye:  # TUM-VI
        for k, v in TUMVI_IMU.items():
            setattr(cfg, k, v)
    tracker = FeatureTracker(cfg, ds.K, ds.dist, fisheye=ds.fisheye)
    kf = MSCKF(cfg, ds.R_IC, ds.p_IC)
    kf.focal = ds.focal

    # static initialization at the first standstill window
    step = int(round(cfg.init_window_s * 200))
    init_end = None
    for s0 in range(0, len(ds.imu_t) - step, step // 4):
        seg = slice(s0, s0 + step)
        if ds.imu_acc[seg].var(axis=0).sum() < cfg.init_acc_var_thresh:
            kf.initialize_static(ds.imu_gyro[seg], ds.imu_acc[seg])
            init_end = ds.imu_t[seg][-1]
            break
    if init_end is None:
        seg = slice(0, step)
        kf.initialize_static(ds.imu_gyro[seg], ds.imu_acc[seg])
        init_end = ds.imu_t[seg][-1]

    theta_base = cfg.gate_value
    fb_ema = None
    fb_hist = []
    cam_start = int(np.searchsorted(ds.cam_t, init_end + cfg.start_shift_s))
    t_prev = init_end
    zupt_n = max(2, int(cfg.zupt_window_s * 200))
    traj_t, traj_p = [], []
    innov_log = []
    rej_log = []
    fb_log, ms_log = [], []
    prev_rej = 0
    t_wall = time.time()

    for ci in range(cam_start, len(ds.cam_t)):
        t_img = ds.cam_t[ci]
        if max_time is not None and t_img - ds.cam_t[cam_start] > max_time:
            break
        i0, i1 = ds.imu_between(t_prev, t_img)
        seg_t, seg_g, seg_a = ds.imu_segment(t_prev, t_img)
        if len(seg_t) >= 2:
            kf.propagate(seg_t, seg_g, seg_a)
        t_prev = t_img

        try:
            buf = np.fromfile(ds.cam_files[ci], dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        except OSError:
            img = None
        if img is None:
            continue
        t_frame = time.perf_counter()
        kf.augment(ci)
        st = tracker.process(ci, img)
        if cfg.adaptive_theta and cfg.gate == "ratio":
            fb = st.get("fb_med", -1.0)
            if fb > 0:
                fb_ema = fb if fb_ema is None else (
                    cfg.at_ema_alpha * fb + (1 - cfg.at_ema_alpha) * fb_ema)
                fb_hist.append(fb)
                if len(fb_hist) >= 40:
                    import statistics
                    fb_ref = statistics.median(fb_hist)
                    ratio = min(max(fb_ema / max(fb_ref, 1e-6),
                                    cfg.at_clip_lo), cfg.at_clip_hi)
                    cfg.gate_value = theta_base * ratio

        if cfg.zupt_enabled and i1 > zupt_n:
            if (ds.imu_acc[i1 - zupt_n:i1].var(axis=0).sum() < cfg.zupt_acc_var
                    and ds.imu_gyro[i1 - zupt_n:i1].var(axis=0).sum() < cfg.zupt_gyro_var):
                kf.update_zupt()

        window_full = len(kf.clones) > cfg.max_clones
        lost = [fid for fid, tr in list(tracker.tracks.items())
                if ci not in tr.obs]
        ready = [fid for fid in lost
                 if len(tracker.tracks[fid].obs) >= cfg.min_track_len]
        for fid in lost:
            if fid not in ready:
                tracker.tracks.pop(fid, None)
        if window_full:
            oldest = next(iter(kf.clones))
            for fid, tr in tracker.tracks.items():
                if oldest in tr.obs and len(tr.obs) >= cfg.min_track_len:
                    ready.append(fid)
        feats = tracker.pop_features(ready)
        ordered = dict(sorted(feats.items(), key=lambda kv: -len(kv[1])))
        if ordered:
            kf.update_features(ordered)
        innov_log.append(-1.0 if kf.last_innov_px is None else kf.last_innov_px)
        rej_log.append(kf.stats["n_gate_rejected"] - prev_rej)
        prev_rej = kf.stats["n_gate_rejected"]

        if window_full:
            old_fid = kf.marginalize_oldest()
            tracker.drop_frame_obs(old_fid)

        ms_log.append(1e3 * (time.perf_counter() - t_frame))
        fb_log.append(st.get("fb_med_all", -1.0))
        traj_t.append(t_img)
        traj_p.append(kf.p.copy())

    wall = time.time() - t_wall
    res = {**kf.stats, "n_frames": len(traj_t), "wall_s": round(wall, 1),
           "fps": round(len(traj_t) / max(wall, 1e-9), 1),
           "focal": float(ds.focal), "fisheye": bool(ds.fisheye),
           "frame_ms_mean": round(float(np.mean(ms_log)), 3) if ms_log else None,
           "gate_us_per_call": (round(kf.stats["gate_ns"] / kf.stats["gate_calls"] / 1e3, 3)
                                if kf.stats["gate_calls"] else None)}
    if traj_t:
        res["ate_se3_m"] = ate_rmse(traj_t, traj_p, ds.gt_t, ds.gt_p, yaw_only=False)
        res["ate_posyaw_m"] = ate_rmse(traj_t, traj_p, ds.gt_t, ds.gt_p, yaw_only=True)
        res["rpe_1s_m"] = rpe_drift(traj_t, traj_p, ds.gt_t, ds.gt_p)
        for k in ("ate_se3_m", "ate_posyaw_m", "rpe_1s_m"):
            if res[k] is not None:
                res[k] = round(res[k], 4)
    run.kf = kf  # gives main() access to the candidate log
    return res, (traj_t, traj_p, innov_log, rej_log, fb_log, ms_log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir")
    ap.add_argument("--gate", default="none", choices=["none", "ratio", "perp", "cond", "abs"])
    ap.add_argument("--parallax", type=float, default=1.0,
                    help="apparent-parallax admission threshold, deg")
    ap.add_argument("--gate-value", type=float, default=None)
    ap.add_argument("--adaptive-theta", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--start-shift", type=float, default=0.0,
                    help="delay the first camera frame by this many seconds")
    ap.add_argument("--cand-log", default=None,
                    help="write the per-candidate log (depth, b, b_perp, n_obs, reproj_px, outcome) as .npy")
    args = ap.parse_args()

    cfg = Config(gate=args.gate, min_parallax_deg=args.parallax,
                 adaptive_theta=args.adaptive_theta, start_shift_s=args.start_shift)
    if args.gate_value is not None:
        cfg.gate_value = args.gate_value
    elif args.gate == "cond":
        cfg.gate_value = 1e4

    res, (tt, tp, innov, rej, fb, ms) = run(args.seq_dir, cfg, max_time=args.max_time)
    res["gate"] = args.gate
    res["gate_value"] = cfg.gate_value
    res["parallax_deg"] = cfg.min_parallax_deg
    res["start_shift_s"] = cfg.start_shift_s
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": cfg.to_dict(), "result": res}, f, indent=1)
    if args.traj and tt:
        arr = np.hstack([np.array(tt).reshape(-1, 1), np.array(tp),
                         np.array(innov).reshape(-1, 1),
                         np.array(rej, dtype=float).reshape(-1, 1),
                         np.array(fb).reshape(-1, 1), np.array(ms).reshape(-1, 1)])
        np.savetxt(args.traj, arr, delimiter=",", fmt="%.6f",
                   header="t,px,py,pz,innov_px,gate_rej,fb_px,frame_ms")
    if args.cand_log:
        np.save(args.cand_log, np.array(run.kf.cand_log, dtype=np.float32).reshape(-1, 6))


if __name__ == "__main__":
    main()
