"""Configuration for the vanilla MSCKF + depth-observability gate.

Deliberately minimal: the only experimental switch is `gate` / `gate_value`.
Everything else is a textbook default, identical across all experiments.
"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- sliding window ---
    max_clones: int = 11
    max_updates_per_frame: int = 40

    # --- feature tracking (standard KLT front-end) ---
    max_features: int = 200
    grid_rows: int = 5
    grid_cols: int = 6
    max_per_grid: int = 8
    min_feat_dist: float = 10.0
    fast_threshold: int = 15
    klt_win: int = 21
    klt_levels: int = 4
    klt_fb_err: float = 0.5
    ransac_thresh_px: float = 1.0
    histogram_equalize: bool = True

    # --- measurement model (textbook: hard 95% chi-square gate) ---
    sigma_px: float = 1.5
    chi2_mult: float = 1.0
    min_track_len: int = 4
    max_depth: float = 60.0
    min_depth: float = 0.2
    max_reproj_err_px: float = 2.0
    min_parallax_deg: float = 1.0     # the STANDARD apparent-parallax check

    # --- THE experimental switch ---
    gate: str = "none"                # none | ratio | perp | cond | abs
    gate_value: float = 0.005         # ratio/perp: min b/d [rad]; cond: max cond number
    log_gate: bool = True             # log b, b_perp, depth, outcome per candidate (no effect on the estimate)
    start_shift_s: float = 0.0        # delay the first camera frame (run-to-run variability)
    adaptive_theta: bool = False      # per-frame theta_t = theta * clip(fb_ema/fb_ref)
    at_ema_alpha: float = 0.2         # EMA smoothing of the fb-error signal
    at_clip_lo: float = 0.5           # theta modulation clamps
    at_clip_hi: float = 4.0

    # --- IMU noise (EuRoC defaults; override per dataset) ---
    sigma_g: float = 1.6968e-4
    sigma_a: float = 2.0e-3
    sigma_wg: float = 1.9393e-5
    sigma_wa: float = 3.0e-3
    imu_noise_mult: float = 5.0
    gravity: float = 9.81007

    # --- initialization / ZUPT (standard practice) ---
    init_window_s: float = 0.75
    init_acc_var_thresh: float = 0.25
    init_att_std_deg: float = 2.0
    init_bg_std: float = 0.02
    init_ba_std: float = 0.15
    zupt_enabled: bool = True
    zupt_acc_var: float = 0.02
    zupt_gyro_var: float = 2e-4
    zupt_window_s: float = 0.4
    zupt_sigma_v: float = 0.02

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        cfg = cls()
        for k, v in d.items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown config key: {k}")
            setattr(cfg, k, v)
        return cfg


TUMVI_IMU = dict(sigma_g=0.00016, sigma_a=0.0028, sigma_wg=0.000022,
                 sigma_wa=0.00086, imu_noise_mult=1.0, gravity=9.80766)
