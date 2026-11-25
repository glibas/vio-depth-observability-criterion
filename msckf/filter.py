"""Error-state EKF with MSCKF visual updates.

State: R_WI, p_WI, v_WI, b_g, b_a  +  sliding window of camera clones.
Error state (15 + 6M): [dtheta_I, dp, dv, dbg, dba, {dtheta_Ci, dp_Ci}].
Rotation error is local/right: R = R_hat @ Exp(dtheta).
"""
from collections import OrderedDict
from time import perf_counter_ns
import numpy as np
from scipy.stats import chi2 as chi2_dist

from .geometry import skew, so3_exp, so3_right_jacobian

CHI2_95 = {i: chi2_dist.ppf(0.95, i) for i in range(1, 200)}


class Clone:
    __slots__ = ("R", "p")

    def __init__(self, R, p):
        self.R = R
        self.p = p


class MSCKF:
    def __init__(self, cfg, R_IC, p_IC):
        self.cfg = cfg
        self.R_IC = R_IC
        self.p_IC = p_IC
        self.g = np.array([0.0, 0.0, -cfg.gravity])

        self.R = np.eye(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.clones = OrderedDict()   # frame_id -> Clone
        self.P = np.zeros((15, 15))
        self.stats = {"n_tried": 0, "n_tri_fail": 0, "n_gate_fail": 0,
                      "n_gate_rejected": 0, "n_used": 0,
                      # where each candidate leaves the admission chain
                      "rej_few_views": 0, "rej_parallax": 0, "rej_dlt": 0,
                      "rej_depth_lin": 0, "rej_gn": 0, "rej_rho": 0,
                      "rej_depth": 0, "rej_reproj": 0, "rej_jac": 0,
                      # in-run cost of the gate arithmetic
                      "gate_ns": 0, "gate_calls": 0}
        self.cand_log = []          # per candidate that reached the gate
        self.last_reason = None
        self._last_geom = None
        self.last_innov_px = None  # RMS residual (px) of last accepted set

        m = cfg.imu_noise_mult
        self.sg2 = (cfg.sigma_g * m) ** 2
        self.sa2 = (cfg.sigma_a * m) ** 2
        self.swg2 = (cfg.sigma_wg * m) ** 2
        self.swa2 = (cfg.sigma_wa * m) ** 2

    # ------------------------------------------------------------- init
    def initialize_static(self, gyro, acc):
        cfg = self.cfg
        self.bg = gyro.mean(axis=0)
        a_mean = acc.mean(axis=0)
        # align measured gravity direction with +Z (a_m measures -g in body)
        # body-frame "up" direction: accelerometer measures -g at rest
        z = a_mean / np.linalg.norm(a_mean)
        x = np.array([1.0, 0.0, 0.0])
        x = x - z * (z @ x)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([0.0, 1.0, 0.0]) - z * z[1]
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        # rows (x, y, z) are orthonormal and map z to e_z: this is R_WI
        self.R = np.vstack([x, y, z])
        self.ba = a_mean - self.R.T @ (-self.g)

        d2r = np.pi / 180.0
        P = np.zeros((15, 15))
        P[0:2, 0:2] = np.eye(2) * (cfg.init_att_std_deg * d2r) ** 2
        P[2, 2] = (0.1 * d2r) ** 2          # yaw: defined as zero by choice
        P[3:6, 3:6] = np.eye(3) * 1e-8      # position defined as origin
        P[6:9, 6:9] = np.eye(3) * 1e-4     # velocity ~ 0 (standstill)
        P[9:12, 9:12] = np.eye(3) * cfg.init_bg_std ** 2
        P[12:15, 12:15] = np.eye(3) * cfg.init_ba_std ** 2
        self.P = P

    # ------------------------------------------------------- propagation
    def propagate(self, t, gyro, acc):
        """Propagate through IMU samples. t, gyro, acc: arrays (N,), (N,3), (N,3).
        Uses midpoint gyro/accel between consecutive samples."""
        n_clones = len(self.clones)
        D = 15 + 6 * n_clones
        for k in range(len(t) - 1):
            dt = t[k + 1] - t[k]
            if dt <= 0:
                continue
            w = 0.5 * (gyro[k] + gyro[k + 1]) - self.bg
            a = 0.5 * (acc[k] + acc[k + 1]) - self.ba
            R0 = self.R
            Ra = R0 @ a
            dR = so3_exp(w * dt)

            # error-state transition (IMU block)
            Phi = np.eye(15)
            Phi[0:3, 0:3] = dR.T
            Phi[0:3, 9:12] = -so3_right_jacobian(w * dt) * dt
            Phi[3:6, 0:3] = -R0 @ skew(a) * (0.5 * dt * dt)
            Phi[3:6, 6:9] = np.eye(3) * dt
            Phi[3:6, 12:15] = -R0 * (0.5 * dt * dt)
            Phi[6:9, 0:3] = -R0 @ skew(a) * dt
            Phi[6:9, 12:15] = -R0 * dt

            Qc = np.zeros((15, 15))
            Qc[0:3, 0:3] = np.eye(3) * self.sg2
            Qc[6:9, 6:9] = np.eye(3) * self.sa2
            Qc[9:12, 9:12] = np.eye(3) * self.swg2
            Qc[12:15, 12:15] = np.eye(3) * self.swa2
            Qd = Phi @ Qc @ Phi.T * dt

            Pii = self.P[0:15, 0:15]
            self.P[0:15, 0:15] = Phi @ Pii @ Phi.T + Qd
            if n_clones:
                self.P[0:15, 15:D] = Phi @ self.P[0:15, 15:D]
                self.P[15:D, 0:15] = self.P[0:15, 15:D].T

            # nominal state
            self.p = self.p + self.v * dt + 0.5 * (Ra + self.g) * dt * dt
            self.v = self.v + (Ra + self.g) * dt
            self.R = R0 @ dR
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------ augmentation
    def augment(self, frame_id):
        R_WC = self.R @ self.R_IC
        p_WC = self.p + self.R @ self.p_IC
        D = self.P.shape[0]
        J = np.zeros((6, D))
        J[0:3, 0:3] = self.R_IC.T
        J[3:6, 0:3] = -self.R @ skew(self.p_IC)
        J[3:6, 3:6] = np.eye(3)
        Pnew = np.zeros((D + 6, D + 6))
        Pnew[:D, :D] = self.P
        Pnew[D:, :D] = J @ self.P
        Pnew[:D, D:] = Pnew[D:, :D].T
        Pnew[D:, D:] = J @ self.P @ J.T
        self.P = Pnew
        self.clones[frame_id] = Clone(R_WC.copy(), p_WC.copy())

    def marginalize_oldest(self):
        fid = next(iter(self.clones))
        idx = 15 + 6 * list(self.clones.keys()).index(fid)
        keep = np.r_[np.arange(0, idx), np.arange(idx + 6, self.P.shape[0])]
        self.P = self.P[np.ix_(keep, keep)]
        del self.clones[fid]
        return fid

    # ----------------------------------------------------- triangulation
    def triangulate(self, obs):
        """obs: {frame_id: (u, v)} normalized coords. Returns p_W or None."""
        cfg = self.cfg
        fids = [f for f in obs if f in self.clones]
        self._last_geom = None
        if len(fids) < 2:
            return self._reject("rej_few_views")
        anchor = self.clones[fids[0]]
        R_WA, p_WA = anchor.R, anchor.p
        Rs, ts, zs = [], [], []
        for f in fids:
            c = self.clones[f]
            Rs.append(c.R.T @ R_WA)
            ts.append(c.R.T @ (p_WA - c.p))
            zs.append(np.array(obs[f]))
        Rs, ts, zs = np.array(Rs), np.array(ts), np.array(zs)

        # baseline check: depth is unobservable without camera translation
        # parallax check between first and last ray (in anchor frame)
        r0 = np.array([zs[0][0], zs[0][1], 1.0])
        r0 /= np.linalg.norm(r0)
        rl = Rs[-1].T @ np.array([zs[-1][0], zs[-1][1], 1.0])
        rl /= np.linalg.norm(rl)
        parallax = np.degrees(np.arccos(np.clip(r0 @ rl, -1, 1)))
        if parallax < self.cfg.min_parallax_deg:
            return self._reject("rej_parallax")

        # linear init from first & last view
        A = np.zeros((4, 4))
        for i in (0, len(fids) - 1):
            Pm = np.hstack([Rs[i], ts[i].reshape(3, 1)])
            A[2 * (i > 0) + 0] = zs[i][0] * Pm[2] - Pm[0]
            A[2 * (i > 0) + 1] = zs[i][1] * Pm[2] - Pm[1]
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-12:
            return self._reject("rej_dlt")
        pA = X[:3] / X[3]
        if pA[2] < cfg.min_depth or pA[2] > cfg.max_depth:
            return self._reject("rej_depth_lin")
        x = np.array([pA[0] / pA[2], pA[1] / pA[2], 1.0 / pA[2]])  # (alpha, beta, rho)

        # Gauss-Newton refinement
        lam = 1e-3
        cost_prev = None
        for _ in range(10):
            Jm = np.zeros((2 * len(fids), 3))
            r = np.zeros(2 * len(fids))
            for i in range(len(fids)):
                h = Rs[i] @ np.array([x[0], x[1], 1.0]) + x[2] * ts[i]
                if h[2] < 1e-6:
                    return self._reject("rej_gn")
                zhat = h[:2] / h[2]
                r[2 * i:2 * i + 2] = zs[i] - zhat
                Jpi = np.array([[1 / h[2], 0, -h[0] / h[2] ** 2],
                                [0, 1 / h[2], -h[1] / h[2] ** 2]])
                dh = np.column_stack([Rs[i][:, 0], Rs[i][:, 1], ts[i]])
                Jm[2 * i:2 * i + 2] = Jpi @ dh
            cost = r @ r
            H = Jm.T @ Jm
            g = Jm.T @ r
            try:
                dx = np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
            except np.linalg.LinAlgError:
                return self._reject("rej_gn")
            x = x + dx
            if cost_prev is not None and abs(cost_prev - cost) < 1e-10:
                break
            cost_prev = cost
        if x[2] <= 0:
            return self._reject("rej_rho")
        depth = 1.0 / x[2]
        if depth < cfg.min_depth or depth > cfg.max_depth:
            return self._reject("rej_depth")
        # ---- THE experimental switch ----
        ps = np.array([self.clones[f].p for f in fids])   # camera centres, world
        if cfg.gate in ("ratio", "perp"):
            t0 = perf_counter_ns()
        disp = ps - ps[0]
        # b: largest camera displacement from the first observing pose
        baseline = np.sqrt((disp * disp).sum(axis=1)).max()
        if cfg.gate == "ratio":
            reject = baseline / depth < cfg.gate_value
            self.stats["gate_ns"] += perf_counter_ns() - t0
            self.stats["gate_calls"] += 1
        # b_perp: displacement component orthogonal to the landmark bearing
        if cfg.gate == "perp" or cfg.log_gate:
            ray = R_WA @ np.array([x[0], x[1], 1.0])
            ray /= np.linalg.norm(ray)
            perp = disp - np.outer(disp @ ray, ray)
            b_perp = np.sqrt((perp * perp).sum(axis=1)).max()
        if cfg.gate == "perp":
            reject = b_perp / depth < cfg.gate_value
            self.stats["gate_ns"] += perf_counter_ns() - t0
            self.stats["gate_calls"] += 1
        if cfg.log_gate:
            self._last_geom = (depth, baseline, b_perp, len(fids))
        if cfg.gate in ("ratio", "perp") and reject:
            self.stats["n_gate_rejected"] += 1
            self._log_candidate(3)
            return self._reject("gate")
        elif cfg.gate == "abs":
            # absolute-baseline variant (msckf_vio-style), for comparison
            if baseline < cfg.gate_value:
                self.stats["n_gate_rejected"] += 1
                return self._reject("gate")
        elif cfg.gate == "cond":
            ev = np.linalg.eigvalsh(H)
            if ev[0] <= 0 or ev[-1] / ev[0] > cfg.gate_value:
                self.stats["n_gate_rejected"] += 1
                return self._reject("gate")
        pA = np.array([x[0], x[1], 1.0]) * depth
        # mean reprojection error (in pixels via focal approx handled by caller)
        errs = []
        for i in range(len(fids)):
            h = Rs[i] @ np.array([x[0], x[1], 1.0]) + x[2] * ts[i]
            if h[2] < 1e-6:
                return self._reject("rej_gn")
            errs.append(np.linalg.norm(zs[i] - h[:2] / h[2]))
        self._last_reproj = float(np.mean(errs))
        self.last_reason = None
        return R_WA @ pA + p_WA

    def _reject(self, key):
        """Count where a candidate left the admission chain; returns None."""
        self.last_reason = key
        if key in self.stats:
            self.stats[key] += 1
        return None

    def _log_candidate(self, outcome, reproj_px=np.nan):
        """outcome: 0 used, 1 reprojection limit, 2 chi2 test, 3 gate, 4 jacobian."""
        if self.cfg.log_gate and self._last_geom is not None:
            self.cand_log.append((*self._last_geom, reproj_px, outcome))

    # ------------------------------------------------------------ update
    def _feature_jacobian(self, p_W, obs):
        """Returns r, H_x (rows already nullspace-projected) or None."""
        fids = [f for f in obs if f in self.clones]
        clone_keys = list(self.clones.keys())
        D = self.P.shape[0]
        n = len(fids)
        r = np.zeros(2 * n)
        Hx = np.zeros((2 * n, D))
        Hf = np.zeros((2 * n, 3))
        for i, f in enumerate(fids):
            c = self.clones[f]
            pC = c.R.T @ (p_W - c.p)
            if pC[2] < 1e-3:
                return None
            zhat = pC[:2] / pC[2]
            r[2 * i:2 * i + 2] = np.array(obs[f]) - zhat
            j = 15 + 6 * clone_keys.index(f)
            Jpi = np.array([[1 / pC[2], 0, -pC[0] / pC[2] ** 2],
                            [0, 1 / pC[2], -pC[1] / pC[2] ** 2]])
            Hx[2 * i:2 * i + 2, j:j + 3] = Jpi @ skew(pC)
            Hx[2 * i:2 * i + 2, j + 3:j + 6] = -Jpi @ c.R.T
            Hf[2 * i:2 * i + 2, :] = Jpi @ c.R.T
        # left-nullspace projection of Hf
        Q, _ = np.linalg.qr(Hf, mode="complete")
        N = Q[:, 3:]
        return N.T @ r, N.T @ Hx

    def update_features(self, feature_obs):
        """feature_obs: {fid: {frame_id: (u,v)}}. Returns n_used."""
        cfg = self.cfg
        sigma = cfg.sigma_px / getattr(self, "focal", 460.0)
        r_list, H_list = [], []
        n_used = 0
        for fid, obs in feature_obs.items():
            if n_used >= cfg.max_updates_per_frame:
                break
            obs_in = {f: z for f, z in obs.items() if f in self.clones}
            if len(obs_in) < cfg.min_track_len:
                continue
            self.stats["n_tried"] += 1
            p_W = self.triangulate(obs_in)
            if p_W is None:
                self.stats["n_tri_fail"] += 1
                continue
            reproj_px = self._last_reproj * getattr(self, "focal", 460.0)
            if reproj_px > cfg.max_reproj_err_px:
                self.stats["n_tri_fail"] += 1
                self.stats["rej_reproj"] += 1
                self._log_candidate(1, reproj_px)
                continue
            out = self._feature_jacobian(p_W, obs_in)
            if out is None:
                self.stats["rej_jac"] += 1
                self._log_candidate(4, reproj_px)
                continue
            r_o, H_o = out
            dof = len(r_o)
            S = H_o @ self.P @ H_o.T + sigma ** 2 * np.eye(dof)
            try:
                maha = r_o @ np.linalg.solve(S, r_o)
            except np.linalg.LinAlgError:
                continue
            if maha > cfg.chi2_mult * CHI2_95.get(dof, 1e9):
                self.stats["n_gate_fail"] += 1
                self._log_candidate(2, reproj_px)
                continue
            self.stats["n_used"] += 1
            self._log_candidate(0, reproj_px)
            r_list.append(r_o)
            H_list.append(H_o)
            n_used += 1
        if not r_list:
            self.last_innov_px = None
            return 0
        r = np.concatenate(r_list)
        self.last_innov_px = float(np.sqrt(np.mean(r ** 2)) * getattr(self, "focal", 460.0))
        H = np.vstack(H_list)
        # QR compression
        D = self.P.shape[0]
        if H.shape[0] > D:
            Q, Rq = np.linalg.qr(H, mode="reduced")
            keep = np.abs(np.diag(Rq)).cumsum() > 0  # all rows of reduced QR
            H = Rq
            r = Q.T @ r
        self._ekf_update(r, H, sigma ** 2)
        return n_used

    def update_zupt(self):
        D = self.P.shape[0]
        H = np.zeros((3, D))
        H[:, 6:9] = np.eye(3)
        r = -self.v
        self._ekf_update(r, H, self.cfg.zupt_sigma_v ** 2)

    def _ekf_update(self, r, H, meas_var):
        P = self.P
        S = H @ P @ H.T + meas_var * np.eye(len(r))
        try:
            K = np.linalg.solve(S, H @ P).T
        except np.linalg.LinAlgError:
            return
        dx = K @ r
        I_KH = np.eye(P.shape[0]) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ (meas_var * np.eye(len(r))) @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        # retract
        self.R = self.R @ so3_exp(dx[0:3])
        self.p += dx[3:6]
        self.v += dx[6:9]
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        for i, c in enumerate(self.clones.values()):
            j = 15 + 6 * i
            c.R = c.R @ so3_exp(dx[j:j + 3])
            c.p += dx[j + 3:j + 6]
