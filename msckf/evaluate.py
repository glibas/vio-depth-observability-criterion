"""Trajectory evaluation: ATE (SE3 and posyaw alignment) and RPE."""
import numpy as np


def associate(t_est, t_gt, max_dt=0.02):
    """Nearest-neighbour timestamp association. Returns (idx_est, idx_gt)."""
    j = np.searchsorted(t_gt, t_est)
    j = np.clip(j, 1, len(t_gt) - 1)
    left = np.abs(t_gt[j - 1] - t_est)
    right = np.abs(t_gt[j] - t_est)
    jj = np.where(left < right, j - 1, j)
    ok = np.abs(t_gt[jj] - t_est) < max_dt
    return np.where(ok)[0], jj[ok]


def align_umeyama(x, y, yaw_only=False):
    """Find R, t minimizing ||y - (R x + t)||. x, y: (N,3)."""
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    if yaw_only:
        # optimal rotation about z only
        C = yc[:, :2].T @ xc[:, :2]
        theta = np.arctan2(C[1, 0] - C[0, 1], C[0, 0] + C[1, 1])
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1.0]])
    else:
        C = yc.T @ xc
        U, _, Vt = np.linalg.svd(C)
        S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R = U @ S @ Vt
    t = my - R @ mx
    return R, t


def ate_rmse(t_est, p_est, t_gt, p_gt, yaw_only=True):
    ie, ig = associate(np.asarray(t_est), np.asarray(t_gt))
    if len(ie) < 10:
        return None
    pe, pg = np.asarray(p_est)[ie], np.asarray(p_gt)[ig]
    R, t = align_umeyama(pe, pg, yaw_only=yaw_only)
    err = pg - (pe @ R.T + t)
    return float(np.sqrt((err ** 2).sum(axis=1).mean()))


def rpe_drift(t_est, p_est, t_gt, p_gt, delta_s=1.0):
    """Median relative position error over delta_s segments (m)."""
    ie, ig = associate(np.asarray(t_est), np.asarray(t_gt))
    if len(ie) < 10:
        return None
    te = np.asarray(t_est)[ie]
    pe, pg = np.asarray(p_est)[ie], np.asarray(p_gt)[ig]
    errs = []
    k = 0
    for i in range(len(te)):
        while k < len(te) and te[k] < te[i] + delta_s:
            k += 1
        if k >= len(te):
            break
        de = np.linalg.norm(pe[k] - pe[i])
        dg = np.linalg.norm(pg[k] - pg[i])
        errs.append(abs(de - dg))
    return float(np.median(errs)) if errs else None
