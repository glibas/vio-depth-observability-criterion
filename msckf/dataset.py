"""EuRoC MAV dataset loader (ASL format)."""
import os
import re
import numpy as np


def _parse_yaml_matrix(text, key):
    m = re.search(key + r":.*?data:\s*\[(.*?)\]", text, re.S)
    vals = [float(x) for x in m.group(1).replace("\n", " ").split(",")]
    return np.array(vals).reshape(4, 4)


def _parse_yaml_list(text, key):
    m = re.search(key + r":\s*\[(.*?)\]", text)
    return [float(x) for x in m.group(1).split(",")]


class EurocDataset:
    fisheye = False

    def __init__(self, seq_dir):
        mav = os.path.join(seq_dir, "mav0")
        self.seq_dir = seq_dir

        # IMU: t[s], gyro[3], accel[3]
        imu = np.loadtxt(os.path.join(mav, "imu0", "data.csv"),
                         delimiter=",", skiprows=1)
        self.imu_t = imu[:, 0] * 1e-9
        self.imu_gyro = imu[:, 1:4]
        self.imu_acc = imu[:, 4:7]

        # cam0: timestamps + image paths
        cam_csv = np.loadtxt(os.path.join(mav, "cam0", "data.csv"),
                             delimiter=",", skiprows=1, dtype=str)
        self.cam_t = cam_csv[:, 0].astype(np.int64) * 1e-9
        self.cam_files = [os.path.join(mav, "cam0", "data", f.strip())
                          for f in cam_csv[:, 1]]

        # calibration
        with open(os.path.join(mav, "cam0", "sensor.yaml")) as f:
            cam_yaml = f.read()
        T_BC = _parse_yaml_matrix(cam_yaml, "T_BS")
        self.R_IC = T_BC[:3, :3]
        self.p_IC = T_BC[:3, 3]
        fu, fv, cu, cv = _parse_yaml_list(cam_yaml, "intrinsics")
        self.K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1.0]])
        self.dist = np.array(_parse_yaml_list(cam_yaml, "distortion_coefficients"))
        self.focal = 0.5 * (fu + fv)

        # ground truth: t[s], p[3], q_wxyz[4], v[3]
        gt = np.loadtxt(os.path.join(mav, "state_groundtruth_estimate0", "data.csv"),
                        delimiter=",", skiprows=1)
        self.gt_t = gt[:, 0] * 1e-9
        self.gt_p = gt[:, 1:4]
        self.gt_q = gt[:, 4:8]
        self.gt_v = gt[:, 8:11]


    def _imu_interp(self, t):
        i = np.searchsorted(self.imu_t, t)
        i = min(max(i, 1), len(self.imu_t) - 1)
        t0, t1 = self.imu_t[i - 1], self.imu_t[i]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        g = (1 - a) * self.imu_gyro[i - 1] + a * self.imu_gyro[i]
        ac = (1 - a) * self.imu_acc[i - 1] + a * self.imu_acc[i]
        return g, ac

    def imu_segment(self, t0, t1):
        """IMU samples covering [t0, t1] exactly, with interpolated endpoints."""
        i0 = np.searchsorted(self.imu_t, t0, side="right")
        i1 = np.searchsorted(self.imu_t, t1, side="right")
        ts, gy, ac = [t0], [], []
        g, a = self._imu_interp(t0)
        gy.append(g); ac.append(a)
        for i in range(i0, i1):
            ts.append(self.imu_t[i]); gy.append(self.imu_gyro[i]); ac.append(self.imu_acc[i])
        if ts[-1] < t1 - 1e-9:
            g, a = self._imu_interp(t1)
            ts.append(t1); gy.append(g); ac.append(a)
        return np.array(ts), np.array(gy), np.array(ac)

    def imu_between(self, t0, t1):
        """IMU sample indices with t0 < t <= t1."""
        i0 = np.searchsorted(self.imu_t, t0, side="right")
        i1 = np.searchsorted(self.imu_t, t1, side="right")
        return i0, i1


class TumViDataset(EurocDataset):
    """TUM-VI (euroc folder layout, fisheye/equidistant cameras, mocap GT)."""
    fisheye = True

    def __init__(self, seq_dir):
        mav = os.path.join(seq_dir, "mav0")
        self.seq_dir = seq_dir

        imu = np.loadtxt(os.path.join(mav, "imu0", "data.csv"),
                         delimiter=",", skiprows=1)
        self.imu_t = imu[:, 0] * 1e-9
        self.imu_gyro = imu[:, 1:4]
        self.imu_acc = imu[:, 4:7]

        cam_csv = np.loadtxt(os.path.join(mav, "cam0", "data.csv"),
                             delimiter=",", skiprows=1, dtype=str)
        self.cam_t = cam_csv[:, 0].astype(np.int64) * 1e-9
        self.cam_files = [os.path.join(mav, "cam0", "data", f.strip())
                          for f in cam_csv[:, 1]]

        with open(os.path.join(seq_dir, "dso", "camchain.yaml")) as f:
            y = f.read()
        cam0 = y.split("cam1:")[0]
        rows = re.findall(r"- \[(.*?)\]", cam0)
        T_ci = np.array([[float(v) for v in r.split(",")] for r in rows[:4]])
        R_ci, t_ci = T_ci[:3, :3], T_ci[:3, 3]
        self.R_IC = R_ci.T
        self.p_IC = -R_ci.T @ t_ci
        intr = re.search(r"intrinsics:\s*\[(.*?)\]", cam0, re.S).group(1)
        fu, fv, cu, cv = [float(v) for v in intr.replace("\n", " ").split(",")]
        self.K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1.0]])
        self.focal = 0.5 * (fu + fv)
        dist = re.search(r"distortion_coeffs:\s*\[(.*?)\]", cam0, re.S).group(1)
        self.dist = np.array([float(v)
                              for v in dist.replace("\n", " ").split(",")])

        gt = np.loadtxt(os.path.join(mav, "mocap0", "data.csv"),
                        delimiter=",", skiprows=1)
        self.gt_t = gt[:, 0] * 1e-9
        self.gt_p = gt[:, 1:4]
        self.gt_q = gt[:, 4:8]
        self.gt_v = np.zeros_like(self.gt_p)


def load_dataset(seq_dir):
    if os.path.exists(os.path.join(seq_dir, "dso", "camchain.yaml")):
        return TumViDataset(seq_dir)
    return EurocDataset(seq_dir)
