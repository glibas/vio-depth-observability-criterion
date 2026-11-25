"""KLT feature tracker with grid-distributed detection and epipolar RANSAC.

Maintains tracks as {feature_id: {frame_id: (u_norm, v_norm)}} where
(u_norm, v_norm) are undistorted normalized image coordinates.
"""
import numpy as np
import cv2


class Track:
    __slots__ = ("obs", "px")

    def __init__(self):
        self.obs = {}   # frame_id -> (u_norm, v_norm)
        self.px = None  # last raw pixel position (for KLT)


class FeatureTracker:
    def __init__(self, cfg, K, dist, fisheye=False):
        self.cfg = cfg
        self.K = K
        self.dist = dist
        self.fisheye = fisheye
        self.prev_img = None
        self.tracks = {}          # id -> Track (active)
        self.next_id = 0
        self.frame_stats = {}

    def _undistort(self, pts_px):
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        if self.fisheye:
            out = cv2.fisheye.undistortPoints(pts, self.K,
                                              self.dist.reshape(4, 1))
        else:
            out = cv2.undistortPoints(pts, self.K, self.dist)
        return out.reshape(-1, 2)

    def _detect(self, img, existing_px, n_needed):
        cfg = self.cfg
        h, w = img.shape
        mask = np.full((h, w), 255, np.uint8)
        r = int(cfg.min_feat_dist)
        for (x, y) in existing_px:
            cv2.circle(mask, (int(x), int(y)), r, 0, -1)
        gh, gw = h // cfg.grid_rows, w // cfg.grid_cols
        counts = np.zeros((cfg.grid_rows, cfg.grid_cols), int)
        for (x, y) in existing_px:
            gr = min(int(y) // gh, cfg.grid_rows - 1)
            gc = min(int(x) // gw, cfg.grid_cols - 1)
            counts[gr, gc] += 1
        fast = cv2.FastFeatureDetector_create(cfg.fast_threshold)
        kps = sorted(fast.detect(img, mask), key=lambda k: -k.response)
        new_pts = []
        for kp in kps:
            if len(new_pts) >= n_needed:
                break
            x, y = kp.pt
            gr = min(int(y) // gh, cfg.grid_rows - 1)
            gc = min(int(x) // gw, cfg.grid_cols - 1)
            if counts[gr, gc] >= cfg.max_per_grid:
                continue
            if mask[int(y), int(x)] == 0:
                continue
            counts[gr, gc] += 1
            cv2.circle(mask, (int(x), int(y)), r, 0, -1)
            new_pts.append((x, y))
        return new_pts

    def process(self, frame_id, img_raw):
        cfg = self.cfg
        fb_thresh = cfg.klt_fb_err
        rs_thresh = cfg.ransac_thresh_px
        img = cv2.equalizeHist(img_raw) if cfg.histogram_equalize else img_raw

        n_prev = len(self.tracks)
        n_survived = 0
        if self.prev_img is not None and self.tracks:
            ids = list(self.tracks.keys())
            p0 = np.array([self.tracks[i].px for i in ids], np.float32).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_img, img, p0, None,
                winSize=(cfg.klt_win, cfg.klt_win), maxLevel=cfg.klt_levels - 1,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            # backward check
            p0r, st_b, _ = cv2.calcOpticalFlowPyrLK(
                img, self.prev_img, p1, None,
                winSize=(cfg.klt_win, cfg.klt_win), maxLevel=cfg.klt_levels - 1,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            fb_err = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
            both = (st.ravel() == 1) & (st_b.ravel() == 1)
            h, w = img.shape
            good = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < fb_thresh)
            pts1 = p1.reshape(-1, 2)
            inb = ((pts1[:, 0] >= 1) & (pts1[:, 0] < w - 1)
                   & (pts1[:, 1] >= 1) & (pts1[:, 1] < h - 1))
            good &= inb

            # epipolar RANSAC on undistorted normalized coordinates
            if good.sum() >= 8:
                g_idx = np.where(good)[0]
                n0 = self._undistort(p0.reshape(-1, 2)[g_idx])
                n1 = self._undistort(pts1[g_idx])
                focal = 0.5 * (self.K[0, 0] + self.K[1, 1])
                _, inliers = cv2.findFundamentalMat(
                    n0, n1, cv2.FM_RANSAC,
                    rs_thresh / focal, 0.999)
                if inliers is not None:
                    bad = g_idx[inliers.ravel() == 0]
                    good[bad] = False

            surv_px = pts1[good]
            surv_norm = self._undistort(surv_px) if good.any() else np.zeros((0, 2))
            j = 0
            for k, i in enumerate(ids):
                if good[k]:
                    tr = self.tracks[i]
                    tr.px = tuple(surv_px[j])
                    tr.obs[frame_id] = tuple(surv_norm[j])
                    j += 1
                else:
                    del self.tracks[i]
            n_survived = int(good.sum())

        # detect replacements
        existing = [t.px for t in self.tracks.values()]
        n_needed = cfg.max_features - len(existing)
        if n_needed > 0:
            new_px = self._detect(img, existing, n_needed)
            if new_px:
                new_norm = self._undistort(new_px)
                for px, nm in zip(new_px, new_norm):
                    tr = Track()
                    tr.px = tuple(px)
                    tr.obs[frame_id] = tuple(nm)
                    self.tracks[self.next_id] = tr
                    self.next_id += 1

        self.prev_img = img
        self.frame_stats = {
            "n_tracked": n_survived,
            "n_active": len(self.tracks),
            "survival": n_survived / max(n_prev, 1),
            "blur": cv2.Laplacian(img_raw, cv2.CV_64F).var(),
            "fb_med": float(np.median(fb_err[good])) if 'fb_err' in dir() and good.any() else -1.0,
            # forward-backward error before the 0.5-px cut, for the sigma_px estimate
            "fb_med_all": float(np.median(fb_err[both])) if 'both' in dir() and both.any() else -1.0,
        }
        return self.frame_stats

    def pop_features(self, ids):
        out = {}
        for i in ids:
            if i in self.tracks:
                out[i] = self.tracks.pop(i).obs
        return out

    def drop_frame_obs(self, frame_id):
        """Remove observations of a marginalized clone from active tracks."""
        for tr in self.tracks.values():
            tr.obs.pop(frame_id, None)
