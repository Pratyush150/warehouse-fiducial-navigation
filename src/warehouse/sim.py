"""Pick missions: plan with A*, drive corner to corner on the estimate, align, pick."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .layout import Layout, corners, make_layout
from .localize import LOCALIZERS
from .robot import CodeReader, DriveModel, wrap

ROBOT_RADIUS = 0.30


@dataclass
class Controller:
    """Kiva-style: rotate in place to face the next corner, then drive straight
    with heading and cross-track correction. Everything runs on the estimate."""
    v_max: float = 1.0
    w_max: float = 1.5
    k_th: float = 3.0
    k_ct: float = 2.0
    align_tol: float = np.radians(5)
    arrive_tol: float = 0.03

    def command(self, est, a, b, final=False):
        """Drive along segment a -> b. Returns (v, w, arrived).

        Intermediate corners are passed within a few centimetres; the final
        stop is held to 3 mm because it is the pick position."""
        x, y, th = est
        seg = np.array(b, float) - a
        L = np.linalg.norm(seg)
        u = seg / L if L > 1e-9 else np.array([np.cos(th), np.sin(th)])
        to_b = np.array(b, float) - [x, y]
        along = float(to_b @ u)
        if along < (0.003 if final else self.arrive_tol):
            return 0.0, 0.0, True
        heading = np.arctan2(u[1], u[0])
        d = np.array([x, y]) - a
        cross = float(u[0] * d[1] - u[1] * d[0])
        e = wrap(heading - th)
        if abs(e) > self.align_tol and np.linalg.norm(to_b) > 0.1:
            return 0.0, float(np.clip(self.k_th * e, -self.w_max, self.w_max)), False
        v = min(self.v_max, 1.5 * along + (0.02 if final else 0.05))
        w = self.k_th * e - self.k_ct * cross
        return float(v), float(np.clip(w, -self.w_max, self.w_max)), False

    def rotate_to(self, est, yaw):
        e = wrap(yaw - est[2])
        if abs(e) < np.radians(0.5):
            return 0.0, 0.0, True
        return 0.0, float(np.clip(self.k_th * e, -self.w_max, self.w_max)), False


@dataclass
class Mission:
    localizer: str = "EKF + gate + recovery"
    marker_spacing: int = 1
    worn: float = 0.0
    reader: CodeReader = field(default_factory=CodeReader)
    drive: DriveModel = field(default_factory=DriveModel)
    n_tasks: int = 15
    dt: float = 0.05
    task_timeout: float = 90.0
    pick_tol_xy: float = 0.03
    pick_tol_th: float = np.radians(2.0)


def run_mission(cfg: Mission, seed: int, layout: Layout | None = None, record=False):
    rng = np.random.default_rng(seed)
    lay = layout or make_layout(cfg.marker_spacing, worn=cfg.worn, seed=seed + 7)
    stations = lay.stations()
    order = rng.choice(len(stations), cfg.n_tasks, replace=False)
    truth = np.array([0.0, 0.0, 0.0])
    loc = LOCALIZERS[cfg.localizer](truth.copy())
    ctl = Controller()
    t, picks, mis_picks, outcome = 0.0, 0, 0, "done"
    errs, misreads_used, log, pick_errs = [], 0, [], []
    for ti in order:
        goal, face = stations[ti]
        start = (int(round(loc.pose[0])), int(round(loc.pose[1])))
        path = lay.astar(start, goal) if lay.free(start) else None
        if path is None:
            outcome = "no path"
            break
        wps = [np.array(c, float) for c in corners(path)]
        a = loc.pose[:2].copy()
        k, phase, t0 = 0, "drive", t
        while True:
            if t - t0 > cfg.task_timeout:
                outcome = "timeout"
                break
            if phase == "drive":
                if k >= len(wps):
                    phase = "align"
                    continue
                v, w, arrived = ctl.command(loc.pose, a, wps[k], final=k == len(wps) - 1)
                if arrived:
                    a, k = wps[k], k + 1
                    continue
            else:
                v, w, done = ctl.rotate_to(loc.pose, face)
                if done:
                    exy = np.linalg.norm(truth[:2] - goal)
                    eth = abs(wrap(truth[2] - face))
                    ok = exy < cfg.pick_tol_xy and eth < cfg.pick_tol_th
                    pick_errs.append((float(exy), float(eth)))
                    picks += ok
                    mis_picks += not ok
                    break
            # a localiser that has estimated the curvature bias also steers against it
            w = w - getattr(loc, "bias", 0.0) * v
            truth, (ds, dth) = cfg.drive.step(truth, None, v, w, cfg.dt, rng)
            loc.predict(ds, dth)
            r = cfg.reader.read(lay, truth, rng)
            if r is not None:
                z, misread = r
                if loc.update(z) and misread:
                    misreads_used += 1
            t += cfg.dt
            err = np.linalg.norm(loc.pose[:2] - truth[:2])
            errs.append(err)
            if record:
                log.append((t, truth.copy(), loc.pose.copy()))
            if lay.clearance(truth[:2]) < ROBOT_RADIUS:
                outcome = "collision"
                break
        if outcome != "done":
            break
    errs = np.array(errs) if errs else np.zeros(1)
    out = {"outcome": outcome, "picks": int(picks), "mis_picks": int(mis_picks), "time": t,
           "err_p50": float(np.median(errs)), "err_max": float(errs.max()),
           "misreads_accepted": misreads_used, "accepted": loc.accepted, "rejected": loc.rejected,
           "pick_errs": pick_errs}
    if record:
        out["log"] = log
    return out
