"""Differential-drive truth model, wheel odometry, and the downward code reader."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import Layout


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class DriveModel:
    """Nominal geometry is what the software believes; the true robot differs.

    radius_err: left wheel (1+e), right wheel (1-e) -> the robot curves
    scale_err : both wheels (1+e)               -> distances are off
    """
    r: float = 0.10
    b: float = 0.50
    radius_err: float = 0.0
    scale_err: float = 0.0
    enc_noise: float = 0.01       # relative noise on each wheel increment

    def step(self, pose, odom_in, v, w, dt, rng):
        """Apply (v, w) to the true robot. Returns new true pose and odometry (ds, dth)."""
        wl, wr = (v - w * self.b / 2) / self.r, (v + w * self.b / 2) / self.r
        rl = self.r * (1 + self.scale_err) * (1 + self.radius_err)
        rr = self.r * (1 + self.scale_err) * (1 - self.radius_err)
        dl, dr = wl * rl * dt, wr * rr * dt                    # true wheel travel
        ds, dth = (dl + dr) / 2, (dr - dl) / self.b
        x, y, th = pose
        new = np.array([x + ds * np.cos(th + dth / 2), y + ds * np.sin(th + dth / 2), wrap(th + dth)])
        # encoders count the true wheel rotation; odometry converts it with the nominal radius
        ol = wl * dt * self.r * (1 + rng.normal(0, self.enc_noise))
        orr = wr * dt * self.r * (1 + rng.normal(0, self.enc_noise))
        return new, ((ol + orr) / 2, (orr - ol) / self.b)


@dataclass
class CodeReader:
    """Downward camera reading floor codes (Kiva-style 2D codes at cell centres)."""
    half_fov: float = 0.15        # m, square field of view around the robot centre
    sigma_xy: float = 0.003
    sigma_th: float = np.radians(0.5)
    p_misread: float = 0.0        # decoded the wrong id: a neighbouring code
    p_miss: float = 0.02

    def read(self, layout: Layout, pose, rng):
        """Returns the robot pose implied by a code in view, or None.

        The code's pose in the robot frame is measured with noise. It is then
        composed with the map pose of the code *as decoded*. A misread
        therefore produces a pose that is exactly one marker spacing wrong, in
        a perfectly plausible place: the hardest kind of outlier.
        """
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        d = layout.markers - [x, y]
        rel = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
        inside = np.all(np.abs(rel) < self.half_fov, axis=1)
        if not inside.any() or rng.random() < self.p_miss:
            return None
        k = int(np.flatnonzero(inside)[0])
        rel_n = rel[k] + rng.normal(0, self.sigma_xy, 2)
        yaw_rel = wrap(-th + rng.normal(0, self.sigma_th))
        code = layout.markers[k]
        misread = False
        if rng.random() < self.p_misread:
            dist = np.linalg.norm(layout.markers - code, axis=1)
            nbr = np.flatnonzero((dist > 0) & (dist <= dist[dist > 0].min() + 1e-9))
            code = layout.markers[rng.choice(nbr)]
            misread = True
        th_hat = wrap(-yaw_rel)
        R = np.array([[np.cos(th_hat), -np.sin(th_hat)], [np.sin(th_hat), np.cos(th_hat)]])
        return np.r_[code - R @ rel_n, th_hat], misread
