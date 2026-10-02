"""Pose estimators fed by wheel odometry and floor-code readings."""
from __future__ import annotations

import numpy as np

from .robot import wrap

CHI2_3_999 = 16.27


def compose(pose, ds, dth):
    x, y, th = pose
    return np.array([x + ds * np.cos(th + dth / 2), y + ds * np.sin(th + dth / 2), wrap(th + dth)])


class OdometryOnly:
    name = "odometry only"

    def __init__(self, pose0):
        self.pose = np.array(pose0, float)
        self.accepted = self.rejected = 0

    def predict(self, ds, dth):
        self.pose = compose(self.pose, ds, dth)

    def update(self, z):
        return False


class ResetOnCode(OdometryOnly):
    """Overwrite the pose with every code reading. Common, simple, trusting."""
    name = "reset on code"

    def update(self, z):
        self.pose = np.array(z, float)
        self.accepted += 1
        return True


class EKF(OdometryOnly):
    """Pose EKF; odometry as the motion model, a code reading as a full-pose measurement.

    gate=True rejects readings whose Mahalanobis distance exceeds chi2(3) 99.9 %.
    recover=k additionally accepts a reading after k consecutive rejections
    that agree with each other, so a filter that has become overconfident
    cannot lock itself out of the codes for ever.
    """

    def __init__(self, pose0, gate=False, recover=0, k_ds=0.02, k_dth=0.02, k_drift=0.01,
                 r_xy=0.005, r_th=np.radians(1.0)):
        super().__init__(pose0)
        self.P = np.diag([1e-4, 1e-4, 1e-4])
        self.gate, self.recover = gate, recover
        self.k_ds, self.k_dth, self.k_drift = k_ds, k_dth, k_drift
        self.R = np.diag([r_xy ** 2, r_xy ** 2, r_th ** 2])
        self.pending = []
        self.name = "EKF" + (" + gate" if gate else "") + (" + recovery" if recover else "")

    def predict(self, ds, dth):
        th = self.pose[2] + dth / 2
        F = np.array([[1, 0, -ds * np.sin(th)], [0, 1, ds * np.cos(th)], [0, 0, 1]])
        G = np.array([[np.cos(th), -ds * np.sin(th) / 2], [np.sin(th), ds * np.cos(th) / 2], [0, 1]])
        sd = self.k_ds * abs(ds) + 1e-5
        st = self.k_dth * abs(dth) + self.k_drift * abs(ds) + 1e-5
        self.pose = compose(self.pose, ds, dth)
        self.P = F @ self.P @ F.T + G @ np.diag([sd ** 2, st ** 2]) @ G.T

    def nis(self, z):
        r = np.array(z) - self.pose
        r[2] = wrap(r[2])
        S = self.P + self.R
        return float(r @ np.linalg.solve(S, r)), r, S

    def update(self, z):
        d2, r, S = self.nis(z)
        if self.gate and d2 > CHI2_3_999:
            self.rejected += 1
            if self.recover:
                if self.pending and np.linalg.norm(np.array(z[:2]) - self.pending[-1][:2]) > 0.5:
                    self.pending = []          # readings disagree with each other: start over
                self.pending.append(np.array(z))
                if len(self.pending) >= self.recover:
                    self.pose = np.array(z, float)
                    self.P = self.R * 4
                    self.pending = []
                    self.accepted += 1
                    return True
            return False
        self.pending = []
        K = self.P @ np.linalg.inv(S)
        self.pose = self.pose + K @ r
        self.pose[2] = wrap(self.pose[2])
        self.P = (np.eye(3) - K) @ self.P
        self.accepted += 1
        return True


class BiasEKF(EKF):
    """EKF with a fourth state: heading drift per metre travelled.

    A wheel-radius mismatch makes every straight line a gentle arc. That is
    not noise, it is a bias, and a filter that models it as noise becomes
    overconfident and gates out the very codes that would correct it. Here
    the bias is estimated from the code readings and removed from odometry.
    """

    def __init__(self, pose0, gate=True, recover=5, bias_sigma0=0.05, bias_walk=1e-4, **kw):
        super().__init__(pose0, gate=gate, recover=recover, **kw)
        self.bias = 0.0
        P = np.zeros((4, 4))
        P[:3, :3] = self.P
        P[3, 3] = bias_sigma0 ** 2
        self.P4, self.bias_walk = P, bias_walk
        self.name = "EKF + gate + bias state"

    @property
    def P(self):
        return self.P4[:3, :3] if hasattr(self, "P4") else self._P

    @P.setter
    def P(self, v):
        if hasattr(self, "P4"):
            self.P4[:3, :3] = v
        else:
            self._P = v

    def predict(self, ds, dth):
        dth_c = dth + self.bias * ds
        th = self.pose[2] + dth_c / 2
        F = np.eye(4)
        F[0, 2], F[1, 2] = -ds * np.sin(th), ds * np.cos(th)
        F[0, 3], F[1, 3] = -ds * np.sin(th) * ds / 2, ds * np.cos(th) * ds / 2
        F[2, 3] = ds
        G = np.zeros((4, 2))
        G[:3] = [[np.cos(th), -ds * np.sin(th) / 2], [np.sin(th), ds * np.cos(th) / 2], [0, 1]]
        sd = self.k_ds * abs(ds) + 1e-5
        st = self.k_dth * abs(dth) + self.k_drift * abs(ds) + 1e-5
        Q = G @ np.diag([sd ** 2, st ** 2]) @ G.T
        Q[3, 3] = (self.bias_walk * abs(ds)) ** 2
        self.pose = compose(self.pose, ds, dth_c)
        self.P4 = F @ self.P4 @ F.T + Q

    def update(self, z):
        H = np.eye(3, 4)
        r = np.array(z) - self.pose
        r[2] = wrap(r[2])
        S = H @ self.P4 @ H.T + self.R
        d2 = float(r @ np.linalg.solve(S, r))
        if self.gate and d2 > CHI2_3_999:
            self.rejected += 1
            if self.recover:
                if self.pending and np.linalg.norm(np.array(z[:2]) - self.pending[-1][:2]) > 0.5:
                    self.pending = []
                self.pending.append(np.array(z))
                if len(self.pending) >= self.recover:
                    self.pose = np.array(z, float)
                    self.P4[:3, :] = 0
                    self.P4[:, :3] = 0
                    self.P4[:3, :3] = self.R * 4
                    self.pending = []
                    self.accepted += 1
                    return True
            return False
        self.pending = []
        K = self.P4 @ H.T @ np.linalg.inv(S)
        dx = K @ r
        self.pose = self.pose + dx[:3]
        self.pose[2] = wrap(self.pose[2])
        self.bias += dx[3]
        self.P4 = (np.eye(4) - K @ H) @ self.P4
        self.accepted += 1
        return True


LOCALIZERS = {
    "odometry only": lambda p: OdometryOnly(p),
    "reset on code": lambda p: ResetOnCode(p),
    "EKF": lambda p: EKF(p),
    "EKF + gate": lambda p: EKF(p, gate=True),
    "EKF + gate + recovery": lambda p: EKF(p, gate=True, recover=5),
    "EKF + gate + bias state": lambda p: BiasEKF(p),
}
