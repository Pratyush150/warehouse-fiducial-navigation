import numpy as np
import pytest

from warehouse import (EKF, CodeReader, Controller, DriveModel, Mission, OdometryOnly, corners,
                       make_layout, run_mission, wrap)
from warehouse.localize import BiasEKF

RNG = np.random.default_rng


# ---------------------------------------------------------------- layout / planning
def test_layout_has_two_deep_shelves_and_one_cell_aisles():
    lay = make_layout()
    assert lay.shelf[2:7, 1].all() and lay.shelf[2:7, 2].all()
    assert not lay.shelf[:, 3].any() and not lay.shelf[7, :].any()


def test_astar_path_is_connected_free_and_shortest_in_moves():
    lay = make_layout()
    p = lay.astar((0, 0), (10, 9))
    assert p[0] == (0, 0) and p[-1] == (10, 9)
    assert all(lay.free(c) for c in p)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(p, p[1:]))
    assert len(p) - 1 == 10 + 9                    # Manhattan distance is achievable here


def test_astar_prefers_fewer_turns_at_equal_length():
    lay = make_layout()
    assert len(corners(lay.astar((0, 0), (7, 6)))) == 3          # one turn


def test_astar_refuses_shelf_cells():
    lay = make_layout()
    assert lay.astar((0, 0), (3, 1)) is None


def test_stations_face_a_shelf():
    lay = make_layout()
    for (i, j), yaw in lay.stations():
        di, dj = int(round(np.cos(yaw))), int(round(np.sin(yaw)))
        assert lay.shelf[i + di, j + dj]


def test_clearance_in_an_aisle_and_inside_a_shelf():
    lay = make_layout()
    assert lay.clearance((7.0, 1.0)) == pytest.approx(0.55)      # aisle between two shelves
    assert lay.clearance((3.0, 1.0)) < 0


# ---------------------------------------------------------------- robot model
def test_calibrated_odometry_matches_truth_without_noise():
    dm = DriveModel(enc_noise=0)
    truth, odom = np.zeros(3), np.zeros(3)
    rng = RNG(0)
    for v, w in [(1.0, 0.0)] * 40 + [(0.0, 1.0)] * 30 + [(0.5, 0.3)] * 40:
        truth, (ds, dth) = dm.step(truth, None, v, w, 0.05, rng)
        odom = np.array([odom[0] + ds * np.cos(odom[2] + dth / 2), odom[1] + ds * np.sin(odom[2] + dth / 2),
                         wrap(odom[2] + dth)])
    assert odom == pytest.approx(truth, abs=1e-12)


def test_radius_mismatch_curves_a_commanded_straight_line():
    dm = DriveModel(radius_err=0.005, enc_noise=0)
    truth = np.zeros(3)
    for _ in range(200):                                          # 10 m at 1 m/s
        truth, _ = dm.step(truth, None, 1.0, 0.0, 0.05, RNG(0))
    assert truth[2] == pytest.approx(-0.02 * 10, rel=0.01)        # -2e/b rad per metre


def test_code_reading_recovers_pose():
    lay = make_layout()
    rd = CodeReader(sigma_xy=0, sigma_th=0, p_miss=0)
    pose = np.array([5.05, 3.0 - 0.07, 0.3])
    z, mis = rd.read(lay, pose, RNG(0))
    assert not mis and z == pytest.approx(pose, abs=1e-12)


def test_no_code_outside_the_field_of_view():
    assert CodeReader(p_miss=0).read(make_layout(), np.array([5.5, 3.5, 0.0]), RNG(0)) is None


def test_misread_is_exactly_one_spacing_off():
    lay = make_layout(2)
    rd = CodeReader(sigma_xy=0, sigma_th=0, p_miss=0, p_misread=1.0)
    z, mis = rd.read(lay, np.array([6.0, 6.0, 0.0]), RNG(1))
    assert mis and np.linalg.norm(z[:2] - [6.0, 6.0]) == pytest.approx(2.0)


# ---------------------------------------------------------------- localisation
def test_gate_rejects_a_one_metre_misread_and_plain_ekf_does_not():
    for gate, accepted in ((True, False), (False, True)):
        ekf = EKF(np.zeros(3), gate=gate)
        assert ekf.update(np.array([1.0, 0.0, 0.0])) is accepted


def test_recovery_relocalises_after_consistent_rejections():
    ekf = EKF(np.zeros(3), gate=True, recover=3)
    for _ in range(2):
        assert not ekf.update(np.array([0.4, 0.0, 0.0]))
    assert ekf.update(np.array([0.4, 0.0, 0.0]))
    assert ekf.pose == pytest.approx([0.4, 0.0, 0.0])


def test_bias_state_learns_wheel_radius_mismatch():
    lay = make_layout()
    dm, rd, rng = DriveModel(radius_err=0.005), CodeReader(), RNG(2)
    truth, loc = np.zeros(3), BiasEKF(np.zeros(3))
    ctl = Controller()
    a = np.zeros(2)
    for b in [np.array([20.0, 0.0]), np.array([20.0, 12.0]), np.array([0.0, 12.0]), np.array([0.0, 0.0])]:
        while True:
            v, w, done = ctl.command(loc.pose, a, b)
            if done:
                break
            truth, (ds, dth) = dm.step(truth, None, v, w, 0.05, rng)
            loc.predict(ds, dth)
            r = rd.read(lay, truth, rng)
            if r is not None:
                loc.update(r[0])
        a = b
    assert loc.bias == pytest.approx(-0.02, abs=0.004)


def test_odometry_only_ignores_codes():
    o = OdometryOnly(np.zeros(3))
    assert not o.update(np.array([1.0, 1.0, 1.0]))
    assert o.pose == pytest.approx([0, 0, 0])


# ---------------------------------------------------------------- missions
def test_calibrated_mission_completes_all_picks():
    r = run_mission(Mission("EKF + gate", n_tasks=5), seed=3)
    assert r["outcome"] == "done" and r["picks"] == 5


def test_bias_state_survives_miscalibration_that_locks_out_the_gated_ekf():
    bad = DriveModel(radius_err=0.005)
    assert run_mission(Mission("EKF + gate", drive=bad, n_tasks=5), seed=0)["outcome"] == "collision"
    r = run_mission(Mission("EKF + gate + bias state", drive=bad, n_tasks=5), seed=0)
    assert r["outcome"] == "done" and r["picks"] == 5
