"""Regenerate docs/*.png. The sweep figure needs results/benchmark.json."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from warehouse import LOCALIZERS, DriveModel, Mission, make_layout, run_mission  # noqa: E402
from warehouse.layout import SHELF_HALF  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import SWEEPS  # noqa: E402

DOCS = ROOT / "docs"
INK, MUTED = "#1f2328", "#8c959f"
C = {"odometry only": "#d0d7de", "reset on code": "#bf8700", "EKF": "#0969da", "EKF + gate": "#cf222e",
     "EKF + gate + recovery": "#8250df", "EKF + gate + bias state": "#1a7f37"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def draw_floor(ax, lay):
    for i in range(lay.nx):
        for j in range(lay.ny):
            if lay.shelf[i, j]:
                ax.add_patch(Rectangle((i - SHELF_HALF, j - SHELF_HALF), 2 * SHELF_HALF, 2 * SHELF_HALF,
                                       fc="#d0d7de", ec="none"))
    ax.plot(lay.markers[:, 0], lay.markers[:, 1], "s", ms=1.5, c=MUTED)
    ax.set_xlim(-0.6, lay.nx - 0.4)
    ax.set_ylim(-0.6, lay.ny - 0.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def lockout_map(seed=0, err=0.005):
    lay = make_layout()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, name in zip(axs, ["EKF + gate", "EKF + gate + recovery", "EKF + gate + bias state"]):
        r = run_mission(Mission(name, drive=DriveModel(radius_err=err)), seed, record=True)
        draw_floor(ax, lay)
        P = np.array([p for _, p, _ in r["log"]])
        ax.plot(P[:, 0], P[:, 1], c=C[name], lw=1)
        if r["outcome"] == "collision":
            ax.plot(*P[-1, :2], "x", c=INK, ms=9, mew=2)
        ax.set_title(f"{name}: {r['outcome']}, {r['picks']}/15 picks in tolerance", loc="left",
                     fontsize=9, color=INK)
    fig.suptitle(f"Same mission, wheel radii mismatched by {err:.1%}. True paths; x = collision.",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "lockout.png", dpi=150)


def sweeps(rows):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, (e, (label, xs, _)) in zip(axs, SWEEPS.items()):
        for n in LOCALIZERS:
            rate = [np.mean([r["outcome"] == "done" and r["picks"] == 15 for r in rows
                             if r["exp"] == e and r["x"] == x and r["loc"] == n]) for x in xs]
            ax.plot(range(len(xs)), rate, "o-", c=C[n], lw=1.6, ms=4, label=n)
        ax.set_xticks(range(len(xs)), [str(x) for x in xs])
        ax.set_xlabel(label)
        ax.set_ylim(-0.04, 1.04)
    axs[0].set_ylabel("missions with all 15 picks in tolerance")
    axs[2].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(DOCS / "sweeps.png", dpi=150)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    sweeps(json.load(open(ROOT / "results" / "benchmark.json")))
    lockout_map()
