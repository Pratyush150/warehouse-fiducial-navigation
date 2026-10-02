"""Three sweeps, every localiser, identical seeded missions (15 picks each).

    python examples/benchmark.py --seeds 40 --jobs 8

E1  fraction of floor codes worn unreadable
E2  code misread rate (wrong id decoded -> pose exactly one spacing off)
E3  wheel-radius mismatch (left +e, right -e)
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from warehouse import LOCALIZERS, CodeReader, DriveModel, Mission, run_mission  # noqa: E402

SWEEPS = {
    "E1": ("worn (unreadable) codes, %", [0, 25, 50, 75], lambda x: dict(worn=x / 100)),
    "E2": ("misread rate, %", [0, 0.5, 2, 5], lambda x: dict(reader=CodeReader(p_misread=x / 100))),
    "E3": ("wheel-radius error, %", [0, 0.25, 0.5, 1, 2], lambda x: dict(drive=DriveModel(radius_err=x / 100))),
}


def job(a):
    exp, x, name, seed = a
    r = run_mission(Mission(localizer=name, **SWEEPS[exp][2](x)), seed)
    return {"exp": exp, "x": x, "loc": name, "seed": seed, **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    tasks = [(e, x, n, s) for e, (_, xs, _) in SWEEPS.items() for x in xs for n in LOCALIZERS
             for s in range(a.seeds)]
    with Pool(a.jobs) as p:
        rows = p.map(job, tasks, chunksize=4)
    Path(a.out).mkdir(exist_ok=True)
    json.dump(rows, open(Path(a.out) / "benchmark.json", "w"))
    print("cell = missions with all 15 picks in tolerance / collisions / median position error (mm)")
    for e, (label, xs, _) in SWEEPS.items():
        print(f"\n{e}  {label}")
        print(f"  {'localiser':<25}" + "".join(f"{x:>17}" for x in xs))
        for n in LOCALIZERS:
            cells = []
            for x in xs:
                rs = [r for r in rows if r["exp"] == e and r["x"] == x and r["loc"] == n]
                perfect = sum(r["outcome"] == "done" and r["picks"] == 15 for r in rs)
                coll = sum(r["outcome"] == "collision" for r in rs)
                err = 1000 * np.median([r["err_p50"] for r in rs])
                cells.append(f"{perfect:>3}/{len(rs)} c{coll:<3}{err:6.1f}")
            print(f"  {n:<25}" + "".join(f"{c:>17}" for c in cells))


if __name__ == "__main__":
    main()
