"""Experiment 3 -- wrapper overhead.

The wrapper does a fixed amount of work per job (make a log dir, spawn, two
reader threads, write a footer). That should be O(1) in the job, i.e. a small
constant the job length swamps. We measure wall-clock for the same command run
bare vs through gpualert and report the absolute difference.

  IV  = mode (bare | wrapped)
  IV  = workload (noop `sleep 0`-ish, short python)
  DV  = wall-clock seconds
  Stats= mean +/- std, median, Welch's t on the difference. Warm-ups discarded.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from eval.stats import welch_t

RESULTS = Path(__file__).resolve().parent.parent / "results"
N = 30
WARMUP = 3

WORKLOADS = {
    "noop": [sys.executable, "-c", "pass"],
    "short_py": [sys.executable, "-c", "x=sum(range(200000))"],
}


def _time_bare(cmd) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0


def _time_wrapped(cmd) -> float:
    from gpualert.launcher import run_job
    t0 = time.perf_counter()
    run_job(cmd)
    return time.perf_counter() - t0


def run() -> dict:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for wname, cmd in WORKLOADS.items():
        bare, wrapped = [], []
        for i in range(N + WARMUP):
            b = _time_bare(cmd)
            w = _time_wrapped(cmd)
            if i >= WARMUP:
                bare.append(b)
                wrapped.append(w)
        bare = np.array(bare)
        wrapped = np.array(wrapped)
        wt = welch_t(wrapped, bare)
        rows.append({
            "workload": wname, "trials": N,
            "bare_mean_ms": round(bare.mean() * 1e3, 2),
            "bare_median_ms": round(float(np.median(bare)) * 1e3, 2),
            "wrapped_mean_ms": round(wrapped.mean() * 1e3, 2),
            "wrapped_median_ms": round(float(np.median(wrapped)) * 1e3, 2),
            "overhead_mean_ms": round((wrapped.mean() - bare.mean()) * 1e3, 2),
            "welch_t": round(wt["t"], 3),
            "welch_p": round(wt["p_value"], 5),
        })
    with (RESULTS / "exp3_overhead.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {"rows": rows}


if __name__ == "__main__":
    for r in run()["rows"]:
        print(f"{r['workload']:>10}  bare={r['bare_mean_ms']}ms "
              f"wrapped={r['wrapped_mean_ms']}ms "
              f"overhead={r['overhead_mean_ms']}ms (p={r['welch_p']})")
