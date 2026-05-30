#!/usr/bin/env python3
"""One command to set up, run, and verify the whole evaluation bench.

    python bench.py

It walks the full workflow and prints a PASS/FAIL line for each stage:

  1. environment    -- numpy, matplotlib, and the gpualert classifier import
  2. corpus         -- build corpus/labels.jsonl + corpus/injected/
  3. experiments    -- run experiments 1-5 into results/
  4. tests          -- pytest (skipped if pytest isn't installed)
  5. reproducibility -- re-run Experiment 1 and confirm the deterministic
                        outputs are byte-identical (this is what lets you trust
                        the committed numbers without re-deriving them by hand)

Exit code is 0 only if every required stage passes. Experiment 3 (timing) is
expected to vary run to run, so it's excluded from the reproducibility check.

If stage 1 fails because gpualert can't be imported, either install it
(`pip install -e ../gpualert`) or set GPUALERT_SRC to the directory holding the
`gpualert/` package.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# deterministic outputs we expect to be identical across runs
DETERMINISTIC = [
    "results/exp1_summary.csv",
    "results/exp1_per_class.csv",
    "results/exp1_confusion.csv",
    "results/exp1_by_source.csv",
]


def _env() -> dict:
    e = dict(os.environ)
    # make eval/ baselines/ corpus/ importable in child processes
    e["PYTHONPATH"] = str(ROOT) + os.pathsep + e.get("PYTHONPATH", "")
    return e


def _run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), env=_env(),
                          capture_output=True, text=True, **kw)


def _hash(paths) -> str:
    h = hashlib.sha256()
    for rel in paths:
        p = ROOT / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def stage(name: str):
    print(f"\n[{name}]")


def main() -> int:
    results = {}

    # 1. environment ---------------------------------------------------------
    stage("1/5 environment")
    probe = _run([PY, "-c",
                  "import numpy, matplotlib; "
                  "import sys; sys.path.insert(0, '.'); "
                  "from eval import gpualert_adapter as g; "
                  "print('gpualert via', g.parse_errors.__module__)"])
    if probe.returncode == 0:
        print("  numpy, matplotlib, gpualert classifier importable")
        print("  " + probe.stdout.strip())
        results["environment"] = True
    else:
        print("  FAILED to import dependencies:")
        print("  " + (probe.stderr.strip().replace("\n", "\n  ")))
        print("\n  Fix: pip install -r requirements.txt && pip install -e ../gpualert")
        print("       (or set GPUALERT_SRC to the gpualert package directory)")
        results["environment"] = False
        return _summary(results)

    # 2. corpus --------------------------------------------------------------
    stage("2/5 corpus")
    r = _run([PY, "corpus/build.py"])
    print("  " + (r.stdout.strip() or r.stderr.strip()))
    results["corpus"] = r.returncode == 0

    # 3. experiments ---------------------------------------------------------
    stage("3/5 experiments")
    r = _run([PY, "eval/run_all.py"])
    tail = "\n  ".join(r.stdout.strip().splitlines()[-3:])
    print("  " + tail)
    results["experiments"] = r.returncode == 0

    # 4. tests ---------------------------------------------------------------
    stage("4/5 tests")
    have_pytest = _run([PY, "-c", "import pytest"]).returncode == 0
    if not have_pytest:
        print("  pytest not installed -- skipping (not required). `pip install pytest` to enable.")
        results["tests"] = None
    else:
        r = _run([PY, "-m", "pytest", "-q", "tests"])
        print("  " + r.stdout.strip().splitlines()[-1])
        results["tests"] = r.returncode == 0

    # 5. reproducibility -----------------------------------------------------
    stage("5/5 reproducibility")
    before = _hash(DETERMINISTIC)
    rr = _run([PY, "eval/exp1_classifier.py"])
    after = _hash(DETERMINISTIC)
    if rr.returncode == 0 and before == after:
        print("  Experiment 1 re-ran byte-identical -- results are deterministic.")
        results["reproducibility"] = True
    else:
        print("  Re-run did not match (this should not happen for a seeded run).")
        results["reproducibility"] = False

    return _summary(results)


def _summary(results: dict) -> int:
    print("\n" + "=" * 48)
    print("  bench summary")
    print("=" * 48)
    ok = True
    for name, val in results.items():
        if val is None:
            mark = "skip"
        elif val:
            mark = "PASS"
        else:
            mark = "FAIL"
            ok = False
        print(f"  {name:<16} {mark}")
    print("=" * 48)
    print("  overall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
