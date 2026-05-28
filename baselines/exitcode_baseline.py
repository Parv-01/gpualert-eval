"""Baseline 2: exit code only -- what Slurm `--mail-type=FAIL` actually knows.

A signal tells you the process was killed and roughly how (SIGSEGV, SIGKILL),
but a plain non-zero exit says nothing about the cause, and a clean exit on a
NaN-diverged run looks like success. This is the floor: it can only ever be
right on segfault and OOM-kill, and it is blind to silent failures.
"""

from __future__ import annotations

NAME = "exitcode"


def classify(sample: dict) -> str:
    code = sample.get("true_exit_code", 0)
    if code == -11:        # SIGSEGV
        return "segfault"
    if code == -9:         # SIGKILL -- most often the cgroup OOM-killer here
        return "oom_killer"
    # every other case (1, 127, -6, 0, ...) is indistinguishable from the
    # exit code alone.
    return "generic"
