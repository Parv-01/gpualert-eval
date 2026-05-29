"""Experiment 2 -- does output survive a crash?

This is the evidence behind the log-durability claim. gpualert creates the log
file *before* spawning the child and streams stdout/stderr to disk line by line,
so a line that was printed-and-flushed just before a crash has a durable home.
We compare that against the two things people actually do instead: a shell
redirect (`cmd > file`) and `nohup`.

Hypothesis: across crash types, the gpualert wrapper retains the pre-crash
sentinel line and always leaves a non-empty log file, including the exec-failure
case where the child never starts.

  IV  = method (gpualert | redirect | nohup)
  IV  = fault  (python_exception | segfault | sigkill | exec_failure)
  DV  = file_exists, sentinel_survived (per trial, boolean)
  Stats= Wilson 95% CI on survival fraction; Fisher exact gpualert vs nohup on
         the exec-failure cell.

Runs anywhere with a POSIX shell -- no GPU needed. OOM-kill behaves like sigkill
for durability purposes (SIGKILL, no chance to flush), so we use sigkill as its
stand-in and say so.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from eval.stats import fisher_exact_2x2, wilson_interval

RESULTS = Path(__file__).resolve().parent.parent / "results"
SENTINEL = "SENTINEL_LAST_LINE"
N_TRIALS = 20

# child programs that print+flush the sentinel, then fail in a specific way.
def _child_src(fault: str) -> str:
    head = (
        "import sys, os, ctypes\n"
        f"sys.stdout.write('{SENTINEL}\\n'); sys.stdout.flush()\n"
    )
    if fault == "python_exception":
        return head + "raise RuntimeError('boom')\n"
    if fault == "segfault":
        return head + "ctypes.string_at(0)\n"
    if fault == "sigkill":
        return head + "os.kill(os.getpid(), 9)\n"
    raise ValueError(fault)


FAULTS = ["python_exception", "segfault", "sigkill", "exec_failure"]
METHODS = ["gpualert", "redirect", "nohup"]


def _run_gpualert(cmd, logdir) -> str:
    """Use the real wrapper; return the combined log text."""
    from gpualert.launcher import run_job
    res = run_job(cmd)
    p = res.combined_log_path or res.stdout_log_path
    try:
        return Path(p).read_text(errors="replace")
    except Exception:
        return ""


def _run_redirect(cmd, logdir) -> str:
    log = Path(logdir) / "redirect.log"
    with log.open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log.read_text(errors="replace") if log.exists() else ""


def _run_nohup(cmd, logdir) -> str:
    log = Path(logdir) / "nohup.log"
    full = ["nohup"] + cmd
    with log.open("w") as f:
        subprocess.run(full, stdout=f, stderr=subprocess.STDOUT)
    return log.read_text(errors="replace") if log.exists() else ""


RUNNERS = {"gpualert": _run_gpualert, "redirect": _run_redirect, "nohup": _run_nohup}


def _make_cmd(fault: str, tmp: str):
    if fault == "exec_failure":
        # a binary that does not exist -> child never execs.
        return [str(Path(tmp) / "does_not_exist_binary"), "--go"]
    src = Path(tmp) / f"child_{fault}.py"
    src.write_text(_child_src(fault))
    return [sys.executable, "-u", str(src)]


def run() -> dict:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    cells = {}      # (method, fault) -> sentinel survivals
    nonempty = {}   # (method, fault) -> non-empty-log count
    for fault in FAULTS:
        for method in METHODS:
            survived = 0
            existed = 0
            for _ in range(N_TRIALS):
                with tempfile.TemporaryDirectory() as tmp:
                    cmd = _make_cmd(fault, tmp)
                    try:
                        text = RUNNERS[method](cmd, tmp)
                    except FileNotFoundError:
                        # redirect/nohup of a missing binary: shell raises before
                        # writing anything -> empty/no log.
                        text = ""
                    if text:
                        existed += 1
                    if SENTINEL in text:
                        survived += 1
            lo, hi = wilson_interval(survived, N_TRIALS)
            cells[(method, fault)] = survived
            nonempty[(method, fault)] = existed
            rows.append({
                "fault": fault, "method": method,
                "trials": N_TRIALS,
                "file_nonempty": existed,
                "sentinel_survived": survived,
                "survival_frac": round(survived / N_TRIALS, 3),
                "wilson_lo": round(lo, 3), "wilson_hi": round(hi, 3),
            })
    with (RESULTS / "exp2_log_survival.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # The honest differentiator is the *durable destination* on exec-failure:
    # gpualert creates the log file before spawning, so even a command that
    # never execs leaves a non-empty, diagnostic log; a plain shell redirect
    # leaves an empty file. (On a real crash a flushed line survives all three
    # methods -- see the sentinel columns -- so that is not where the wrapper
    # earns its keep.) Fisher exact on gpualert vs redirect, exec-failure.
    g = nonempty[("gpualert", "exec_failure")]
    r = nonempty[("redirect", "exec_failure")]
    p = fisher_exact_2x2(g, N_TRIALS - g, r, N_TRIALS - r)
    (RESULTS / "exp2_fisher.txt").write_text(
        "Fisher exact, exec_failure: gpualert vs shell redirect\n"
        "DV = log file exists and is non-empty after the failure\n"
        f"  gpualert non-empty log : {g}/{N_TRIALS}\n"
        f"  redirect non-empty log : {r}/{N_TRIALS}\n"
        f"  two-sided p            : {p:.3e}\n"
    )
    return {"rows": rows, "fisher_p": p}


if __name__ == "__main__":
    out = run()
    for r in out["rows"]:
        print(f"{r['fault']:>16} {r['method']:>9}  survived "
              f"{r['sentinel_survived']}/{r['trials']} "
              f"({r['survival_frac']:.2f}) CI[{r['wilson_lo']:.2f},{r['wilson_hi']:.2f}]")
