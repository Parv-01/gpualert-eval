#!/usr/bin/env python3
"""Capture a labelled corpus on real hardware.

This is the on-node version of the corpus: instead of replaying recorded
captures (what corpus/build.py does so the repo runs anywhere), it actually runs
each injector on the GPU/Slurm node you're sitting on and records what really
happened. The injected fault is still the label, so nothing here needs hand
annotation.

For every mode it runs the injector REPEAT times, captures the merged
stdout+stderr to corpus/injected/<id>.log, and writes a manifest row with the
real GPU name, driver version, and process exit code. The wild set is folded in
at the end so the output is a complete, eval-ready corpus/labels.jsonl.

    python inject/capture.py --repeat 30
    python inject/capture.py --repeat 30 --modes cuda_oom nccl segfault
    python eval/run_all.py        # then score it exactly like the synthetic run

GPU-bound injectors exit 3 when no CUDA device is visible; those samples are
skipped with a warning rather than recorded as bogus. `nccl` needs torchrun and
`oom_killer` needs systemd-run (or a cgroup memory cap) -- if the launcher isn't
present that mode is skipped and reported, so a partial node still gives you a
partial corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "corpus" / "injected"
WILD = ROOT / "corpus" / "wild"
MANIFEST = ROOT / "corpus" / "labels.jsonl"
PY = sys.executable

NO_GPU_EXIT = 3  # injectors' sentinel for "no CUDA device" (see inject/_common.py)

# mode -> how to launch it, relative to inject/. `bare_ratio` mixes in the
# no-traceback variant for the two modes where it matters, matching the synth
# corpus so the two are comparable.
MODES = {
    "cuda_oom":        {"cmd": [PY, "cuda_oom.py"], "gpu": True},
    "nccl":            {"cmd": ["torchrun", "--nproc_per_node=2", "nccl_timeout.py"],
                        "gpu": True, "needs": "torchrun"},
    "cuda_runtime":    {"cmd": [PY, "cuda_runtime.py"], "gpu": True},
    "ram_oom":         {"cmd": [PY, "ram_oom.py"], "gpu": False},
    "segfault":        {"cmd": [PY, "segfault.py"], "gpu": False},
    "file_not_found":  {"cmd": [PY, "file_not_found.py"], "gpu": False},
    "permission":      {"cmd": [PY, "permission_denied.py"], "gpu": False},
    "missing_module":  {"cmd": [PY, "missing_module.py"], "gpu": False},
    "div_zero":        {"cmd": [PY, "div_zero.py"], "gpu": False},
    "device_mismatch": {"cmd": [PY, "device_mismatch.py"], "gpu": True},
    "nan_loss":        {"cmd": [PY, "nan_loss.py"], "gpu": False},
    "oom_killer":      {"cmd": ["systemd-run", "--scope", "-p", "MemoryMax=500M",
                                PY, "oom_killer.py"],
                        "gpu": False, "needs": "systemd-run"},
    "traceback":       {"cmd": [PY, "traceback_generic.py"], "gpu": False},
    "assertion":       {"cmd": [PY, "assertion.py"], "gpu": False, "bare_ratio": 0.33,
                        "bare_flag": "--bare"},
    "runtime_error":   {"cmd": [PY, "runtime_error.py"], "gpu": False, "bare_ratio": 0.33,
                        "bare_flag": "--bare"},
}


def gpu_info() -> tuple[str, str]:
    """(name, driver) from nvidia-smi, or ('unknown','unknown')."""
    if not shutil.which("nvidia-smi"):
        return ("unknown", "unknown")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        first = out.stdout.strip().splitlines()[0]
        name, driver = [x.strip() for x in first.split(",")[:2]]
        return (name.replace(" ", "-"), driver)
    except Exception:
        return ("unknown", "unknown")


def run_one(cmd, timeout=120) -> tuple[str, int]:
    """Run a command under a shell and return (merged_log_text, exit_code).

    We run through `bash -c '...; exit $?'` on purpose: that's how a Slurm job
    actually runs your command, and it means the shell prints the conventional
    signal message ("Segmentation fault (core dumped)", "Killed") into the
    captured stream -- exactly what ends up in a real job log. A bare
    subprocess wouldn't capture those, because the child can't print its own
    death-by-signal notice.

    Exit codes are normalised from bash's 128+N signal convention back to the
    signed form (-11 = SIGSEGV) the rest of the corpus uses.
    """
    inner = shlex.join(cmd) + "; ec=$?; exit $ec"
    try:
        proc = subprocess.run(["bash", "-c", inner], cwd=str(HERE),
                              capture_output=True, text=True, timeout=timeout)
        text = (proc.stdout or "") + (proc.stderr or "")
        code = proc.returncode
        if code and code > 128:
            code = -(code - 128)
        return text, code
    except subprocess.TimeoutExpired:
        return "[capture] timed out\n", -15


def kernel_oom_line() -> str | None:
    """Most recent kernel OOM-kill line from the journal/dmesg, if readable.

    A cgroup OOM-kill leaves "Out of memory: Killed process ..." in the kernel
    log; that's the line the classifier keys on. The shell only sees "Killed",
    so for the oom_killer mode we append the kernel evidence when we can read it
    (needs journal/dmesg access -- run the sbatch with the right privileges).
    """
    for probe in (["journalctl", "-k", "--since", "-2 min", "--no-pager"],
                  ["dmesg"]):
        if not shutil.which(probe[0]):
            continue
        try:
            out = subprocess.run(probe, capture_output=True, text=True, timeout=10)
            for line in reversed(out.stdout.splitlines()):
                if "Out of memory: Killed process" in line or "oom-kill" in line:
                    return line.strip()
        except Exception:
            continue
    return None


def capture(repeat: int, modes: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gpu, driver = gpu_info()
    print(f"[capture] node GPU={gpu} driver={driver}")
    rows = []
    skipped = {}
    for mode in modes:
        spec = MODES[mode]
        need = spec.get("needs")
        if need and not shutil.which(need):
            skipped[mode] = f"missing launcher '{need}'"
            print(f"[capture] SKIP {mode}: {skipped[mode]}")
            continue
        got = 0
        for i in range(1, repeat + 1):
            cmd = list(spec["cmd"])
            if spec.get("bare_ratio") and (i / repeat) <= spec["bare_ratio"]:
                cmd.append(spec["bare_flag"])
            text, code = run_one(cmd)
            if spec.get("gpu") and code == NO_GPU_EXIT:
                skipped[mode] = "no CUDA device visible"
                break
            if mode == "oom_killer":
                kline = kernel_oom_line()
                if kline:
                    text = text + kline + "\n"
            sid = f"{mode}-injected-{i:04d}"
            (OUT / f"{sid}.log").write_text(text)
            rows.append({
                "id": sid, "mode": mode, "source": "injected",
                "log_path": f"corpus/injected/{sid}.log",
                "gpu": gpu, "driver": driver, "true_exit_code": code,
            })
            got += 1
        