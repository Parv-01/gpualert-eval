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
at the end so the output is a complete, eval-ready corpus/labels.jsonl. The
manifest is rewritten after every mode, so a run you interrupt still leaves a
usable corpus of whatever finished.

    python inject/capture.py --repeat 30
    python inject/capture.py --repeat 30 --modes cuda_oom nccl segfault
    python eval/run_all.py        # then score it exactly like the synthetic run

Notes on the awkward modes (these need nothing more than the right context, no
root):
  * GPU modes (cuda_oom, cuda_runtime, device_mismatch, nccl) need torch with
    CUDA importable in the *active* environment. Without it the injector exits 3
    and the mode is skipped. Check with:
        python -c "import torch; print(torch.cuda.is_available())"
  * nccl also needs torchrun (ships with torch) and 2 ranks.
  * oom_killer needs to run inside a memory-capped cgroup so the kernel OOM
    killer fires. The clean unprivileged way is a Slurm job with a tight --mem
    (see inject/capture.sbatch). On a login node it's skipped -- we never call
    `systemd-run` interactively, so nothing prompts for a password.
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

# mode -> how to launch it, relative to inject/. oom_killer is resolved at run
# time (see _oom_cmd) because how you trigger it depends on the cgroup you're
# in. bare_ratio mixes in the no-traceback variant for the two modes where it
# matters, matching the synth corpus so the two are comparable.
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
    "oom_killer":      {"cmd": None, "gpu": False, "resolve": "oom"},
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
    captured stream -- exactly what ends up in a real job log.

    stdin is /dev/null so nothing can block on an interactive prompt, and exit
    codes are normalised from bash's 128+N signal convention back to the signed
    form (-11 = SIGSEGV) the rest of the corpus uses.
    """
    inner = shlex.join(cmd) + "; ec=$?; exit $ec"
    try:
        proc = subprocess.run(["bash", "-c", inner], cwd=str(HERE),
                              capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
        text = (proc.stdout or "") + (proc.stderr or "")
        code = proc.returncode
        if code and code > 128:
            code = -(code - 128)
        return text, code
    except subprocess.TimeoutExpired:
        return "[capture] timed out\n", -15


def _oom_cmd():
    """How to trigger the OOM-killer here, or None if we shouldn't try.

    Inside a Slurm job the step cgroup already enforces --mem, so just running
    the allocator gets it killed by the kernel. On a bare login node we don't
    have a memory cap we can rely on without root, so we skip and point at the
    sbatch path rather than hang on a privilege prompt.
    """
    if os.environ.get("SLURM_JOB_ID"):
        return [PY, "oom_killer.py"]
    return None


def kernel_oom_line() -> str | None:
    """Most recent kernel OOM-kill line from the journal/dmesg, if readable."""
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


def _was_killed(text: str, code: int) -> bool:
    return code == -9 or "Killed" in text or "Out of memory" in text or "oom-kill" in text


def write_manifest(rows: list[dict]) -> None:
    with MANIFEST.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def capture(repeat: int, modes: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gpu, driver = gpu_info()
    print(f"[capture] node GPU={gpu} driver={driver}")
    rows: list[dict] = []
    wild = load_wild()
    skipped = {}
    # keep rows from earlier passes for modes we're NOT capturing now, so
    # capturing modes across several runs (login node, then inside a Slurm job,
    # then after installing torch) accumulates instead of clobbering the manifest.
    existing: list[dict] = []
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("source") == "injected" and r.get("mode") not in modes:
                existing.append(r)
    for mode in modes:
        spec = MODES[mode]

        if spec.get("resolve") == "oom":
            resolved = _oom_cmd()
            if resolved is None:
                skipped[mode] = ("needs a memory-capped cgroup; run it inside an "
                                 "sbatch job with a tight --mem (see "
                                 "inject/capture.sbatch). Skipped on this node.")
                print(f"[capture] SKIP {mode}: {skipped[mode]}")
                continue
            spec = dict(spec, cmd=resolved)

        need = spec.get("needs")
        if need and not shutil.which(need):
            skipped[mode] = f"missing launcher '{need}'"
            print(f"[capture] SKIP {mode}: {skipped[mode]}")
            continue

        timeout = 25 if mode == "oom_killer" else 120
        got = 0
        for i in range(1, repeat + 1):
            cmd = list(spec["cmd"])
            if spec.get("bare_ratio") and (i / repeat) <= spec["bare_ratio"]:
                cmd.append(spec["bare_flag"])
            text, code = run_one(cmd, timeout=timeout)

            if spec.get("gpu") and code == NO_GPU_EXIT:
                skipped[mode] = ("no CUDA visible -- needs torch+CUDA in this env "
                                 "(python -c 'import torch; print(torch.cuda.is_available())')")
                break

            if mode == "oom_killer":
                kline = kernel_oom_line()
                if kline:
                    text = text + kline + "\n"
                if i == 1 and not _was_killed(text, code):
                    skipped[mode] = ("ran but was not OOM-killed -- the cgroup "
                                     "didn't enforce a memory cap. Use sbatch with "
                                     "a tight --mem (see inject/capture.sbatch).")
                    print(f"[capture] SKIP {mode}: {skipped[mode]}")
                    break

            sid = f"{mode}-injected-{i:04d}"
            (OUT / f"{sid}.log").write_text(text)
            rows.append({
                "id": sid, "mode": mode, "source": "injected",
                "log_path": f"corpus/injected/{sid}.log",
                "gpu": gpu, "driver": driver, "true_exit_code": code,
            })
            got += 1
        print(f"[capture] {mode}: {got} samples")
        write_manifest(existing + rows + wild)   # checkpoint after every mode

    write_manifest(existing + rows + wild)
    n_inj = len(existing) + len(rows)
    print(f"\n[capture] wrote {n_inj + len(wild)} samples ({n_inj} injected, "
          f"{len(wild)} wild) -> {MANIFEST}")
    if skipped:
        print("[capture] skipped modes:")
        for m, why in skipped.items():
            print(f"    {m}: {why}")
        print("[capture] (the corpus is still usable; eval scores whatever modes "
              "you captured.)")


def load_wild() -> list[dict]:
    rows = []
    meta = WILD / "wild_labels.tsv"
    if not meta.exists():
        return rows
    for line in meta.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sid, mode, code, gpu, driver = line.split("\t")
        rows.append({
            "id": sid, "mode": mode, "source": "wild",
            "log_path": f"corpus/wild/{sid}.log",
            "gpu": gpu, "driver": driver, "true_exit_code": int(code),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=30, help="samples per class")
    ap.add_argument("--modes", nargs="*", default=list(MODES),
                    help="subset of modes to capture (default: all 15)")
    args = ap.parse_args()
    bad = [m for m in args.modes if m not in MODES]
    if bad:
        ap.error(f"unknown modes: {bad}. choose from {list(MODES)}")
    capture(args.repeat, args.modes)


if __name__ == "__main__":
    main()
