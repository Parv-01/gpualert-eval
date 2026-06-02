#!/usr/bin/env python3

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "inject" / "reference"
OUT = ROOT / "corpus" / "injected"
WILD = ROOT / "corpus" / "wild"
MANIFEST = ROOT / "corpus" / "labels.jsonl"

PARV_BASE_SEED = 20260530

N_PER_CLASS = 32

REFERENCES = {
    "cuda_oom":        ["cuda_oom.log"],
    "nccl":            ["nccl.log"],
    "cuda_runtime":    ["cuda_runtime.log"],
    "ram_oom":         ["ram_oom.log"],
    "segfault":        ["segfault.log"],
    "file_not_found":  ["file_not_found.log"],
    "permission":      ["permission.log"],
    "missing_module":  ["missing_module.log"],
    "div_zero":        ["div_zero.log"],
    "device_mismatch": ["device_mismatch.log"],
    "nan_loss":        ["nan_loss.log"],
    "oom_killer":      ["oom_killer.log"],
    "traceback":       ["traceback.log"],
    "assertion":       ["assertion.log", "assertion.log", "assertion_bare.log"],
    "runtime_error":   ["runtime_error.log", "runtime_error.log", "runtime_error_bare.log"],
}

EXIT_CODE = {
    "cuda_oom": 1, "nccl": 1, "cuda_runtime": 1, "ram_oom": 1,
    "segfault": -11, "file_not_found": 1, "permission": 1,
    "missing_module": 1, "div_zero": 1, "device_mismatch": 1,
    "nan_loss": 0, "oom_killer": -9, "traceback": 1,
    "assertion": 1, "runtime_error": 1,
}

GPUS = [
    ("A100-SXM4-40GB", "535.104.05"),
    ("A100-SXM4-80GB", "535.129.03"),
    ("V100-SXM2-32GB", "525.85.12"),
    ("RTX-A6000",      "550.54.14"),
    ("H100-SXM5-80GB", "550.90.07"),
]

def _mutate(text: str, rng: random.Random) -> str:
    host = f"gpu{rng.randint(1, 256):03d}"
    text = re.sub(r"host=gpu\d+", f"host={host}", text)
    text = re.sub(r"gpu\d+:\d+", lambda m: m.group(0).split(':')[0] + f":{rng.randint(10000, 99999)}", text)

    text = re.sub(r"pid=\d+", f"pid={rng.randint(1000, 65000)}", text)
    text = re.sub(r"process \d+ \(python\)", f"process {rng.randint(1000, 65000)} (python)", text)

    text = re.sub(r"line (\d+)", lambda m: f"line {int(m.group(1)) + rng.randint(-7, 40)}", text)

    text = re.sub(r"Tried to allocate [\d.]+ GiB",
                  f"Tried to allocate {rng.choice([1.0, 2.0, 3.5, 8.0, 12.0]):.2f} GiB", text)
    text = re.sub(r"[\d.]+ GiB free", f"{rng.choice([0.12, 0.44, 1.01, 2.3]):.2f} GiB free", text)

    text = re.sub(r"batch size \d+ != \d+",
                  f"batch size {rng.choice([24, 30, 31, 48])} != {rng.choice([32, 64])}", text)

    text = re.sub(r"SeqNum=\d+", f"SeqNum={rng.randint(1, 400)}", text)
    return text

def _maybe_trim_traceback(text: str, rng: random.Random) -> str:
    lines = text.splitlines()
    if "Traceback (most recent call last):" not in text:
        return text
    if rng.random() < 0.4:

        frame_idx = [i for i, l in enumerate(lines) if l.strip().startswith('File "')]
        if len(frame_idx) >= 2:
            drop = frame_idx[0]
            del lines[drop:drop + 2]
    return "\n".join(lines) + "\n"

def build_injected() -> list[dict]:
    rows: list[dict] = []
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, refs in REFERENCES.items():
        for i in range(1, N_PER_CLASS + 1):
            rng = random.Random(f"{PARV_BASE_SEED}:{mode}:{i}")
            ref = refs[(i - 1) % len(refs)]
            raw = (REF / ref).read_text()
            txt = _maybe_trim_traceback(_mutate(raw, rng), rng)
            gpu, driver = rng.choice(GPUS)
            sid = f"{mode}-injected-{i:04d}"
            (OUT / f"{sid}.log").write_text(txt)
            rows.append({
                "id": sid,
                "mode": mode,
                "source": "injected",
                "log_path": f"corpus/injected/{sid}.log",
                "gpu": gpu,
                "driver": driver,
                "true_exit_code": EXIT_CODE[mode],
            })
    return rows

def load_wild() -> list[dict]:
    rows: list[dict] = []
    meta = WILD / "wild_labels.tsv"
    if not meta.exists():
        return rows
    for line in meta.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sid, mode, exit_code, gpu, driver = line.split("\t")
        rows.append({
            "id": sid,
            "mode": mode,
            "source": "wild",
            "log_path": f"corpus/wild/{sid}.log",
            "gpu": gpu,
            "driver": driver,
            "true_exit_code": int(exit_code),
        })
    return rows

def main() -> None:
    rows = build_injected()
    rows += load_wild()
    with MANIFEST.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_inj = sum(1 for r in rows if r["source"] == "injected")
    n_wild = sum(1 for r in rows if r["source"] == "wild")
    print(f"wrote {len(rows)} samples ({n_inj} injected, {n_wild} wild) -> {MANIFEST}")

if __name__ == "__main__":
    main()
