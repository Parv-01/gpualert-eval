"""Baseline 1: a naive keyword grep, the kind a sysadmin tapes together.

Fixed, case-sensitive substring rules in a flat first-match order. No
specificity ordering, no regex alternations. This is deliberately dumber than
the gpualert classifier so the comparison is honest about what the extra
engineering buys. Notably it keys on the bare `RuntimeError` string, so it
mislabels CUDA/NCCL/device-mismatch RuntimeErrors -- but it also catches an
`AssertionError` even when it sits under a traceback header.
"""

from __future__ import annotations

# (substring, mode) -- first hit wins, evaluated top to bottom.
RULES = [
    ("CUDA out of memory", "cuda_oom"),
    ("NCCL", "nccl"),
    ("Segmentation fault", "segfault"),
    ("MemoryError", "ram_oom"),
    ("FileNotFoundError", "file_not_found"),
    ("PermissionError", "permission"),
    ("ModuleNotFoundError", "missing_module"),
    ("ZeroDivisionError", "div_zero"),
    ("AssertionError", "assertion"),
    ("nan", "nan_loss"),
    ("Killed", "oom_killer"),
    ("RuntimeError", "runtime_error"),
    ("Traceback", "traceback"),
]

NAME = "grep"


def classify(sample: dict) -> str:
    text = sample["log"]
    for needle, mode in RULES:
        if needle in text:
            return mode
    return "generic"
