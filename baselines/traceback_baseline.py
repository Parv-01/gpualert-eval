"""Baseline 3: parse the Python exception type off the last traceback line.

Stronger than grep on anything that surfaces as a Python exception, because it
reads the actual exception class. But it cannot tell CUDA/NCCL/device-mismatch
RuntimeErrors apart (all -> runtime_error), and it is blind to failures that
never raise: segfaults, OOM-kills and NaN-in-stdout produce no traceback, so it
returns generic for them.
"""

from __future__ import annotations

import re

NAME = "traceback"

# exception class -> mode. Order doesn't matter; we match the exact class name.
EXC_TO_MODE = {
    "OutOfMemoryError": "cuda_oom",      # torch.cuda.OutOfMemoryError
    "MemoryError": "ram_oom",
    "FileNotFoundError": "file_not_found",
    "PermissionError": "permission",
    "ModuleNotFoundError": "missing_module",
    "ImportError": "missing_module",
    "ZeroDivisionError": "div_zero",
    "AssertionError": "assertion",
    "RuntimeError": "runtime_error",
}

# a final "ExceptionName: message" or bare "ExceptionName" line.
_EXC_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*Error)\b", re.MULTILINE)


def classify(sample: dict) -> str:
    text = sample["log"]
    matches = _EXC_LINE.findall(text)
    if not matches:
        return "generic"
    # take the last exception name on the line (strip any module prefix).
    exc = matches[-1].split(".")[-1]
    if exc in EXC_TO_MODE:
        return EXC_TO_MODE[exc]
    # a real exception, just not one we have a dedicated bucket for.
    return "traceback"
