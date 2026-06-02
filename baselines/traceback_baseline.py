from __future__ import annotations

import re

NAME = "traceback"

EXC_TO_MODE = {
    "OutOfMemoryError": "cuda_oom",
    "MemoryError": "ram_oom",
    "FileNotFoundError": "file_not_found",
    "PermissionError": "permission",
    "ModuleNotFoundError": "missing_module",
    "ImportError": "missing_module",
    "ZeroDivisionError": "div_zero",
    "AssertionError": "assertion",
    "RuntimeError": "runtime_error",
}

_EXC_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*Error)\b", re.MULTILINE)

def classify(sample: dict) -> str:
    text = sample["log"]
    matches = _EXC_LINE.findall(text)
    if not matches:
        return "generic"

    exc = matches[-1].split(".")[-1]
    if exc in EXC_TO_MODE:
        return EXC_TO_MODE[exc]

    return "traceback"
