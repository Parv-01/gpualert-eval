from __future__ import annotations

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
