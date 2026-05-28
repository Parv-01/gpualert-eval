"""Canonical failure classes for the evaluation.

The 15 keys here are a one-to-one mapping onto the gpualert classifier's
own labels (gpualert/parse_errors.py, ERROR_PATTERNS, in priority order). We
keep short snake_case keys for the corpus + plots and carry the exact label
string the classifier emits so the adapter can translate without guessing.

`GENERIC` is the 16th bucket: it is what a classifier returns when nothing
matches. It is a legal prediction but never a ground-truth corpus mode -- an
injected fault always has a known cause.
"""

from __future__ import annotations

# order matches gpualert's ERROR_PATTERNS priority (first match wins). Keeping
# the same order makes the confusion matrix read top-left to bottom-right in
# the same priority the classifier actually applies.
CLASSES = [
    "cuda_oom",
    "nccl",
    "cuda_runtime",
    "ram_oom",
    "segfault",
    "file_not_found",
    "permission",
    "missing_module",
    "div_zero",
    "device_mismatch",
    "nan_loss",
    "oom_killer",
    "traceback",
    "assertion",
    "runtime_error",
]

GENERIC = "generic"

# Exact label string gpualert prints -> our short key. Lifted verbatim from
# ERROR_PATTERNS so a label change upstream fails loudly in tests rather than
# silently misscoring.
LABEL_TO_MODE = {
    "GPU out-of-memory (CUDA OOM)": "cuda_oom",
    "NCCL communication error": "nccl",
    "CUDA runtime error": "cuda_runtime",
    "System out-of-memory (RAM)": "ram_oom",
    "Segmentation fault": "segfault",
    "File not found": "file_not_found",
    "Permission denied": "permission",
    "Missing Python module": "missing_module",
    "Division by zero": "div_zero",
    "Tensor device mismatch": "device_mismatch",
    "NaN detected in loss": "nan_loss",
    "Process killed by OS (OOM)": "oom_killer",
    "Python exception (traceback)": "traceback",
    "Assertion failed": "assertion",
    "Python RuntimeError": "runtime_error",
}

# Human-readable names for tables/figures.
PRETTY = {
    "cuda_oom": "CUDA OOM",
    "nccl": "NCCL",
    "cuda_runtime": "CUDA runtime",
    "ram_oom": "RAM OOM",
    "segfault": "Segfault",
    "file_not_found": "FileNotFound",
    "permission": "Permission",
    "missing_module": "MissingModule",
    "div_zero": "DivByZero",
    "device_mismatch": "DeviceMismatch",
    "nan_loss": "NaN loss",
    "oom_killer": "OOM-killer",
    "traceback": "Traceback",
    "assertion": "Assertion",
    "runtime_error": "RuntimeError",
    "generic": "generic",
}

ALL_PRED_LABELS = CLASSES + [GENERIC]


def is_class(mode: str) -> bool:
    return mode in CLASSES
