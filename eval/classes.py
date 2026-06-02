from __future__ import annotations

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
