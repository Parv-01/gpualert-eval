"""Shared bits for the fault injectors.

Each injector is a standalone script that triggers exactly one failure mode on
a real GPU/Slurm node. The point is that the *injected fault is the label*: if
you run inject/cuda_oom.py, the resulting log is ground-truth `cuda_oom`, no
annotation needed.

Keep these dependency-light. Torch is imported lazily inside the injectors
that need it so the non-GPU ones still run anywhere.
"""

from __future__ import annotations

import os
import sys


def banner(mode: str) -> None:
    """Print a small header so a captured log is self-identifying."""
    host = os.uname().nodename if hasattr(os, "uname") else "unknown"
    print(f"[inject] mode={mode} host={host} pid={os.getpid()}", flush=True)


def need_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on node
        print(f"[inject] torch unavailable: {e}", file=sys.stderr, flush=True)
        sys.exit(3)
    import torch
    if not torch.cuda.is_available():
        print("[inject] no CUDA device visible", file=sys.stderr, flush=True)
        sys.exit(3)
    return torch
