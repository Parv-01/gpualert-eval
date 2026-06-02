from __future__ import annotations

import os
import sys

def banner(mode: str) -> None:
    host = os.uname().nodename if hasattr(os, "uname") else "unknown"
    print(f"[inject] mode={mode} host={host} pid={os.getpid()}", flush=True)

def need_torch():
    try:
        import torch
    except Exception as e:
        print(f"[inject] torch unavailable: {e}", file=sys.stderr, flush=True)
        sys.exit(3)
    import torch
    if not torch.cuda.is_available():
        print("[inject] no CUDA device visible", file=sys.stderr, flush=True)
        sys.exit(3)
    return torch
