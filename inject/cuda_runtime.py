#!/usr/bin/env python3
"""Index a CUDA tensor out of bounds -> 'RuntimeError: CUDA error'.

The device-side assert surfaces as a CUDA runtime error on the next sync.
"""
from _common import banner, need_torch

if __name__ == "__main__":
    banner("cuda_runtime")
    torch = need_torch()
    idx = torch.tensor([0, 5_000_000], device="cuda")  # way out of range
    src = torch.zeros(8, device="cuda")
    out = src[idx]            # device-side assert
    torch.cuda.synchronize()  # force the error to surface
    print(out)
