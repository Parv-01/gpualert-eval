#!/usr/bin/env python3
"""Allocate a tensor far larger than GPU memory -> CUDA out of memory."""
from _common import banner, need_torch

if __name__ == "__main__":
    banner("cuda_oom")
    torch = need_torch()
    # ~256 GB of float32; no consumer/datacenter card holds this.
    n = 256 * 1024 * 1024 * 1024 // 4
    x = torch.empty(n, dtype=torch.float32, device="cuda")
    print(x.shape)
