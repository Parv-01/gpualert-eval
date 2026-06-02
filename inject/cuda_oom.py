#!/usr/bin/env python3
"""Fill the GPU with 1 GB tensors until it runs out -> CUDA out of memory."""
from _common import banner, need_torch

if __name__ == "__main__":
    banner("cuda_oom")
    torch = need_torch()
    block = 1024 * 1024 * 1024 // 4   # 1 GB of float32
    hold = []
    while True:
        hold.append(torch.empty(block, dtype=torch.float32, device="cuda"))
        torch.cuda.synchronize()
