#!/usr/bin/env python3
from _common import banner, need_torch

if __name__ == "__main__":
    banner("cuda_oom")
    torch = need_torch()
    block = 1024 * 1024 * 1024 // 4
    hold = []
    while True:
        hold.append(torch.empty(block, dtype=torch.float32, device="cuda"))
        torch.cuda.synchronize()
