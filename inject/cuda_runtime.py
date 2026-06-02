#!/usr/bin/env python3
from _common import banner, need_torch

if __name__ == "__main__":
    banner("cuda_runtime")
    torch = need_torch()
    idx = torch.tensor([0, 5_000_000], device="cuda")
    src = torch.zeros(8, device="cuda")
    out = src[idx]
    torch.cuda.synchronize()
    print(out)
