#!/usr/bin/env python3
from _common import banner, need_torch

if __name__ == "__main__":
    banner("device_mismatch")
    torch = need_torch()
    a = torch.ones(4, device="cuda")
    b = torch.ones(4)
    print((a + b).sum())
