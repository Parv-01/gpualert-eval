#!/usr/bin/env python3
"""Add a CPU tensor to a CUDA tensor -> RuntimeError: ... same device."""
from _common import banner, need_torch

if __name__ == "__main__":
    banner("device_mismatch")
    torch = need_torch()
    a = torch.ones(4, device="cuda")
    b = torch.ones(4)            # stays on CPU
    print((a + b).sum())          # raises the device-mismatch RuntimeError
