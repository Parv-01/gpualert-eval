#!/usr/bin/env python3
from _common import banner

if __name__ == "__main__":
    banner("traceback")
    cfg = {"lr": 3e-4}

    warmup = cfg["warmup_steps"]
    print(warmup)
