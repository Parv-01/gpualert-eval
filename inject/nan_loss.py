#!/usr/bin/env python3
"""Drive the loss to NaN and print it the way a train loop would.

No exception is raised -- this is the silent-divergence case that only shows
up in stdout, which is exactly why an exit-code-only notifier misses it.
"""
import math

from _common import banner

if __name__ == "__main__":
    banner("nan_loss")
    loss = 2.7
    for step in range(1, 6):
        loss = loss * 10.0       # blow up
        if step >= 3:
            loss = float("nan")
        print(f"step {step:04d} | loss = {loss}", flush=True)
    if math.isnan(loss):
        print("NaN detected in loss, aborting.", flush=True)
