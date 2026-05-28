#!/usr/bin/env python3
"""Start a 2-rank NCCL group with a sub-second timeout, then stall one rank.

Run under torchrun with 2 procs on one node:
    torchrun --nproc_per_node=2 inject/nccl_timeout.py
Rank 1 sleeps past the watchdog so rank 0's collective trips a NCCL timeout.
"""
import datetime
import os
import time

from _common import banner

if __name__ == "__main__":
    banner("nccl")
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(milliseconds=800),
    )
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    t = torch.ones(1, device="cuda")
    if rank == 1:
        time.sleep(30)  # blow past the 800ms watchdog
    dist.all_reduce(t)  # rank 0 raises a NCCL timeout here
    print("done", rank)
