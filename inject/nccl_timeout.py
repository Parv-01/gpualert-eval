#!/usr/bin/env python3
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
        time.sleep(30)
    dist.all_reduce(t)
    print("done", rank)
