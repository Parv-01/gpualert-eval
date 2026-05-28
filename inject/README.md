# Fault injectors

One script per failure mode. Each prints a small `[inject]` banner so a captured
log is self-identifying, then triggers exactly one real fault. Run on a GPU/Slurm
node; the GPU-bound ones (`cuda_*`, `device_mismatch`, `nccl_timeout`) exit 3 if
no CUDA device is visible.

`run_all.sh` drives the straightforward ones through the gpualert wrapper and
writes labelled logs into `corpus/injected/`. Two need special launchers:

    torchrun --nproc_per_node=2 inject/nccl_timeout.py
    systemd-run --scope -p MemoryMax=500M python inject/oom_killer.py

`reference/` holds one recorded capture per mode. `corpus/build.py` replays
these (with seeded cosmetic mutation) to rebuild the corpus on machines without
a GPU, so the evaluation is reproducible anywhere. If you re-capture on real
hardware, drop the new logs in `reference/` and the synthetic corpus tracks your
node instead of mine.
