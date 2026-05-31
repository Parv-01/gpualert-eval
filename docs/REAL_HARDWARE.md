# Running on real hardware (GPU / Slurm node)

The committed results come from the portable synthetic corpus so the repo runs
anywhere. For the paper you want the same experiments on logs captured from a
real GPU/Slurm node. This is the step-by-step. Nothing here needs hand-labelling
-- the injector that produced a log *is* its label.

## What you need on the node

- A GPU visible to the node (`nvidia-smi` works).
- Python with this repo's deps: `pip install -r requirements.txt`.
- The gpualert package importable: `pip install -e ../gpualert` (or
  `export GPUALERT_SRC=/path/to/gpualert`).
- `torch` + `torchrun` (for the `nccl` mode) and `systemd-run` or a cgroup
  memory cap (for the `oom_killer` mode). Anything missing is skipped and
  reported -- a partial node still gives a partial corpus.

You do **not** need to add any new files to the node beyond this repo. The
capture harness (`inject/capture.py`) and the Slurm script (`inject/capture.sbatch`)
are already here.

## Option A — interactive node (you have a shell on the GPU box)

```bash
cd gpualert-eval
export GPUALERT_SRC=../gpualert            # or pip install -e ../gpualert

# 1. capture >=30 real samples per class. Records GPU name, driver and the real
#    exit code into corpus/labels.jsonl; surfaces signal messages (segfault,
#    killed) the way a Slurm job would.
python inject/capture.py --repeat 30

# 2. the two modes that need special launchers (capture.py skips them, so run
#    them where torchrun / systemd-run exist):
torchrun --nproc_per_node=2 inject/nccl_timeout.py \
    > corpus/injected/nccl-injected-extra.log 2>&1 || true
systemd-run --scope -p MemoryMax=500M python inject/oom_killer.py \
    > corpus/injected/oom_killer-injected-extra.log 2>&1 || true

# 3. score it exactly like the synthetic run, and verify
python eval/run_all.py
python bench.py        # optional: confirms determinism end to end
```

Results land in `results/` (same files as the committed run, now from real logs):
`exp1_summary.csv`, `exp1_per_class.csv`, `exp1_confusion.png`,
`exp1_by_source.csv`, `exp1_by_gpu.csv`, and the Exp 2-5 tables.

## Option B — submit through Slurm

Edit the `#SBATCH` lines in `inject/capture.sbatch` for your partition/account,
then from the repo root:

```bash
sbatch inject/capture.sbatch
```

It runs steps 1-3 above on an allocated GPU and writes the job log next to the
results. To sweep more than one GPU type (so `exp1_by_gpu.csv` is meaningful),
submit it once per partition / `--constraint` and let each run append to the
corpus, or capture into per-GPU manifests and concatenate.

## What "build a corpus" means here

One captured stdout+stderr log per real fault, plus a manifest row. The manifest
(`corpus/labels.jsonl`) carries, per sample: `mode` (the injector = the label),
`source` (`injected`), `log_path`, `gpu`, `driver`, and `true_exit_code`. The
harness fills all of these automatically. Aim for:

- **>=30 samples/class** so the per-class bootstrap CIs stay tight.
- **variety within a class** so the classifier can't memorise one string. The
  injectors already vary line numbers, sizes and full-vs-bare tracebacks; run
  across a couple of GPU types and driver versions if you can.
- **the wild set untouched** -- it's folded in automatically and stays the
  held-out generalisation probe.

## Replacing the synthetic corpus permanently

If you want the repo's default corpus to *be* your real captures (so a plain
`make all` reproduces them), copy your captured logs over the recorded
references:

```bash
cp corpus/injected/cuda_oom-injected-0001.log inject/reference/cuda_oom.log
# ...one representative capture per mode...
```

After that, `python corpus/build.py` on any machine replays *your* node's logs
instead of mine. Commit the updated `inject/reference/*.log` and the regenerated
`corpus/` + `results/`.

## Honest notes

- A bare `subprocess` can't capture a process's death-by-signal notice; the
  harness runs each injector under `bash -c` so the shell prints "Segmentation
  fault (core dumped)" / "Killed" into the log, which is what a real Slurm job
  log contains. Exit codes are normalised from bash's 128+N form back to the
  signed `-11` style used everywhere else.
- `oom_killer` attribution lives in the kernel log ("Out of memory: Killed
  process ..."). The harness appends that line from `journalctl -k`/`dmesg` when
  it can read it; without kernel-log access you only get the shell's "Killed",
  so run that mode where you can read the journal (the sbatch path usually can).
- Expect the real confusion matrix to still show the `assertion`/`runtime_error`
  -> `traceback` shadow from priority ordering; that's a property of the rules,
  not the corpus.
