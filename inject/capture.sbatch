#!/usr/bin/env bash
#SBATCH --job-name=gpualert-corpus
#SBATCH --partition=gpu            # <-- set to your GPU partition
#SBATCH --gres=gpu:1              # one GPU is enough for the corpus
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=gpualert-corpus-%j.log
#
# Capture the labelled fault corpus on a real GPU/Slurm node, then score it.
# Submit from the repo root:   sbatch inject/capture.sbatch
#
# What you need on the node before submitting:
#   - the gpualert package importable (pip install -e ../gpualert), or
#     export GPUALERT_SRC=/path/to/gpualert
#   - this repo's deps: pip install -r requirements.txt
#   - torch + torchrun for the nccl mode; systemd-run (or a cgroup cap) for
#     oom_killer. Modes whose launcher is missing are skipped and reported.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load cuda 2>/dev/null || true          # adjust to your site's modules
# source ~/envs/gpualert/bin/activate         # or activate your venv

export PYTHONPATH="$PWD:${GPUALERT_SRC:-../gpualert}:${PYTHONPATH:-}"

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# 1. build the corpus from real faults on this node (>=30 per class)
python inject/capture.py --repeat 30

# 2. the two modes that need special launchers (capture.py skips them above;
#    run them here where torchrun / systemd-run are available)
torchrun --nproc_per_node=2 inject/nccl_timeout.py \
    > corpus/injected/nccl-injected-extra.log 2>&1 || true
systemd-run --scope -p MemoryMax=500M python inject/oom_killer.py \
    > corpus/injected/oom_killer-injected-extra.log 2>&1 || true

# 3. score it exactly like the synthetic run
python eval/run_all.py

echo "done. results in results/ ; corpus manifest in corpus/labels.jsonl"
