#!/usr/bin/env bash
#SBATCH --job-name=gpualert-corpus
#SBATCH --partition=gpu            # <-- set to your GPU partition
#SBATCH --gres=gpu:1              # one GPU is enough for the corpus
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G                  # tight on purpose: the oom_killer mode needs
                                  # the job cgroup to actually run out of memory
#SBATCH --time=00:30:00
#SBATCH --output=gpualert-corpus-%j.log
#
# Capture the labelled fault corpus on a real GPU/Slurm node, then score it.
# Submit from the repo root:   sbatch inject/capture.sbatch
#
# Nothing here needs root. Inside this job the step cgroup enforces --mem, so the
# oom_killer mode gets killed by the kernel for real -- no systemd-run, no polkit
# prompt. capture.py detects SLURM_JOB_ID and runs oom_killer directly.
#
# Before submitting:
#   - gpualert importable: pip install -e ../gpualert  (or set GPUALERT_SRC)
#   - this repo's deps:     pip install -r requirements.txt
#   - torch with CUDA in the env for the GPU modes (cuda_oom, cuda_runtime,
#     device_mismatch, nccl). Missing pieces are skipped and reported, not fatal.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load cuda 2>/dev/null || true          # adjust to your site's modules
# source ~/envs/gpualert/bin/activate         # or activate your venv

export PYTHONPATH="$PWD:${GPUALERT_SRC:-../gpualert}:${PYTHONPATH:-}"

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# 1. capture every mode that works here, including oom_killer (the job cgroup
#    enforces --mem, so the kernel OOM-killer fires). nccl needs torchrun + 2
#    ranks; it's launched separately below if capture.py skipped it.
python inject/capture.py --repeat 30

# 2. nccl, if torchrun is present, via its own launcher
if command -v torchrun >/dev/null 2>&1; then
    python inject/capture.py --repeat 30 --modes nccl || true
fi

# 3. score the captured corpus
python eval/run_all.py

echo "done. results in results/ ; corpus manifest in corpus/labels.jsonl"
