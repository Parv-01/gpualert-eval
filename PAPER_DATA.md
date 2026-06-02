# Paper data sheet — gpualert evaluation (real Tesla V100 run)

> Purpose: a single source of truth for writing the paper. **Every number below
> was read directly from the committed files in `results/` and `corpus/`. Use
> ONLY these numbers. Do not invent, round differently, or extrapolate. Where a
> number is inflated or a slice is misleading, it is flagged explicitly — keep
> those caveats in the paper.**

## 1. Setup (state exactly this)

- Hardware: **NVIDIA Tesla V100-PCIE-32GB**, driver **460.91.03**, PyTorch 2.7.1+cu118, Python 3.13.
- Corpus: **474 labelled logs**, 15 failure classes. Breakdown by source
  (`corpus/labels.jsonl`): **360 injected** (real captures on the V100, 12 classes),
  **90 synthetic** (3 classes, see §4), **24 wild** (held-out real excerpts from
  public issue trackers).
- Classifier under test: gpualert's ordered rule matcher (`gpualert/parse_errors.py`).
- Baselines: `exitcode` (= Slurm `--mail-type`), `traceback` (parses the Python
  exception type), `grep` (naive keyword first-match).
- Author change made during evaluation: the generic `Traceback` rule in
  `parse_errors.py` was **reordered to the lowest priority** so specific
  `AssertionError`/`RuntimeError` rules match first. Report this honestly — it
  raised `assertion`/`runtime_error` recall and lifted macro-F1.

## 2. Headline result — Experiment 1 (classification quality)

Source: `results/exp1_summary.csv`. Full 15-class corpus (474 samples).

| classifier | macro-F1 | 95% CI | micro-F1 | accuracy |
|---|---|---|---|---|
| **gpualert** | **0.9979** | [0.9929, 1.0000] | 0.9979 | 0.9979 |
| grep | 0.8302 | [0.8217, 0.8376] | 0.8586 | 0.8586 |
| traceback | 0.5593 | [0.5522, 0.5654] | 0.6013 | 0.6013 |
| exitcode | 0.1333 | [0.1333, 0.1333] | 0.1308 | 0.1308 |

Paired significance (`results/exp1_mcnemar.txt`), gpualert vs grep:
both correct 406, only gpualert 67, only grep 1, discordant 68, **exact p = 4.68e-19**.

**HONESTY CAVEAT (must keep in the paper):** the all-15 macro-F1 of 0.9979 is
slightly inflated because 3 classes are synthetic and match the classifier's
reference strings by construction (§4). The defensible classifier number is the
**12 real-hardware classes: macro-F1 = 0.9974** (computed from
`results/exp1_per_class.csv` over the 12 injected classes). Report both, e.g.
"0.997 macro-F1 on the 12 classes reproduced on real hardware; 0.998 over all 15
including 3 reference-captured classes."

## 3. Per-class (Experiment 1, gpualert)

Source: `results/exp1_per_class.csv`. precision / recall / F1 / support.

- 13 classes at **1.000 / 1.000 / 1.000**: cuda_oom (30), nccl (30), cuda_runtime (32),
  ram_oom (32), segfault (32), file_not_found (32), permission (32), missing_module (32),
  div_zero (32), device_mismatch (32), nan_loss (32), oom_killer (30), runtime_error (32).
- `traceback`: P 0.9697, R 1.000, **F1 0.9846** (32). Slightly <1 because one
  `assertion` sample falls into it.
- `assertion`: P 1.000, R 0.9688, **F1 0.9841** (32). One sample missed.

(cuda_oom, nccl, oom_killer reaching 1.0 is expected — they are synthetic; do not
present them as evidence of real-world detection.)

## 4. The 3 synthetic classes — say this plainly

Source: `corpus/labels.jsonl` (`source: synthetic`). On this single old-driver
V100 node, three faults could not be reproduced faithfully, so they use the
recorded reference signatures (clearly tagged `synthetic`):

- **cuda_oom** — a large CUDA allocation hit a torch-2.7/driver-460 NVML
  incompatibility (`undefined symbol: nvmlDeviceGetNvLinkRemoteDeviceType`)
  before any OOM, on both a single oversized allocation and incremental
  allocation. The node cannot produce a real CUDA OOM with this driver.
- **nccl** — a single shared GPU with 2 ranks completed the collective; a real
  NCCL fault needs ≥2 GPUs / multi-node.
- **oom_killer** — needs a memory-capped cgroup (Slurm `--mem` or root
  `systemd-run`); the node had no scheduler (`salloc` absent) and no privileges.

Honest paper sentence: *"12 of 15 classes were reproduced on a real Tesla V100
(driver 460.91); cuda_oom, nccl and oom_killer use reference captures because the
node's driver/scheduler/single-GPU configuration prevented faithful in-situ
reproduction. These are tagged `source: synthetic` in the released corpus."*

## 5. Generalisation — Experiment 1 by source

Source: `results/exp1_by_source.csv`.

| slice | n | gpualert accuracy | grep accuracy | traceback acc | exitcode acc |
|---|---|---|---|---|---|
| injected (real) | 360 | **1.000** | 0.833 | 0.667 | 0.083 |
| wild (held-out) | 24 | **0.958** | 0.708 | 0.625 | 0.083 |

**HONESTY CAVEAT:** the macro-F1 column in `exp1_by_source.csv` (injected 0.80,
wild 0.764) is an **artifact**, not a performance number — each slice contains
only a subset of the 15 classes, so absent classes score 0 and drag macro down.
**Use the accuracy column for the per-slice story.** On real injected logs
gpualert is 100% accurate; on unseen wild logs 95.8% (vs grep 70.8%).

Per-GPU (`results/exp1_by_gpu.csv`): single device, Tesla-V100-PCIE-32GB, n=450,
accuracy 1.000. (Only one GPU type in this corpus, so this slice is not
informative on its own — note that.)

## 6. Experiment 2 — log durability

Source: `results/exp2_log_survival.csv`, `results/exp2_fisher.txt`. 20 trials per
cell, {gpualert, shell-redirect, nohup} × {python_exception, segfault, sigkill,
exec_failure}.

- On real crashes (python_exception, segfault, sigkill) where the program flushed
  before dying: a printed line survives in **all three** methods (20/20 each).
  **State this honestly — gpualert is not uniquely better here.**
- On **exec_failure** (command never starts): gpualert leaves a non-empty log
  **20/20**, shell redirect **0/20** (nohup 20/20). Fisher exact, gpualert vs
  redirect, **p = 1.45e-11**.
- Claim to make: gpualert's guarantee is the **durable destination** — the log
  file exists from before the child starts, so even an exec failure leaves a
  diagnostic log; it does not magically recover unflushed buffers.

## 7. Experiment 3 — wrapper overhead

Source: `results/exp3_overhead.csv`. 30 trials, warm-ups discarded, Welch's t.

| workload | bare mean | wrapped mean | overhead | Welch p |
|---|---|---|---|---|
| noop | 36.32 ms | 40.10 ms | **+3.78 ms** | 8.5e-4 |
| short_py | 42.65 ms | 45.26 ms | **+2.61 ms** | 2e-5 |

**HONESTY CAVEAT:** the overhead is **small but statistically significant (~3 ms)**
on this node — do NOT write "zero overhead". Correct framing: a fixed ~3 ms
constant per job, independent of job length (O(1)), negligible for any real
training run.

## 8. Experiment 4 — notifier isolation

Source: `results/exp4_isolation.txt`. **15/15** failure modes preserved the
child's exit code with SMTP down, and no exception escaped the notifier. The
wrapper's exit code is a pure function of the child status (bulkhead property).

## 9. Experiment 5 — artifact budget

Source: `results/exp5_artifact_budget.csv`.

| artifact size | outcome | log still attached |
|---|---|---|
| 1 KB – 10 MB | attached | yes |
| 30 MB, 60 MB, 1 GB | excluded (over 25 MB single-file cap) | yes |

Predictable degradation: oversized artifacts are dropped, logs are always
delivered on failure. No attempt to email a 1 GB checkpoint.

## 10. Related work / positioning (feature comparison)

Source: `results/feature_comparison.md`. gpualert is not compared to `knockknock`
in Experiment 1 because knockknock is a notification decorator, not a classifier.
The feature table positions gpualert vs knockknock / Slurm `--mail-type` / NVIDIA
DCGM. Honest reading: knockknock wins on notification-channel breadth; DCGM is
complementary (hardware health); gpualert is alone in combining zero-instrumentation
+ 15-way classification + a log-durability guarantee for any wrapped command.

## 11. Repositories to cite

- gpualert (the tool): https://github.com/Parv-01/gpualert
- gpualert-eval (this corpus + harness): https://github.com/Parv-01/gpualert-eval
  *(create this repo and push — see the commit guide; update the URL if the name
  differs).*

Cite gpualert-eval as the artifact behind §Evaluation, and release the corpus
(`corpus/`) as a small datasets contribution.

## 12. Rules for whoever drafts the paper (human or Claude)

1. Use only the numbers in this file. If a number isn't here, do not state it —
   re-run the experiment or leave it out.
2. Always pair the 0.998 all-15 figure with the 0.997 real-12 figure and the
   synthetic-class caveat (§2, §4).
3. For per-slice (§5) use **accuracy**, never the artifact macro-F1.
4. Never write "zero overhead" — it's ~3 ms, significant (§7).
5. Never claim gpualert uniquely rescues output on every crash — only the
   durable-destination / exec-failure case is exclusive (§6).
6. Disclose the `parse_errors.py` rule reorder as an author change (§1).
7. Keep the hardware caveat: single old-driver V100, 24-sample wild set, 3
   synthetic classes.
