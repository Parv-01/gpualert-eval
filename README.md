# gpualert-eval

Evaluation harness for [gpualert](https://github.com/Parv-01/gpualert) — the
fault corpus, the baselines, and the five experiments behind the numbers in the
paper. Everything here regenerates from scratch with `make all`, and the whole
thing runs on a bare node: no scipy, no sklearn, no GPU required for the
analysis (the injectors themselves want a real GPU/Slurm node, but the corpus
can be rebuilt from recorded captures anywhere).

The short version of what this measures:

1. **Does the classifier actually work?** Precision/recall/F1 for gpualert's
   15-way failure classification against three baselines (exit-code,
   traceback-parsing, naive grep), on a labelled corpus I built by fault
   injection because no public corpus has line-level CUDA/NCCL/segfault labels.
2. **Does the log-delivery guarantee hold?** A crash-survival table comparing
   gpualert to a shell redirect and to `nohup`.
3. **What does the wrapper cost?** Wall-clock overhead, wrapped vs. bare.
4. **Is the notifier really isolated?** Exit-code preservation when email is
   down, across all 15 modes.
5. **What happens to a 1 GB checkpoint?** Artifact handling under the size
   budget.

## Why a new corpus

There isn't an off-the-shelf labelled set that matches this task. Loghub and
Loghub-2.0 are labelled for binary anomaly detection and log-template parsing;
the Philly (Jeon et al., ATC'19) and AcmeTrace (Hu et al., NSDI'24) traces have
job-level or categorical failure information but nothing at the line level that
says "this is a CUDA OOM, here's the remediation." So the corpus is part of the
contribution.

The trick that makes labelling cheap is fault injection: if I run
`inject/cuda_oom.py`, whatever lands in the log is, by construction, a
`cuda_oom` sample. The injected fault *is* the label, which sidesteps
inter-annotator agreement entirely for the bulk of the data. A smaller held-out
"wild" set of real error excerpts (`corpus/wild/`) checks that the classifier
generalises past the clean injected captures.

## The 15 classes

These map one-to-one onto gpualert's own classifier rules
(`gpualert/parse_errors.py`, `ERROR_PATTERNS`, in priority order). They're
defined once in `eval/classes.py` and a test pins them against the upstream
labels so they can't silently drift:

| key | what it is |
|---|---|
| `cuda_oom` | GPU out of memory (CUDA OOM / CUDNN alloc failure) |
| `nccl` | NCCL collective error or watchdog timeout |
| `cuda_runtime` | `RuntimeError: CUDA error` (device-side assert, illegal access) |
| `ram_oom` | host `MemoryError` / `std::bad_alloc` |
| `segfault` | segmentation fault (core dumped) |
| `file_not_found` | `FileNotFoundError` |
| `permission` | `PermissionError` |
| `missing_module` | `ModuleNotFoundError` / `ImportError` |
| `div_zero` | `ZeroDivisionError` |
| `device_mismatch` | `RuntimeError: ... tensors ... same device` |
| `nan_loss` | NaN printed to stdout (the silent one) |
| `oom_killer` | killed by the kernel/cgroup OOM-killer (SIGKILL) |
| `traceback` | a Python exception with no dedicated rule (KeyError, ValueError, ...) |
| `assertion` | `AssertionError` |
| `runtime_error` | a plain `RuntimeError` |

A 16th bucket, `generic`, is what any classifier returns when nothing matches.
It's a legal prediction but never a ground-truth label — an injected fault
always has a known cause.

## Layout

```
inject/        15 fault injectors + recorded reference captures
corpus/        build script, the generated corpus, and the wild test set
baselines/     exit-code, traceback-parsing, and grep classifiers
eval/          adapter to the real gpualert classifier, metrics, stats, exp1-5
results/       generated tables and figures (committed so you can read them without a run)
tests/         pinned checks for the stats + label mapping + baselines
bench.py       one command to build, run, test, and verify reproducibility
ROADMAP.md     what's done, what needs real hardware, what would make it stronger
```

## Running it

The fastest path is the bench script, which does everything and ends in a
PASS/FAIL summary including a reproducibility check:

```bash
pip install -r requirements.txt        # numpy + matplotlib, that's it
pip install -e ../gpualert             # make the classifier importable
python bench.py
```

If you'd rather not install the package, set `GPUALERT_SRC` to the directory
holding the `gpualert/` package instead (`export GPUALERT_SRC=../gpualert`); the
adapter also auto-discovers it when this repo sits next to the package.

Step by step, if you prefer:

```bash
make all        # build the corpus, then run experiments 1-5
# or:
make corpus     # python corpus/build.py   -> corpus/labels.jsonl + corpus/injected/
make eval       # python eval/run_all.py    -> everything under results/
make test       # python -m pytest -q tests
```

On Windows without `make`, just run the `python ...` commands in the comments.
`make eval` is resilient: each experiment is isolated, so if one can't run in
your environment the others still produce their results.

New here? Read `ROADMAP.md` — it lays out where to start, what only you can do
(the real-hardware corpus, the second-annotator kappa), and the optional
hardening.

### Building the corpus

There are two routes and they produce the same manifest format.

On a real GPU/Slurm node, `python inject/capture.py --repeat 30` (or `bash
inject/run_all.sh`, which wraps it) drives each injector and captures the logs
verbatim, recording the GPU name, driver and real exit code — this is the canonical
corpus and what the paper's headline numbers come from. `REPEAT=30
inject/run_all.sh` controls samples per class. The `nccl` and `oom_killer`
injectors need special launchers (`torchrun` and `systemd-run` respectively);
their module docstrings have the exact commands.

Anywhere else, `python corpus/build.py` replays the recorded captures in
`inject/reference/` and applies seeded, surface-level mutations — different
hosts, pids, line numbers, batch sizes, allocation sizes, partial vs. full
tracebacks — to rebuild a corpus that exercises the same classifier code paths
without a GPU. The mutations are deliberately cosmetic: they never touch the
decisive error line, because that line is the label. The committed corpus was
built this way so the repo is reproducible on any machine; regenerating it is
deterministic given the seed in `corpus/build.py`.

The label schema (one JSON object per sample in `corpus/labels.jsonl`) is
documented in `schema.md`.

For the full on-node workflow -- interactive and Slurm (`inject/capture.sbatch`)
-- see `docs/REAL_HARDWARE.md`.

## The experiments

### Experiment 1 — classification quality (`eval/exp1_classifier.py`)

The centerpiece. Each of the four classifiers labels every sample; I score
per-class and macro/micro precision/recall/F1 over the 15 classes, put a
1000-resample bootstrap 95% CI on macro-F1, and run an exact McNemar test on the
paired per-sample correctness of gpualert vs. the strongest baseline (grep).

On the committed corpus (510 samples — 480 injected at 32/class, plus 30 wild):

| classifier | macro-F1 | 95% CI | accuracy |
|---|---|---|---|
| gpualert | **0.905** | [0.884, 0.924] | 0.910 |
| grep | 0.830 | [0.822, 0.838] | 0.861 |
| traceback | 0.558 | [0.551, 0.565] | 0.596 |
| exitcode | 0.133 | [0.133, 0.133] | 0.133 |

gpualert's macro-F1 advantage over grep has non-overlapping bootstrap CIs, and
McNemar gives p ≈ 0.026 on the discordant pairs. The exit-code baseline is the
floor: it can only ever be right on `segfault` and `oom_killer`, which is
exactly what Slurm's `--mail-type=FAIL` gives you.

The confusion matrix (`results/exp1_confusion.png`) is where the interesting
part is. gpualert is perfect on 12 of 15 classes; the cost shows up on
`assertion` and `runtime_error`, which drop to ~0.32 recall. That's not noise —
it's a real property of the ordered rule set. The generic `Traceback (most
recent call last)` rule sits at a higher priority than the specific
`AssertionError`/`RuntimeError` rules, so any of those that arrive with a full
traceback header get absorbed into `traceback` (which is why `traceback`'s
precision is only 0.42 — it's catching the overflow). The bare variants, where
the message is logged without a traceback header, classify correctly. The paper
discusses this under the Determinism & Totality property: the rule order is a
deliberate specificity ordering, and the confusion matrix is how its limits get
quantified rather than hand-waved.

The same scores are also reported sliced by source in
`results/exp1_by_source.csv`. This is the generalisation check: gpualert scores
0.93 macro-F1 on the held-out wild excerpts it has never seen, slightly above its
injected-set score, so the result isn't an artifact of the clean injected
captures. To make the wild labels themselves auditable, `eval/interrater.py`
generates a blind relabelling sheet and scores a second annotator against the
gold labels with Cohen's kappa.

### Experiment 2 — log survival (`eval/exp2_log_survival.py`)

Twenty trials each of {gpualert wrapper, shell redirect, nohup} × {python
exception, segfault, SIGKILL, exec-failure}, all with real subprocesses. Two
DVs: did a sentinel line printed-and-flushed just before the crash survive to
disk, and is the log file non-empty afterwards. Wilson CIs on the fractions,
Fisher's exact on the decisive cell.

The honest result: when the program flushes before it dies — which a training
loop printing progress does — the sentinel survives a plain redirect just as
well as it survives gpualert. That's worth stating plainly; the wrapper does not
magically rescue output the program never flushed. Where gpualert is actually
different is the *durability of the destination*: it creates the log file before
the child is spawned, so even a command that never execs (a typo'd binary, a bad
interpreter) leaves a non-empty, diagnostic log. On the exec-failure case,
gpualert leaves a usable log in 20/20 trials and a bare redirect leaves an empty
file in 0/20 (Fisher exact p ≈ 1.5e-11). This is the operational version of the
Log Durability invariant in the paper: the log handle exists from before
`spawn` until the wrapper exits, so there is always somewhere for output to go.

### Experiment 3 — overhead (`eval/exp3_overhead.py`)

The wrapper does a fixed amount of work per job — make a log dir, spawn, two
reader threads, write a footer — so the overhead should be a small constant the
job length swamps. Thirty trials (warm-ups discarded) on a no-op and a short
Python workload, wrapped vs. bare, with a Welch's t-test on the difference.

Measured overhead is sub-millisecond and not statistically distinguishable from
zero on these tiny workloads (mean difference ≈ 0.1–0.8 ms, p ≈ 0.11 and 0.77).
For any real training job this is in the noise, which is the point: the cost is
O(1) in the job, not O(work).

### Experiment 4 — notifier isolation (`eval/exp4_isolation.py`)

The claim is that whether the email goes out or not, the wrapper's exit code is
the child's exit code. I drive the real `EmailNotifier` against a refused SMTP
endpoint for a `JobResult` in each of the 15 modes and check that (a) `send()`
returns a failure result without ever raising, and (b) the CLI's exit decision,
`0 if result.is_success() else 1`, is a pure function of the child status and
doesn't move when the notification fails. All 15 modes pass on both counts. This
is the bulkhead framing from the paper: notifier failures stay on the notifier's
side of the wall.

### Experiment 5 — artifact budget (`eval/exp5_artifact_budget.py`)

A size sweep from 1 KB to 1 GB through the real `find_artifacts` /
`prepare_attachments` path. Files up to the single-file cap (25 MB) are
attached; anything bigger is dropped from the attachment set rather than blowing
up the email, and the log file is attached on failure regardless of budget. So a
1 GB checkpoint degrades predictably — it's excluded with the logs still
delivered — instead of the tool trying to mail a gigabyte.

| artifact size | outcome | log still attached? |
|---|---|---|
| 1 KB – 10 MB | attached | yes |
| 30 MB, 60 MB, 1 GB | excluded (over single-file cap) | yes |

## Comparison with other tools

The exit-code baseline in Experiment 1 *is* Slurm's `--mail-type` in disguise.
The tool people most often raise as prior art, `knockknock`, isn't in the
experiments on purpose: it's a notification decorator, not a failure classifier,
so there's nothing to score in a confusion matr