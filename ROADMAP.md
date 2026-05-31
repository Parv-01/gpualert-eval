# Roadmap — what's done, what's left, what needs you

This is the working state of the evaluation, in plain terms: what already runs,
what only you can do (it needs real hardware or a second person), and the
optional things that would make the evidence harder to argue with.

## Where to start

```bash
cd gpualert-eval
pip install -r requirements.txt
pip install -e ../gpualert        # makes the classifier importable
python bench.py                   # builds corpus, runs everything, verifies it
```

If `bench.py` ends in `overall: PASS`, the bench is healthy on your machine and
the committed numbers reproduce. Read `README.md` for the per-experiment walk
through and `results/` for the tables and the confusion figure.

## Done (runs today, on any machine)

- 15 fault injectors + recorded reference captures (`inject/`).
- Portable corpus builder — 510 labelled samples, 480 injected at 32/class plus
  30 wild (`corpus/build.py`, seeded and deterministic).
- Three baselines (exit-code, traceback-parse, grep) and an adapter onto the
  real gpualert classifier (`baselines/`, `eval/`).
- Experiments 1–5 with from-scratch stats (bootstrap, McNemar, Fisher, Wilson,
  Welch, Cohen's kappa). All five produce committed tables/figures in `results/`.
- Generalisation slice: macro-F1 reported separately for injected vs wild
  (`results/exp1_by_source.csv`) — gpualert holds up on the unseen wild set.
- Feature comparison vs knockknock / Slurm / DCGM (`results/feature_comparison.md`).
- Real-hardware capture harness (`inject/capture.py`) that records gpu/driver/
  exit-code and surfaces signal messages, a Slurm template (`inject/capture.sbatch`),
  and a step-by-step guide (`docs/REAL_HARDWARE.md`).
- Per-GPU macro-F1 slice (`results/exp1_by_gpu.csv`) and Cohen's-kappa inter-
  annotator workflow (`eval/interrater.py`).
- `bench.py` one-command run + reproducibility check. `tests/` (12 checks).

## Needs you — real hardware (this is the big one for the paper)

The committed numbers come from the *portable synthetic corpus* (recorded
captures replayed with seeded mutation). That's perfect for reproducibility and
for the methods section, but the paper's headline table should come from logs
captured on a real GPU/Slurm node. Only you can run that, because it needs a GPU
and a scheduler.

On a node (full step-by-step in `docs/REAL_HARDWARE.md`):

```bash
# all 15 injectors, recording gpu/driver/exit-code into an eval-ready manifest:
python inject/capture.py --repeat 30        # or: REPEAT=30 bash inject/run_all.sh
# the two that need special launchers:
torchrun --nproc_per_node=2 inject/nccl_timeout.py
systemd-run --scope -p MemoryMax=500M python inject/oom_killer.py
```

Then drop the captured logs into `inject/reference/` (overwriting mine) so the
builder reproduces *your* node, or wire `run_all.sh` to write straight into
`corpus/injected/`. Re-run `python bench.py`. The kind of corpus you're building
here is: one captured stdout+stderr log per real fault, labelled by which
injector produced it — that's why no hand-annotation is needed. Aim for ≥30 per
class so the per-class CIs stay tight; vary batch size, driver, and full-vs-
partial tracebacks (the injectors already take the relevant flags) so the
classifier can't key on one exact string.

Capture the `gpu`/`driver` strings into the manifest while you're at it (the
schema already has the fields) so you can show the classifier isn't secretly
keying on a driver version.

## Needs you — a second person (cheap, high credibility)

The wild set is the generalisation probe, so its labels need to be trustworthy.
Have a colleague label a subset blind and report Cohen's kappa:

```bash
python eval/interrater.py template --n 50     # writes a blank relabel sheet
# colleague fills the human_label column
python eval/interrater.py score corpus/wild/relabel_template.tsv
```

Report the kappa in the paper next to the wild-set numbers. While you're there,
expand the wild set from 2 to 3–5 real excerpts per class and replace the
per-class source notes in `corpus/wild/sources.md` with the exact permalinks you
pulled each from. That's the one provenance gap a reviewer can poke at.

## Optional — makes it more robust

- **Bigger corpus.** Bump `N_PER_CLASS` in `corpus/build.py`. Tighter CIs.
- **More baselines.** A regex/log-template parser (Drain) would be a stronger
  third comparator than grep if you want to pre-empt "why not a real log parser?"
- **Calibration / abstention.** gpualert already emits a confidence
  (`get_error_confidence`); an experiment showing it abstains to `generic` rather
  than mislabelling on low-confidence inputs would strengthen the safety story.
- **Fix or own the traceback shadow.** Experiment 1 shows `assertion` /
  `runtime_error` recall at ~0.32 because the generic `Traceback` rule outranks
  the specific ones. Either reorder those rules upstream and re-run (the eval
  will pick up the change automatically), or keep it and discuss it as a
  precision/recall trade-off. Don't leave it unexplained.
- **Per-GPU slice.** Already wired up (`results/exp1_by_gpu.csv`); it just needs
  a real corpus spanning a few GPU types/drivers to become meaningful.

## Not in this repo (paper-side)

Writing the §Evaluation / §Reliability Properties / §Related Work sections,
fixing the three cited statistics, and picking the venue all live in the paper,
not here. This repo's job is to produce the numbers and the artifact those
sections cite.
