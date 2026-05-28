# Corpus label schema

Every sample in the corpus is one captured log plus one row in
`corpus/labels.jsonl`. The manifest is JSON Lines, one object per sample:

```json
{
  "id": "cuda_oom-injected-0007",
  "mode": "cuda_oom",
  "source": "injected",
  "log_path": "corpus/injected/cuda_oom-injected-0007.log",
  "gpu": "A100-SXM4-40GB",
  "driver": "535.104.05",
  "true_exit_code": 1
}
```

Fields:

- `mode` — the ground-truth failure class. One of the 15 keys in
  `eval/classes.py` (`CLASSES`). For injected samples the mode is *defined by
  the injector that produced the log*, so the label is exact by construction.
- `source` — `injected` (produced by a script in `inject/`, or by the
  portable synthesiser in `corpus/build.py` that replays a recorded reference
  capture) or `wild` (a real excerpt copied from a public issue tracker /
  forum, see `corpus/wild/sources.md`).
- `log_path` — path to the captured log, relative to the repo root. The log is
  the merged stdout+stderr the wrapper would see, exactly as written to disk.
- `gpu` / `driver` — the device the capture came from, or the device string the
  synthesiser stamped into the log. Kept so we can check the classifier isn't
  secretly keying on a driver string.
- `true_exit_code` — the process exit code. `-N` means killed by signal N
  (e.g. `-11` = SIGSEGV, `-9` = SIGKILL), matching `subprocess` conventions.

The `mode` is the only thing a classifier is scored against. Everything else is
metadata for slicing results (injected vs wild, per-GPU, etc.).
