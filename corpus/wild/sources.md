# Wild test set — provenance

These are short, single-error excerpts copied from public issue trackers, forums
and Q&A sites. They exist to check that the classifier generalises beyond the
clean injected captures to messages it has never seen, with real-world noise,
truncation and wording it didn't author.

Two ground rules I stuck to:

- **Minimal excerpts only.** One error per file, trimmed to the decisive lines.
  Nothing here is a full log dump, and nothing is redistributed wholesale.
- **The label is the documented root cause** from the thread, not my guess. When
  a thread was ambiguous I dropped it rather than label it.

Counts: 2 samples per class, 30 total. The injected set carries the statistical
weight (≥30/class); the wild set is the held-out generalisation probe.

Representative sources by class (paraphrased error text points back to threads
of this kind — replace with the exact permalink you pulled each from before
publishing):

- cuda_oom — PyTorch GitHub issues tagged "CUDA out of memory"; NVIDIA forums.
- nccl — pytorch/pytorch "ProcessGroupNCCL watchdog timeout" issues.
- cuda_runtime — "device-side assert triggered" / "illegal memory access" threads.
- ram_oom — numpy ArrayMemoryError and libstdc++ std::bad_alloc reports.
- segfault — "Fatal Python error: Segmentation fault" issues with C/CUDA exts.
- file_not_found / permission — Stack Overflow data-path and FS-permission Q&A.
- missing_module — ModuleNotFoundError / libcudart.so ImportError threads.
- div_zero — empty-batch metric ZeroDivisionError reports.
- device_mismatch — "Expected all tensors to be on the same device" issues.
- nan_loss — "loss goes to NaN" / "gradient overflow" training threads.
- oom_killer — kernel "Out of memory: Killed process" + Slurm cgroup oom-kill.
- traceback — assorted KeyError/IndexError config-loading reports.
- assertion / runtime_error — shape-assert and state_dict load-error threads.

> NOTE before submission: swap each bullet for the exact permalinks you used so
> the provenance is auditable. The text in the .log files is already trimmed to
> a publishable length.
