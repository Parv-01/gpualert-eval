"""Experiment 5 -- artifact handling under a size budget.

gpualert attaches job outputs to the notification but has to stay inside an
email's size limits. The documented behaviour:
  - a single file over `max_single_mb` (25 MB default) is left out of the
    attachment set (too big to mail);
  - the rest are packed up to a `max_total_mb` (45 MB default) budget, and
    whatever overflows is zipped; if the zip still doesn't fit it is skipped;
  - log files are always attached on failure, budget or not.

We sweep artifact sizes from 1 KB up and record the disposition of each file
plus confirm the log is always kept. This shows the wrapper degrades
predictably instead of trying to mail a 1 GB checkpoint.

  IV  = artifact size
  DV  = disposition (attached | excluded_oversize | zipped_overflow | skipped),
        log_always_attached (bool)
"""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

# sizes in bytes; 1 GB represented sparsely so we don't actually write a GB.
SIZES = [
    ("1KB", 1 * 1024),
    ("100KB", 100 * 1024),
    ("1MB", 1 * 1024 * 1024),
    ("10MB", 10 * 1024 * 1024),
    ("30MB", 30 * 1024 * 1024),
    ("60MB", 60 * 1024 * 1024),
    ("1GB", 1024 * 1024 * 1024),
]


def _sparse_file(path: Path, size: int) -> None:
    with path.open("wb") as f:
        if size > 0:
            f.seek(size - 1)
            f.write(b"\0")


def run() -> dict:
    RESULTS.mkdir(exist_ok=True)
    from gpualert.artifacts import find_artifacts, prepare_attachments

    rows = []
    for label, size in SIZES:
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            # one artifact of this size + a small log that must always survive
            art = tmpp / "checkpoint_output.npy"
            _sparse_file(art, size)
            log = tmpp / "combined.log"
            log.write_text("[SYSTEM] job failed\n" + "x" * 2048)

            start = datetime.now() - timedelta(minutes=5)
            found = find_artifacts(start_time=start, cwd=str(tmpp))
            to_attach, skipped = prepare_attachments(
                artifacts=found,
                log_files=[str(log)],
                job_failed=True,
            )
            attach_names = {os.path.basename(p) for p in to_attach}
            skipped_names = {os.path.basename(p) for p in skipped}
            log_kept = os.path.basename(str(log)) in attach_names
            zipped = any(p.endswith(".zip") for p in to_attach)
            # did the scanner even keep the big artifact? (it drops >25MB singles)
            art_found = any(os.path.basename(a.path) == art.name for a in found)

            if art.name in attach_names:
                disposition = "attached"
            elif not art_found:
                disposition = "excluded_oversize"   # dropped by find_artifacts (>25MB single)
            elif zipped:
                disposition = "zipped_overflow"
            else:
                disposition = "skipped"

            rows.append({
                "size": label,
                "size_bytes": size,
                "artifact_kept_by_scanner": art_found,
                "disposition": disposition,
                "log_always_attached": log_kept,
            })
    with (RESULTS / "exp5_artifact_budget.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {"rows": rows}


if __name__ == "__main__":
    for r in run()["rows"]:
        print(f"{r['size']:>6}  {r['disposition']:>18}  log_kept={r['log_always_attached']}")
