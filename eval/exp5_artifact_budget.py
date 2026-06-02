from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

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

            art_found = any(os.path.basename(a.path) == art.name for a in found)

            if art.name in attach_names:
                disposition = "attached"
            elif not art_found:
                disposition = "excluded_oversize"
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
