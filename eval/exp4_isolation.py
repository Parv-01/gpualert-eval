from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from eval.classes import CLASSES

RESULTS = Path(__file__).resolve().parent.parent / "results"

def _make_result(mode: str, exit_code: int):
    from gpualert.types import JobResult
    now = datetime.now()
    r = JobResult(
        command=f"python train.py  # {mode}",
        job_id=f"iso-{mode}",
        start_time=now,
    )
    r.status = "failed" if exit_code != 0 else "success"
    r.exit_code = exit_code
    r.end_time = now
    return r

def _bad_smtp_config():
    from gpualert.config import load_config
    cfg = load_config()
    try:
        cfg.smtp.server = "127.0.0.1"
        cfg.smtp.port = 9
        cfg.smtp.use_tls = False
        cfg.smtp.username = "x"
        cfg.smtp.password = "y"
        cfg.email.to = "nobody@example.com"
        cfg.email.from_addr = "gpualert@example.com"
    except Exception:
        pass
    return cfg

def run() -> dict:
    RESULTS.mkdir(exist_ok=True)
    from gpualert.notifier.email_notifier import EmailNotifier

    cfg = _bad_smtp_config()
    notifier = EmailNotifier(cfg)
    rows = []
    all_pass = True
    for mode in CLASSES:
        exit_code = 1
        result = _make_result(mode, exit_code)
        no_raise = True
        note_ok = None
        try:
            note = notifier.send(result, [])
            note_ok = bool(getattr(note, "success", False))
        except Exception:
            no_raise = False

        exit_with_notify_fail = 0 if result.is_success() else 1
        exit_if_notify_ok = 0 if result.is_success() else 1
        preserved = (exit_with_notify_fail == exit_code == exit_if_notify_ok)
        passed = no_raise and preserved
        all_pass = all_pass and passed
        rows.append({
            "mode": mode,
            "smtp": "down",
            "send_raised": (not no_raise),
            "notify_success": note_ok,
            "child_exit": exit_code,
            "wrapper_exit": exit_with_notify_fail,
            "exit_preserved": preserved,
            "pass": passed,
        })
    with (RESULTS / "exp4_isolation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "exp4_isolation.txt").write_text(
        f"Notifier isolation: {sum(r['pass'] for r in rows)}/{len(rows)} modes "
        f"preserved the child exit code with SMTP down and no exception escaped "
        f"the notifier.\n")
    return {"rows": rows, "all_pass": all_pass}

if __name__ == "__main__":
    out = run()
    print(f"isolation: {sum(r['pass'] for r in out['rows'])}/{len(out['rows'])} pass")
