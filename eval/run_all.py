from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import (
    exp1_classifier,
    exp2_log_survival,
    exp3_overhead,
    exp4_isolation,
    exp5_artifact_budget,
)

EXPERIMENTS = [
    ("Experiment 1: classification quality", exp1_classifier.run),
    ("Experiment 2: log survival", exp2_log_survival.run),
    ("Experiment 3: wrapper overhead", exp3_overhead.run),
    ("Experiment 4: notifier isolation", exp4_isolation.run),
    ("Experiment 5: artifact budget", exp5_artifact_budget.run),
]

def main() -> int:
    failures = 0
    for title, fn in EXPERIMENTS:
        print(f"\n=== {title} ===")
        try:
            fn()
            print("  ok")
        except Exception:
            failures += 1
            print("  FAILED:")
            traceback.print_exc()
    print(f"\nDone. {len(EXPERIMENTS) - failures}/{len(EXPERIMENTS)} experiments "
          f"produced results in results/.")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
