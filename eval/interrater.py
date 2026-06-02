from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset import load
from eval.stats import cohen_kappa

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "corpus" / "wild" / "relabel_template.tsv"
SEED = 20260530

def make_template(n: int) -> None:
    wild = [s for s in load() if s["source"] == "wild"]
    rng = random.Random(SEED)
    rng.shuffle(wild)
    chosen = wild[: min(n, len(wild))]
    lines = ["# fill in human_label with one of the 15 class keys (see eval/classes.py).",
             "# id\tlog_excerpt\thuman_label"]
    for s in chosen:
        excerpt = " ".join(s["log"].split())[:200]
        lines.append(f"{s['id']}\t{excerpt}\t")
    TEMPLATE.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(chosen)} rows -> {TEMPLATE}")
    print("fill the human_label column, then: python eval/interrater.py score "
          f"{TEMPLATE.relative_to(ROOT)}")

def score(path: str) -> None:
    gold = {s["id"]: s["mode"] for s in load()}
    a, b = [], []
    missing = 0
    for line in Path(path).read_text().splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3 or not parts[2].strip():
            missing += 1
            continue
        sid, _excerpt, human = parts[0], parts[1], parts[2].strip()
        if sid in gold:
            a.append(gold[sid])
            b.append(human)
    if not a:
        print("no labelled rows found -- fill the human_label column first.")
        return
    k = cohen_kappa(a, b)
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    print(f"scored {len(a)} rows ({missing} blank/skipped)")
    print(f"  raw agreement : {agree:.3f}")
    print(f"  Cohen's kappa : {k:.3f}")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="write a blank relabel sheet")
    t.add_argument("--n", type=int, default=50)
    s = sub.add_parser("score", help="score a filled-in sheet")
    s
