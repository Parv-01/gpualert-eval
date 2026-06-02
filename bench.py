import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
DETERMINISTIC = [
    "results/exp1_summary.csv",
    "results/exp1_per_class.csv",
    "results/exp1_confusion.csv",
    "results/exp1_by_source.csv",
]

def _env():
    e = dict(os.environ)
    e["PYTHONPATH"] = str(ROOT) + os.pathsep + e.get("PYTHONPATH", "")
    return e

def _run(cmd):
    return subprocess.run(cmd, cwd=str(ROOT), env=_env(), capture_output=True, text=True)

def _hash(paths):
    h = hashlib.sha256()
    for rel in paths:
        p = ROOT / rel
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()

def main():
    r = _run([PY, "eval/run_all.py"])
    print(r.stdout.strip() or r.stderr.strip())
    before = _hash(DETERMINISTIC)
    _run([PY, "eval/exp1_classifier.py"])
    after = _hash(DETERMINISTIC)
    print("deterministic:", before == after)
    return 0 if r.returncode == 0 and before == after else 1

if __name__ == "__main__":
    raise SystemExit(main())
