from __future__ import annotations

import os
import sys
from pathlib import Path

from eval.classes import GENERIC, LABEL_TO_MODE

NAME = "gpualert"

def _ensure_importable() -> None:
    try:
        import gpualert.parse_errors
        return
    except Exception:
        pass
    candidates = []
    env = os.environ.get("GPUALERT_SRC")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()

    candidates += [
        here.parents[2] / "gpualert",
        here.parents[2],
        here.parents[1].parent / "gpualert",
    ]
    for c in candidates:
        if c and (c / "gpualert" / "parse_errors.py").exists():
            sys.path.insert(0, str(c))
            return
        if c and (c / "parse_errors.py").exists():
            sys.path.insert(0, str(c.parent))
            return
    raise ImportError(
        "Could not import gpualert. Set GPUALERT_SRC to the directory holding "
        "the gpualert/ package."
    )

_ensure_importable()
from gpualert.parse_errors import parse_errors

def classify(sample: dict) -> str:
    summary = parse_errors(
        stdout="",
        stderr=sample["log"],
        exit_code=sample.get("true_exit_code", 0) or 0,
    )
    if not summary:
        return GENERIC
    label = summary.splitlines()[0].strip()
    if label in LABEL_TO_MODE:
        return LABEL_TO_MODE[label]

    return GENERIC
