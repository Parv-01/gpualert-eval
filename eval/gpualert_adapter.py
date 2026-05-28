"""Wrap the real gpualert classifier as a scorer.

We call gpualert.parse_errors.parse_errors() exactly as the tool does, then map
its emitted label back to one of our 15 mode keys. No reimplementation -- if
the upstream rules change, this picks it up.

Importing gpualert: if it isn't already on the path, set GPUALERT_SRC to the
directory that contains the `gpualert/` package, e.g.
    GPUALERT_SRC=/path/to/gpualert
This repo is meant to live next to the main package, so we also try a couple of
sensible relative locations before giving up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from eval.classes import GENERIC, LABEL_TO_MODE

NAME = "gpualert"


def _ensure_importable() -> None:
    try:
        import gpualert.parse_errors  # noqa: F401
        return
    except Exception:
        pass
    candidates = []
    env = os.environ.get("GPUALERT_SRC")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    # repo is typically a sibling of the gpualert checkout
    candidates += [
        here.parents[2] / "gpualert",          # ../../gpualert
        here.parents[2],                         # ../.. itself contains gpualert/
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
from gpualert.parse_errors import parse_errors  # noqa: E402


def classify(sample: dict) -> str:
    """Return a mode key (one of CLASSES, or 'generic')."""
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
    # the exit-code fallback message ("Process exited with code N...")
    return GENERIC
