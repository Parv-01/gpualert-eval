"""Load the corpus manifest + log text into memory."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "corpus" / "labels.jsonl"


def load(manifest: Path = MANIFEST, source: str | None = None) -> list[dict]:
    """Return a list of samples: {id, mode, source, gpu, driver,
    true_exit_code, log}. `source` filters to 'injected' or 'wild' if given."""
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found. Run `python corpus/build.py` first."
        )
    out = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if source and rec["source"] != source:
            continue
        rec["log"] = (ROOT / rec["log_path"]).read_text()
        out.append(rec)
    return out
