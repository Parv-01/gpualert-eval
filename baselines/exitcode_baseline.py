from __future__ import annotations

NAME = "exitcode"

def classify(sample: dict) -> str:
    code = sample.get("true_exit_code", 0)
    if code == -11:
        return "segfault"
    if code == -9:
        return "oom_killer"

    return "generic"
