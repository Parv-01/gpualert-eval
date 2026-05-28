#!/usr/bin/env python3
"""Raise an exception with no dedicated rule -> generic Python traceback.

ValueError/KeyError/TypeError aren't in the classifier's specific rules, so the
expected label is the catch-all 'Python exception (traceback)'.
"""
from _common import banner

if __name__ == "__main__":
    banner("traceback")
    cfg = {"lr": 3e-4}
    # KeyError: a config key the code assumed was present.
    warmup = cfg["warmup_steps"]
    print(warmup)
