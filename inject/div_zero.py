#!/usr/bin/env python3
"""Divide by an empty-batch count -> ZeroDivisionError.

Mirrors the classic 'last batch had 0 valid samples' metric bug.
"""
from _common import banner

if __name__ == "__main__":
    banner("div_zero")
    correct, n_valid = 31, 0
    acc = correct / n_valid
    print(acc)
