#!/usr/bin/env python3
"""Fail a shape assertion -> AssertionError.

Pass --bare to print the assertion without the traceback header (some loops
log the message and exit non-zero); otherwise it propagates with a full
traceback. Both happen in the wild and they classify differently, which is the
whole point of having both in the corpus.
"""
import sys

from _common import banner

if __name__ == "__main__":
    banner("assertion")
    batch, expected = 30, 32
    if "--bare" in sys.argv:
        print(f"AssertionError: batch size {batch} != {expected}", file=sys.stderr)
        sys.exit(1)
    assert batch == expected, f"batch size {batch} != {expected}"
