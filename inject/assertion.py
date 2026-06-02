#!/usr/bin/env python3
import sys

from _common import banner

if __name__ == "__main__":
    banner("assertion")
    batch, expected = 30, 32
    if "--bare" in sys.argv:
        print(f"AssertionError: batch size {batch} != {expected}", file=sys.stderr)
        sys.exit(1)
    assert batch == expected, f"batch size {batch} != {expected}"
