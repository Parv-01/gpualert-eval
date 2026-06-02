#!/usr/bin/env python3
import sys

from _common import banner

if __name__ == "__main__":
    banner("runtime_error")
    msg = "CUDA capability sm_90 is not compatible with this build"
    if "--bare" in sys.argv:
        print(f"RuntimeError: {msg}", file=sys.stderr)
        sys.exit(1)
    raise RuntimeError(msg)
