#!/usr/bin/env python3
import ctypes

from _common import banner

if __name__ == "__main__":
    banner("segfault")

    ctypes.string_at(0)
    print("unreachable")
