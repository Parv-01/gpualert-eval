#!/usr/bin/env python3
"""Dereference a null pointer via ctypes -> segmentation fault (core dumped)."""
import ctypes

from _common import banner

if __name__ == "__main__":
    banner("segfault")
    # read from address 0; the kernel sends SIGSEGV.
    ctypes.string_at(0)
    print("unreachable")
