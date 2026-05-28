#!/usr/bin/env python3
"""Ask Python for an absurd allocation -> MemoryError (host RAM)."""
from _common import banner

if __name__ == "__main__":
    banner("ram_oom")
    # bytearray of ~10 TB; the allocator refuses -> MemoryError.
    buf = bytearray(10 * 1024 * 1024 * 1024 * 1024)
    print(len(buf))
