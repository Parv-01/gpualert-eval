#!/usr/bin/env python3
from _common import banner

if __name__ == "__main__":
    banner("ram_oom")

    buf = bytearray(10 * 1024 * 1024 * 1024 * 1024)
    print(len(buf))
