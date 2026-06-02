#!/usr/bin/env python3
from _common import banner

if __name__ == "__main__":
    banner("oom_killer")
    chunks = []
    while True:

        chunks.append(bytearray(64 * 1024 * 1024))
        for b in chunks[-1:]:
            b[::4096] = b"\x01" * (len(b) // 4096 + (1 if len(b) % 4096 else 0))
