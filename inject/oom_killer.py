#!/usr/bin/env python3
"""Get killed by the kernel OOM-killer under a tight cgroup memory cap.

Run it constrained so the kernel reaps the process instead of Python raising:
    systemd-run --scope -p MemoryMax=500M python inject/oom_killer.py
The process dies with SIGKILL; the give-away is 'Killed' on the shell and an
'Out of memory: Killed process' line in dmesg/journal that gets tee'd in.
"""
from _common import banner

if __name__ == "__main__":
    banner("oom_killer")
    chunks = []
    while True:
        # touch the pages so they're resident and count against the cgroup.
        chunks.append(bytearray(64 * 1024 * 1024))
        for b in chunks[-1:]:
            b[::4096] = b"\x01" * (len(b) // 4096 + (1 if len(b) % 4096 else 0))
