#!/usr/bin/env python3
from _common import banner

if __name__ == "__main__":
    banner("permission")
    with open("/etc/gpualert_checkpoint.pt", "w") as f:
        f.write("ckpt")
