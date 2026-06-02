#!/usr/bin/env python3
from _common import banner

if __name__ == "__main__":
    banner("file_not_found")

    with open("/data/imagenet/train_manifest.json") as f:
        f.read()
