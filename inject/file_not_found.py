#!/usr/bin/env python3
"""Open a data path that doesn't exist -> FileNotFoundError."""
from _common import banner

if __name__ == "__main__":
    banner("file_not_found")
    # a stand-in for the usual "wrong --data-dir" mistake.
    with open("/data/imagenet/train_manifest.json") as f:
        f.read()
