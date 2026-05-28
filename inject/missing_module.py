#!/usr/bin/env python3
"""Import a package that isn't installed -> ModuleNotFoundError."""
from _common import banner

if __name__ == "__main__":
    banner("missing_module")
    import flash_attn_3  # not installed in this env
    print(flash_attn_3.__version__)
