#!/usr/bin/env python3
from _common import banner

if __name__ == "__main__":
    banner("missing_module")
    import flash_attn_3
    print(flash_attn_3.__version__)
