#!/usr/bin/env bash
# Run every injector on a real node and capture labelled logs.
#
# This is the "Hour 1-4 on the GPU node" step. It drives each injector through
# the gpualert wrapper so the captured logs are byte-for-byte what gpualert
# delivers. Logs land in corpus/injected/ and a manifest row is appended.
#
# On a node without CUDA the GPU-bound injectors exit(3) and are skipped; use
# `python corpus/build.py` instead to synthesise the full corpus from the
# recorded references in inject/reference/.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="$ROOT/corpus/injected"
mkdir -p "$OUT"
REPEAT="${REPEAT:-30}"   # samples per class

run() {  # mode  cmd...
  local mode="$1"; shift
  for i in $(seq 1 "$REPEAT"); do
    local id; id="$(printf '%s-injected-%04d' "$mode" "$i")"
    # gpualert writes a merged log; we copy it out under the sample id.
    gpualert run --quiet -- "$@" \
      > "$OUT/$id.log" 2>&1
    echo "{\"id\": \"$id\", \"mode\": \"$mode\", \"source\": \"injected\", \"log_path\": \"corpus/injected/$id.log\"}" \
      >> "$ROOT/corpus/labels.jsonl"
  done
}

cd "$HERE"
run cuda_oom        python cuda_oom.py
run cuda_runtime    python cuda_runtime.py
run ram_oom         python ram_oom.py
run segfault        python segfault.py
run file_not_found  python file_not_found.py
run permission      python permission_denied.py
run missing_module  python missing_module.py
run div_zero        python div_zero.py
run device_mismatch python device_mismatch.py
run nan_loss        python nan_loss.py
run traceback       python traceback_generic.py
run assertion       python assertion.py
run runtime_error   python runtime_error.py
# nccl + oom_killer need special launchers (torchrun / systemd-run); see their
# module docstrings. They are driven separately in the harness.
echo "done. logs in $OUT"
