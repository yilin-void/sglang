#!/bin/bash
# Multi-GPU driver: convert all PARTS shards across NG GPUs. Runs inside container.
# Usage: convert_all.sh <SCRIPT.py> <MODEL_REAL> <OUT_REAL> <PARTS> <NG> <LOGDIR>
set -uo pipefail
SCRIPT=$1; MODEL=$2; OUT=$3; PARTS=${4:-94}; NG=${5:-8}; LOGDIR=${6:-$OUT/../logs}
mkdir -p "$OUT" "$LOGDIR"
CONV=$(cd "$(dirname "$SCRIPT")" && pwd); BASE=$(basename "$SCRIPT")
cd "$CONV"
fail=0
r=0
while [ "$r" -lt "$PARTS" ]; do
  pids=()
  for g in $(seq 0 $((NG-1))); do
    [ "$r" -ge "$PARTS" ] && break
    CUDA_VISIBLE_DEVICES=$g python3 "$BASE" --model-dir "$MODEL" --output-dir "$OUT" \
      --parts "$PARTS" --rank "$r" > "$LOGDIR/rank_$(printf %03d $r).log" 2>&1 &
    pids+=("$!:$r")
    r=$((r+1))
  done
  for pr in "${pids[@]}"; do
    p=${pr%:*}; rr=${pr#*:}
    if ! wait "$p"; then echo "RANK $rr FAILED (see $LOGDIR/rank_$(printf %03d $rr).log)"; fail=1; fi
  done
done
if [ "$fail" -eq 0 ]; then echo "ALL_RANKS_DONE_OK"; else echo "SOME_RANKS_FAILED"; fi
exit $fail
