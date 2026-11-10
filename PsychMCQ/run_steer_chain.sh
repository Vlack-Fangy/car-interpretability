#!/bin/bash
# Waits for each model's probe to land, then steers it immediately -- probe is
# CPU-bound and steer is GPU-bound, so they overlap instead of running serially.
cd "$(dirname "$0")"
PY=${PY:-python}
for K in llama1b qwen3b llama8b; do
  while [ ! -f "artifacts/probe_${K}.json" ]; do sleep 30; done
  sleep 5
  echo "=== steering $K ==="
  CUDA_VISIBLE_DEVICES=1 $PY 07_steer.py --models $K --n 128 --alpha 1.0 --n-rand 3
done
echo "=== STEER CHAIN COMPLETE ==="
