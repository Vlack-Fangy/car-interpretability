#!/bin/bash
cd "$(dirname "$0")"
export DDSM_ROOT="${DDSM_ROOT:?Set DDSM_ROOT to your authorized dataset directory}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python 10_cnn_metrics.py \
  --n-train 3000 --n-test 2010 --n-pairs 96 --n-steer 128
echo "=== CNN METRICS COMPLETE ==="
