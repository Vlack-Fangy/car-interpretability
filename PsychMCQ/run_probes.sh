#!/bin/bash
# One probe process per model, in parallel. The probe is CPU-bound and the models
# are independent, so running them concurrently cuts wall clock from the SUM of
# the three (~5h, llama8b dominating at 33 layers x 4096 dims x 23k rows) to the
# MAX of the three. The box has 128 cores and 1.2 TB RAM; peak is ~30 GB each.
cd "$(dirname "$0")"
PY=${PY:-python}
for K in llama1b qwen3b llama8b; do
  OMP_NUM_THREADS=32 $PY 06_probe.py --pool mean --models $K \
      > artifacts/log_probe_$K.txt 2>&1 &
done
wait
echo "=== ALL PROBES COMPLETE ==="
