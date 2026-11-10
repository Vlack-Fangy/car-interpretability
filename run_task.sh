#!/bin/bash
# run_task.sh <TASK> <GPU>
# Full chain for one sub-project. Test splits here are 200 rows (100 paired items),
# so 05_patch is capped at 96/64 pairs -- build_pairs_matched cannot exceed the
# item count, and asking for 256 would silently return short.
set -e
T=$1; G=$2
cd "$(dirname "$0")/$T"
PY=${PY:-python}
echo "########## $T on GPU $G ##########"
CUDA_VISIBLE_DEVICES=$G $PY 00_gate.py  --bs 16
CUDA_VISIBLE_DEVICES=$G $PY 04_cache.py --randinit --bs 8
CUDA_VISIBLE_DEVICES=$G $PY 05_patch.py --bs 8 --n-pairs 96 --n-pairs-l1 64
$PY 06_probe.py --pool mean
CUDA_VISIBLE_DEVICES=$G $PY 07_steer.py --n 128 --alpha 1.0 --n-rand 3
$PY 08_report.py
echo "########## $T COMPLETE ##########"
