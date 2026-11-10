# CAR Interpretability Experiments

Task-specific research code for examining where models encode information and
whether interventions change their decisions. The experiments combine task
checks, hidden-state probes, activation patching, steering and control baselines.
This is the original experiment collection; the newer configuration-driven
[TrustScore](https://github.com/TRUSTMAPS/trustscore) pipeline is a separate project.

## Tasks

| Folder | Experiment |
|---|---|
| `Dreadit` | Stress detection |
| `PsychMCQ` | Synthesized two-option psychology questions |
| `MedQA` | Medical questions reduced to paired two-option choices |
| `MedLFQA` | Long-form medical QA-derived experiment |
| `MedHallu` | Grounded versus hallucinated answer experiment |
| `Suicide` | Suicidal-ideation dataset classification |
| `DummyCNNBreast` | Breast-image CNN controls and attribution diagnostics |

These are research tasks, not clinical decision tools. Folder names retain the
original experiment naming. Dataset scripts document their transformations.

## Setup

Use Python 3.11 and install a PyTorch build appropriate to your CPU/GPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MODEL_ROOT="$HOME/models"
```

Model IDs are in each task's `00_gate.py`. Local checkpoints are preferred;
some loaders fall back to Hugging Face when a checkpoint is absent. Obtain model
access and dataset permissions separately. No checkpoints, dataset text, images,
activation caches or historical result files are bundled.

## Start with one task

For MedQA, prepare and audit the derived dataset before any model experiment:

```bash
cd MedQA
python 01_data.py --help
python 01_data.py --build
python 01_data.py --audit
CUDA_VISIBLE_DEVICES=0 python 00_gate.py --models llama1b --n-eval 5
```

Use an allocated GPU. Data preparation may download the full source dataset.
Inspect the small task gate before running the remaining stages. The numbered
scripts expose their own `--help`:

1. `00_gate.py`: task performance, answer-token resolution and intervention viability.
2. `04_cache.py`: residual representations and random-initialization controls.
3. `05_patch.py`: activation replacement experiments.
4. `06_probe.py`: linear probes and control baselines.
5. `07_steer.py`: interventions along probe and random directions.
6. `08_report.py`: local report assembly.

Not every task has `01_data.py`; Dreaddit loads its source from `00_gate.py`.
Scripts depend on artifacts from earlier stages. `run_task.sh` is a full-run
convenience script, not a five-example pilot. Inspect its settings before use.
Medical MCQ reports used final-token pooling; do not use a mean-pooling launcher
as an exact reproduction of that protocol. CNN scripts are a separate workflow
and require `DDSM_ROOT` plus their own training/checkpoint setup.

## Checks and interpretation

```bash
python -m compileall -q .
bash -n run_task.sh
```

Release validation checks source syntax and selected CLI entry points; no GPU
experiments were rerun for this release. A readable probe direction is not proof
of causal use. Patching and steering require a meaningful task readout. Historical
probe selection differs from TrustScore's later training-only selection protocol.
CNN controls found strong border artifacts in the original crops, so high scores
do not establish lesion reasoning. Do not combine these measures into a universal
trust score or compare protocols without checking prompts, splits and controls.

## License

Original code is MIT licensed. Third-party datasets, model weights and libraries
retain their own terms. Dataset identifiers are recorded in each task's source;
access and redistribution rights must be checked before downloading or sharing
artifacts. Generated `data/`, `artifacts/`, checkpoints and results are ignored.
