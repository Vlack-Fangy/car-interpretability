"""
Single source of truth. Nothing else hardcodes a path, a seed, or a
hyperparameter. If a number matters, it lives here.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
# Override per machine without editing this file:
#     export DDSM_ROOT=/path/to/ddsm
DATA_ROOT = Path(os.environ.get(
    "DDSM_ROOT", Path.home() / "Desktop/CAR/data/ddsm_sanchris"))
MANIFEST = ROOT / "manifest.csv"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
for _d in (CKPT_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(exist_ok=True)

# ----------------------------------------------------------------- seed
SEED = 1337                      # must match build_split.py

# ------------------------------------------------------------ data contract
# Models consume (N, 1, 227, 227) float32 in [0,1].
# 227 is native to this dataset -- no resize, no interpolation artefacts.
# All three backbones end in adaptive pooling so they accept it directly.
IMG_SIZE = 227
IN_CHANNELS = 1                  # what Quantus sees
N_FEATURES = IMG_SIZE * IMG_SIZE * IN_CHANNELS      # 51,529

# Applied INSIDE the model, after grayscale->3ch replication.
# Quantus never sees normalized space, so perturb_baseline="black" == 0.0
# is genuinely black.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LABEL_MAP = {0: "benign", 1: "malignant"}

# ------------------------------------------------------------- training
# IDENTICAL across all three models. Only LR may vary, and only within
# the documented sweep grid. Anything else confounds the comparison.
TRAIN = dict(
    epochs=20,                   # revisit after dihedral_check sets effective N
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    optimizer="adamw",
    scheduler="cosine",
    warmup_epochs=1,
    label_smoothing=0.0,
    early_stopping=False,        # fixed budget -> identical protocol
    num_workers=4,
    amp=False,        # AMP adds nondeterminism; reproducibility wins here
)
LR_SWEEP_GRID = (3e-5, 1e-4, 3e-4)

# Augmentation: geometric only. No intensity ops -- they would interact
# with the [0,1] contract and with black-baseline perturbation.
AUGMENT = dict(hflip=0.5, vflip=0.5, rot90=False)   # rot90 OFF: dihedral_check
#   found crops are already rotated relative to each other (raw corr 0.19 ->
#   0.65 once rotation is allowed). Adding it re-samples existing variation.

# ------------------------------------------------------------- gates
GATES = dict(
    accuracy_floor=0.65,         # balanced acc; below this XAI numbers are noise
    accuracy_ceiling=0.90,       # above this, suspect leakage before celebrating
    saturation_p=0.999,          # max(softmax) above this counts as saturated
    saturation_max_frac=0.50,    # >50% saturated -> FaithCorr will NaN out
    determinism_atol=0.0,        # bitwise. no tolerance.
)

# ------------------------------------------------------------- XAI
XAI_METHODS = ("vanilla_gradients", "integrated_gradients")

IG_STEPS = 50
IG_BASELINE = "black"            # zeros in [0,1] space == empty breast

# One policy for every method, or the comparison is invalid.
ATTR_POST = dict(abs=True, normalise=True)
# NOTE: no channel-reduction step. The grayscale->3ch replication is linear,
# so d(loss)/dx for the 1-channel input is already the sum over the three
# replicated channels. Correct Jacobian, not a post-hoc aggregation choice.

# ------------------------------------------------------------- metrics
# Three different Y-sources on purpose, so the numbers are not redundant:
#   FaithfulnessCorrelation -> MODEL BEHAVIOR (output)
#   AvgSensitivity          -> INPUT VARIATION
#   Sparseness              -> SELF-REFERENTIAL
N_EVAL = 128                     # one crop per view-group, class-balanced

METRICS = dict(
    faithfulness_correlation=dict(
        nr_runs=100,             # 50 gives a noisy Pearson over 50 points
        subset_size=2576,        # ~5% of 51,529
        perturb_baseline="black",
        return_aggregate=False,
    ),
    avg_sensitivity=dict(
        nr_samples=10,
        lower_bound=0.1,         # L-inf radius in [0,1] space
        return_aggregate=False,
    ),
    sparseness=dict(
        return_aggregate=False,
    ),
)
FAITHCORR_SUBSET_SWEEP = (0.01, 0.05, 0.10)   # report which fraction was used

QUANTUS_COMMON = dict(
    channel_first=True,
    softmax=False,               # LOGITS. softmax saturates -> Pearson undefined
    device=None,                 # filled at runtime
)

# ------------------------------------------------------------- reporting
SHEET_COLUMNS = [
    "M1_task", "M2_dataset", "M3_model", "M4_xai_method",
    "M5_faithfulness_correlation",   # higher better
    "M6_avg_sensitivity",            # LOWER better -- never average with others
    "M7_sparseness",
    "M8_balanced_accuracy",
]
TASK_NAME = "Cancer Detection"
DATASET_NAME = "DDSM Breast Cancer (benign vs malignant masses)"