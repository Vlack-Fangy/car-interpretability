#!/usr/bin/env python3
"""
09_pixel_floor.py

The precondition for every interpretability number on the DDSM CNNs.

All three checkpoints report balanced accuracy 1.000 with 96-99% of predictions
saturated. That is not a good result on a cancer-detection task -- it is the
signature of a shortcut. This script asks the only question that matters before
any attribution or probe is computed:

    can the LABEL be recovered from raw pixels WITHOUT the lesion?

Three logistic regressions on downsampled raw pixels:

    full    the whole downsampled crop
    border  a ring of the outermost RING px only, lesion region zeroed out
    centre  the central square only -- this is where the lesion actually is

If `border` scores near `full`, the crops are separable from framing artefacts
(padding, aspect, scanner bezel, segmentation crop geometry) rather than from
tissue. A CNN trained on them will learn that, will hit 100% accuracy, and every
saliency map / probe / patching curve computed on it will describe the artefact.

usage:
  python 09_pixel_floor.py
  python 09_pixel_floor.py --n 1500 --size 32 --ring 4
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "v0"))

import config as C                                    # noqa: E402
from data_loader import load_manifest                 # noqa: E402


def load_pixels(df, size, n, seed=C.SEED):
    from PIL import Image
    rng = np.random.default_rng(seed)
    if n and n < len(df):
        idx = rng.choice(len(df), n, replace=False)
        df = df.iloc[sorted(idx)]
    X, y = [], []
    for p, lab in zip(df["path"], df["label"]):
        im = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
        X.append(np.asarray(im, dtype=np.float32).ravel() / 255.0)
        y.append(int(lab))
    return np.stack(X), np.array(y)


def masks(size, ring):
    """border ring, and the complementary centre square."""
    b = np.zeros((size, size), dtype=bool)
    b[:ring, :] = b[-ring:, :] = b[:, :ring] = b[:, -ring:] = True
    return b.ravel(), (~b).ravel()


def fit(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    m = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, ytr)
    return (float(m.score(Xte, yte)),
            float(roc_auc_score(yte, m.decision_function(Xte))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="images per split")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--ring", type=int, default=4)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_manifest()
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "eval"]
    print(f"train={len(tr)} eval={len(te)}  -> sampling {a.n} per split "
          f"at {a.size}x{a.size}", flush=True)

    Xtr, ytr = load_pixels(tr, a.size, a.n)
    Xte, yte = load_pixels(te, a.size, a.n)
    bmask, cmask = masks(a.size, a.ring)

    res = {}
    for name, sl in (("full", slice(None)), ("border", bmask), ("centre", cmask)):
        acc, auc = fit(Xtr[:, sl], ytr, Xte[:, sl], yte)
        res[name] = {"acc": acc, "auroc": auc,
                     "n_features": int(Xtr[:, sl].shape[1])}
        print(f"  {name:7s} acc={acc:.3f}  auroc={auc:.3f}  "
              f"({res[name]['n_features']} px)", flush=True)

    res["verdict"] = (
        "SHORTCUT: the label is recoverable from the border alone, so the CNNs "
        "are separating framing artefacts rather than lesion tissue. Every "
        "downstream interpretability number describes that artefact."
        if res["border"]["auroc"] >= 0.90 else
        "OK: the border alone does not carry the label.")
    print("\n  " + res["verdict"])
    with open(os.path.join(a.out, "pixel_floor.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"  -> {a.out}/pixel_floor.json")


if __name__ == "__main__":
    main()
