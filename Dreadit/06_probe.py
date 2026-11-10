#!/usr/bin/env python3
"""
06_probe.py

Layer-wise linear probing. Fit on TRAIN, evaluate on TEST -- no CV on the
reported split, so nothing is tuned on the numbers that get published.

Headline:
  acc(l)  logistic regression on the residual stream at layer l

Floors -- the headline means nothing without them. A probe scoring 0.80 is only
interesting relative to the strongest trivial explanation:
  layer0     probe on the embedding output = bag-of-words. If layer l does not
             beat this, contextualisation contributed nothing.
  randinit   same architecture, random weights. Isolates what PRETRAINING bought.
  tfidf      no model at all. The lexical floor.

  A1 = acc_peak - max(all floors)

Diagnostic (reported separately, NOT folded into A1):
  subreddit  random binary label assigned per subreddit, fixed across train/test.
             A probe can only learn it by encoding topic identity. Since subreddit
             correlates with the stress label in Dreaddit, this bounds how much of
             the headline could be topic leakage rather than stress.

Also reads the logit lens out of the cache and reports the crossover depth.

usage:
  python 06_probe.py
"""

import os

# Must run BEFORE torch. setdefault so a shell override still wins.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import argparse
import json
# import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C_GRID = [0.001, 0.01, 0.1, 1.0]


def fit_eval(Xtr, ytr, Xte, yte, seed=0):
    """Small C sweep on a held-out slice of TRAIN. Returns (acc, auroc, w)."""
    n = len(ytr)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * 0.85)
    tr, va = idx[:cut], idx[cut:]

    best, bestC = -1, C_GRID[0]
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit(Xtr[tr], ytr[tr])
        s = m.score(Xtr[va], ytr[va])
        if s > best:
            best, bestC = s, C
    m = LogisticRegression(C=bestC, max_iter=2000)
    m.fit(Xtr, ytr)
    acc = float(m.score(Xte, yte))
    auc = float(roc_auc_score(yte, m.decision_function(Xte)))
    w = m.coef_[0].astype(np.float32)
    return acc, auc, w / (np.linalg.norm(w) + 1e-9)


def sweep(tr, te, pool, seed=0):
    """Probe every layer. Returns accs, aurocs, directions [L+1, D]."""
    Xtr_all, Xte_all = tr[pool], te[pool]
    ytr, yte = tr["label"], te["label"]
    Lp1 = Xtr_all.shape[1]
    accs, aucs, ws = [], [], []
    for l in range(Lp1):
        a, u, w = fit_eval(Xtr_all[:, l].astype(np.float32), ytr,
                           Xte_all[:, l].astype(np.float32), yte, seed)
        accs.append(a)
        aucs.append(u)
        ws.append(w)
        print(f"    layer {l:3d}  acc {a:.3f}  auroc {u:.3f}", flush=True)
    return np.array(accs), np.array(aucs), np.stack(ws)


def subreddit_control(tr, te, pool, layer, seed=0):
    """Random but FIXED binary label per subreddit -- learnable iff topic is encoded."""
    subs = sorted(set(tr["subreddit"].tolist()) | set(te["subreddit"].tolist()))
    rng = np.random.default_rng(seed)
    lut = {s: int(v) for s, v in zip(subs, rng.integers(0, 2, len(subs)))}
    ytr = np.array([lut[s] for s in tr["subreddit"]])
    yte = np.array([lut[s] for s in te["subreddit"]])
    if len(set(yte.tolist())) < 2:
        return float("nan")
    a, _, _ = fit_eval(tr[pool][:, layer].astype(np.float32), ytr,
                       te[pool][:, layer].astype(np.float32), yte, seed)
    return a


def tfidf_floor(tr, te):
    v = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Xtr = v.fit_transform(tr["text"].tolist())
    Xte = v.transform(te["text"].tolist())
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(Xtr, tr["label"])
    return float(m.score(Xte, te["label"]))


def run(key, args):
    print(f"\n{'=' * 70}\n{key}\n{'=' * 70}", flush=True)
    tr = np.load(os.path.join(args.out, f"cache_{key}_train.npz"), allow_pickle=True)
    te = np.load(os.path.join(args.out, f"cache_{key}_test.npz"), allow_pickle=True)

    print(f"  probing ({args.pool}-pooled)")
    accs, aucs, ws = sweep(tr, te, args.pool, args.seed)
    L = len(accs) - 1
    peak = int(np.argmax(accs))

    floors = {"layer0": float(accs[0])}

    ri_path = os.path.join(args.out, f"cache_{key}_randinit_test.npz")
    if os.path.exists(ri_path):
        print("  random-init floor")
        ri_tr = np.load(os.path.join(args.out, f"cache_{key}_randinit_train.npz"),
                        allow_pickle=True)
        ri_te = np.load(ri_path, allow_pickle=True)
        ri, _, _ = sweep(ri_tr, ri_te, args.pool, args.seed)
        floors["randinit"] = float(ri.max())
    else:
        print("  (no random-init cache -- rerun 04_cache.py --randinit for that floor)")

    print("  tfidf floor")
    floors["tfidf"] = tfidf_floor(tr, te)

    sub = subreddit_control(tr, te, args.pool, peak, args.seed)

    # logit lens: depth at which the running LD crosses zero / peaks
    lens = te["lens"].mean(0)
    lens_auroc = np.array([roc_auc_score(te["label"], te["lens"][:, l])
                           for l in range(lens.shape[0])])

    res = {
        "n_layers": L,
        "pool": args.pool,
        "probe_acc_by_layer": accs.tolist(),
        "probe_auroc_by_layer": aucs.tolist(),
        "probe_peak_acc": float(accs[peak]),
        "probe_peak_layer": peak,
        "probe_peak_frac": float(peak / L),
        "floors": floors,
        "A1_readability": float(accs[peak] - max(floors.values())),
        "subreddit_control_acc": sub,
        "lens_ld_by_layer": lens.tolist(),
        "lens_auroc_by_layer": lens_auroc.tolist(),
        "lens_peak_layer": int(np.argmax(lens_auroc)),
        "lens_peak_frac": float(np.argmax(lens_auroc) / L),
    }
    np.save(os.path.join(args.out, f"probe_dirs_{key}.npy"), ws)
    with open(os.path.join(args.out, f"probe_{key}.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n  peak acc {accs[peak]:.3f} @ layer {peak}/{L} ({peak / L:.2f})")
    for k_, v in floors.items():
        print(f"  floor {k_:10s} {v:.3f}")
    print(f"  A1 readability        {res['A1_readability']:+.3f}")
    print(f"  subreddit control     {sub:.3f}   (topic leakage bound)")
    print(f"  lens peak auroc layer {res['lens_peak_layer']}/{L}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["llama1b", "qwen3b", "llama8b"])
    ap.add_argument("--pool", default="mean", choices=["mean", "last"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    table = {k: run(k, args) for k in args.models}

    print("\n" + "=" * 96)
    print(f"{'model':10} {'L':>3} {'peak':>6} {'l/L':>6} {'lay0':>6} {'rand':>6} "
          f"{'tfidf':>6} {'A1':>7} {'subrdt':>7} {'lens l/L':>9}")
    print("-" * 96)
    for k, r in table.items():
        f_ = r["floors"]
        print(f"{k:10} {r['n_layers']:3d} {r['probe_peak_acc']:6.3f} "
              f"{r['probe_peak_frac']:6.2f} {f_['layer0']:6.3f} "
              f"{f_.get('randinit', float('nan')):6.3f} {f_['tfidf']:6.3f} "
              f"{r['A1_readability']:+7.3f} {r['subreddit_control_acc']:7.3f} "
              f"{r['lens_peak_frac']:9.2f}")
    print("=" * 96)


if __name__ == "__main__":
    main()