#!/usr/bin/env python3
"""
10_cnn_metrics.py

Ports the four interpretability metrics from the LLM arm (CAR/Dreadit and the
task folders beside it) to the DDSM CNNs, and writes artifacts with the SAME JSON
schema, so the root results_table.py reads them with no changes.

    Faithfulness - Probe Readability   A1 = probe peak acc - max(floors)
    Sensitivity  - Steering Causality  A2 = 1 - dLD_random / dLD_probe
    Sparseness   - Patch Localization  A3 = 1 - norm. entropy of d(recovery)
    Probe AUROC                        best tap, held-out eval split

TRANSLATION FROM THE TRANSFORMER VERSION
  transformer layer      -> conv stage ("tap"); 9 taps per model so depth is
                            comparable across the three backbones
  residual stream at t   -> the tap's feature map, global-average-pooled over
                            space to a channel vector (the CNN's only
                            position-free summary)
  logit diff (yes-no)    -> LD = logit(malignant) - logit(benign)
  final-position edit    -> channel-space edit broadcast over all spatial cells
  clean/corrupt prompts  -> clean = a malignant crop, corrupt = a benign crop

Channel width differs per tap (64...512), unlike a transformer's constant d_model,
so features are kept as a per-tap list rather than one [N, L, D] array.

READ 09_pixel_floor.py FIRST. On this dataset a logistic regression on a 4-pixel
border ring -- with the lesion region zeroed -- separates benign from malignant at
AUROC 0.998, against 0.750 for the lesion centre. The three CNNs sit at balanced
accuracy 1.000 with 96-99% saturation because they learned that framing artefact.
Everything this script computes is therefore a faithful description of a model
that solves the task by a shortcut, NOT evidence about lesion reasoning. The
`pixel` floor below is what makes that visible in A1: it is the model-free
baseline, and a representation that cannot beat it has added nothing.

usage:
  python 10_cnn_metrics.py                       # all three models
  python 10_cnn_metrics.py --models resnet18
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "v0"))

import config as C                                    # noqa: E402
from data_loader import load_manifest                 # noqa: E402
from models.registry import get_module                # noqa: E402
from models.wrap import build_model                   # noqa: E402

# Nine taps per model: the stem output plus each subsequent stage. Chosen so the
# three backbones have the same number of depth samples and l/L is comparable.
TAPS = {
    "resnet18": ["maxpool", "layer1.0", "layer1.1", "layer2.0", "layer2.1",
                 "layer3.0", "layer3.1", "layer4.0", "layer4.1"],
    "densenet121": ["features.pool0", "features.denseblock1",
                    "features.transition1", "features.denseblock2",
                    "features.transition2", "features.denseblock3",
                    "features.transition3", "features.denseblock4",
                    "features.norm5"],
    "efficientnet_b0": [f"features.{i}" for i in range(9)],
}
C_GRID = [0.001, 0.01, 0.1, 1.0]


# ----------------------------------------------------------------- data

def load_images(df, n, seed=C.SEED, size=C.IMG_SIZE):
    from PIL import Image
    rng = np.random.default_rng(seed)
    if n and n < len(df):
        df = df.iloc[sorted(rng.choice(len(df), n, replace=False))]
    X = np.zeros((len(df), 1, size, size), dtype=np.float32)
    for i, p in enumerate(df["path"]):
        im = Image.open(p).convert("L")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        X[i, 0] = np.asarray(im, dtype=np.float32) / 255.0
    return X, df["label"].to_numpy().astype(int)


# -------------------------------------------------------------- forward

@torch.no_grad()
def forward_feats(model, taps, X, dev, bs=32):
    """LD per image + GAP'd feature vector at every tap."""
    buf, feats = {}, [[] for _ in taps]
    hooks = [get_module(model.backbone, p).register_forward_hook(
        (lambda i: (lambda m, a, o: buf.__setitem__(i, o)))(i))
        for i, p in enumerate(taps)]
    lds = []
    try:
        for b in range(0, len(X), bs):
            xb = torch.from_numpy(X[b:b + bs]).to(dev)
            buf.clear()
            lg = model(xb).float()
            lds.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
            for i in range(len(taps)):
                h = buf[i]
                h = h[0] if isinstance(h, tuple) else h
                feats[i].append(h.float().mean(dim=(2, 3)).cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
    return np.concatenate(lds), [np.concatenate(f) for f in feats]


# ---------------------------------------------------------------- probe

def fit_eval(Xtr, ytr, Xte, yte, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    cut = int(len(ytr) * 0.85)
    tr, va = idx[:cut], idx[cut:]
    best, bestC = -1, C_GRID[0]
    for c in C_GRID:
        m = LogisticRegression(C=c, max_iter=3000).fit(Xtr[tr], ytr[tr])
        s = m.score(Xtr[va], ytr[va])
        if s > best:
            best, bestC = s, c
    m = LogisticRegression(C=bestC, max_iter=3000).fit(Xtr, ytr)
    w = m.coef_[0].astype(np.float32)
    return (float(m.score(Xte, yte)),
            float(roc_auc_score(yte, m.decision_function(Xte))),
            w / (np.linalg.norm(w) + 1e-9))


def pixel_floor(Xtr, ytr, Xte, yte, size=32):
    """Model-free baseline -- the CNN analogue of the LLM arm's TF-IDF floor."""
    import torch.nn.functional as F
    ds = lambda X: F.adaptive_avg_pool2d(
        torch.from_numpy(X), size).numpy().reshape(len(X), -1)
    a, _, _ = fit_eval(ds(Xtr), ytr, ds(Xte), yte)
    return a


# --------------------------------------------------------------- patch

def make_pairs(ld, y, sd, n_pairs, seed=0):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    cand = [(int(c), int(k)) for c in rng.permutation(pos)[:200]
            for k in rng.permutation(neg)[:200]]
    rng.shuffle(cand)
    out = [(c, k) for c, k in cand if ld[c] - ld[k] > sd]
    return out[:n_pairs]


def spatial_mask(shape, region, frac=0.25):
    """
    Boolean [H, W] mask covering `frac` of the map's AREA.

    WHY A SUBREGION AT ALL. In a transformer, L0 patching replaces ONE token
    position and leaves the other S-1 intact, so information can still route
    around the patch and recovery genuinely varies with depth. A CNN has no
    residual stream carrying the input forward: everything after a tap is a
    deterministic function of that tap's feature map alone. Replacing the WHOLE
    map is therefore a complete state swap, and recovery comes out at exactly
    1.000 at every tap with zero variance -- measured, not hypothesised. That
    makes A3 undefined (no positive increments to take an entropy over).

    Patching a fixed AREA FRACTION restores the transformer's situation: the
    unpatched cells still carry the corrupt image, so recovery rises with depth
    as receptive fields widen. The fraction is held constant across taps because
    the map shrinks with depth (57x57 -> 8x8 here); patching a fixed number of
    cells instead would confound depth with resolution.

    region="centre" is the lesion site -- the region a cancer model SHOULD use.
    region="border" is the framing ring that 09_pixel_floor.py shows is
    sufficient to separate the classes on its own.
    """
    H, W = shape
    side_h = max(1, int(round(H * frac ** 0.5)))
    side_w = max(1, int(round(W * frac ** 0.5)))
    m = np.zeros((H, W), dtype=bool)
    if region == "centre":
        t, l = (H - side_h) // 2, (W - side_w) // 2
        m[t:t + side_h, l:l + side_w] = True
    elif region == "border":
        # ring of equal area, taken from the outside in
        r = max(1, int(round((H - (H * H * (1 - frac)) ** 0.5) / 2)))
        m[:r, :] = m[-r:, :] = m[:, :r] = m[:, -r:] = True
    else:
        m[:] = True
    return m


@torch.no_grad()
def patch_curve(model, taps, X, pairs, dev, bs=16, region="centre", frac=0.25):
    """recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt) per tap."""
    L = len(taps)
    rec = np.zeros((L, len(pairs)))
    ident, readout = [], []
    for b in range(0, len(pairs), bs):
        ch = pairs[b:b + bs]
        xc = torch.from_numpy(X[[c for c, _ in ch]]).to(dev)
        xk = torch.from_numpy(X[[k for _, k in ch]]).to(dev)
        cache = {}
        hs = [get_module(model.backbone, p).register_forward_hook(
            (lambda i: (lambda m, a, o: cache.__setitem__(i, o.detach().clone())))(i))
            for i, p in enumerate(taps)]
        try:
            ldc = (lambda g: g[:, 1] - g[:, 0])(model(xc).float()).cpu().numpy()
            clean = {i: v for i, v in cache.items()}
            cache.clear()
            ldk = (lambda g: g[:, 1] - g[:, 0])(model(xk).float()).cpu().numpy()
            corrupt = {i: v for i, v in cache.items()}
        finally:
            for h in hs:
                h.remove()
        den = ldc - ldk

        def run_with(i, src, whole=False):
            def hook(m, a, o):
                cur = o[0] if isinstance(o, tuple) else o
                if whole:
                    return src
                msk = spatial_mask(cur.shape[-2:], region, frac)
                t = torch.from_numpy(msk).to(cur.device)
                out = cur.clone()
                out[:, :, t] = src[:, :, t]
                return out
            h = get_module(model.backbone, taps[i]).register_forward_hook(hook)
            try:
                g = model(xk).float()
            finally:
                h.remove()
            return ((g[:, 1] - g[:, 0]).cpu().numpy() - ldk) / den

        # CHECK 1 -- corrupt's own activations back in. must be 0.
        ident.append(run_with(L // 2, corrupt[L // 2], whole=True))
        for i in range(L):
            rec[i, b:b + len(ch)] = run_with(i, clean[i])
        # CHECK 2 -- the WHOLE last map feeds pool -> head directly, so it must
        # be 1. rec[L-1] is a subregion patch and is NOT expected to be 1.
        readout.append(run_with(L - 1, clean[L - 1], whole=True))
    return (rec, float(np.abs(np.concatenate(ident)).max()),
            float(np.abs(np.concatenate(readout) - 1.0).max()))


def localization(curve):
    d = np.clip(np.diff(np.clip(curve, 0, None)), 0, None)
    if d.sum() <= 0 or len(d) < 2:
        return 0.0
    p = d / d.sum()
    nz = p[p > 0]
    return float(1.0 - (-(nz * np.log(nz)).sum()) / np.log(len(d)))


# --------------------------------------------------------------- steer

@torch.no_grad()
def steer_curve(model, taps, X, dirs, dev, alpha=1.0, n_rand=3, n=128,
                bs=16, seed=0):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs = X[sel]
    L = len(taps)
    d_probe, d_rand = np.zeros(L), np.zeros(L)

    def ld_with(i, vec, sign):
        def hook(m, a, o):
            h = o[0] if isinstance(o, tuple) else o
            v = torch.as_tensor(vec, device=h.device, dtype=h.dtype)
            scale = h.float().mean(dim=(2, 3)).norm(dim=-1)
            return h + sign * alpha * scale.view(-1, 1, 1, 1) * v.view(1, -1, 1, 1)
        hk = get_module(model.backbone, taps[i]).register_forward_hook(hook)
        try:
            g = model(xb).float()
        finally:
            hk.remove()
        return (g[:, 1] - g[:, 0]).cpu().numpy()

    for b in range(0, len(Xs), bs):
        xb = torch.from_numpy(Xs[b:b + bs]).to(dev)
        for i in range(L):
            w = dirs[i] / (np.linalg.norm(dirs[i]) + 1e-9)
            if not np.isfinite(w).all() or np.linalg.norm(w) < 1e-8:
                continue          # degenerate probe direction -> zero effect
            d_probe[i] += np.abs(ld_with(i, w, +1) - ld_with(i, w, -1)).sum()
            acc = 0.0
            for _ in range(n_rand):
                r = rng.standard_normal(len(w)).astype(np.float32)
                r /= np.linalg.norm(r)
                acc += np.abs(ld_with(i, r, +1) - ld_with(i, r, -1)).sum()
            d_rand[i] += acc / n_rand
    return d_probe / len(Xs), d_rand / len(Xs)


# ---------------------------------------------------------------- main

def run(name, args, data):
    (Xtr, ytr, Xte, yte) = data
    taps = TAPS[name]
    dev = args.device
    print(f"\n{'=' * 66}\n{name}  ({len(taps)} taps)\n{'=' * 66}", flush=True)

    ck = os.path.join(C.CKPT_DIR, f"{name}_{args.tag}.pt")
    model = build_model(name, pretrained=False, ckpt=ck, device=dev).eval()

    ld_te, f_te = forward_feats(model, taps, Xte, dev, args.bs)
    ld_tr, f_tr = forward_feats(model, taps, Xtr, dev, args.bs)

    from sklearn.metrics import roc_auc_score
    gate_auc = float(roc_auc_score(yte, ld_te))
    print(f"  gate: LD auroc={gate_auc:.4f}  acc={(( ld_te > 0) == yte).mean():.4f}",
          flush=True)

    # ---- probe -----------------------------------------------------------
    accs, aucs, dirs = [], [], []
    for i in range(len(taps)):
        a, u, w = fit_eval(f_tr[i], ytr, f_te[i], yte, args.seed)
        accs.append(a); aucs.append(u); dirs.append(w)
        print(f"    tap {i} {taps[i]:22s} acc {a:.3f} auroc {u:.3f}", flush=True)
    peak = int(np.argmax(accs))

    floors = {"layer0": float(accs[0])}
    print("  pixel floor (model-free)", flush=True)
    floors["pixel"] = pixel_floor(Xtr, ytr, Xte, yte)
    print("  random-init floor", flush=True)
    ri = build_model(name, pretrained=False, device=dev).eval()
    _, rf_te = forward_feats(ri, taps, Xte, dev, args.bs)
    _, rf_tr = forward_feats(ri, taps, Xtr, dev, args.bs)
    floors["randinit"] = float(max(
        fit_eval(rf_tr[i], ytr, rf_te[i], yte, args.seed)[0]
        for i in range(len(taps))))
    del ri; torch.cuda.empty_cache()
    rng = np.random.default_rng(args.seed + 991)
    floors["shuffled"] = fit_eval(f_tr[peak], rng.permutation(ytr),
                                  f_te[peak], rng.permutation(yte),
                                  args.seed)[0]

    probe = {"n_layers": len(taps) - 1, "pool": "gap", "taps": taps,
             "n_train": int(len(ytr)), "n_test": int(len(yte)),
             "probe_acc_by_layer": accs, "probe_auroc_by_layer": aucs,
             "probe_peak_acc": float(accs[peak]), "probe_peak_layer": peak,
             "probe_peak_frac": float(peak / (len(taps) - 1)),
             "floors": floors,
             "A1_readability": float(accs[peak] - max(floors.values())),
             "subreddit_control_acc": float("nan"),
             "lens_peak_frac": float("nan")}
    for k_, v in floors.items():
        print(f"  floor {k_:9s} {v:.3f}")
    print(f"  A1 readability   {probe['A1_readability']:+.3f}")

    # ---- patch -----------------------------------------------------------
    pairs = make_pairs(ld_te, yte, ld_te.std(), args.n_pairs, args.seed)
    print(f"  patch: {len(pairs)} pairs", flush=True)
    rec, idc, rdo = patch_curve(model, taps, Xte, pairs, dev, args.bs,
                                region="centre", frac=args.patch_frac)
    rec_b, _, _ = patch_curve(model, taps, Xte, pairs, dev, args.bs,
                              region="border", frac=args.patch_frac)
    curve, curve_b = rec.mean(1), rec_b.mean(1)
    print(f"    CHECK identity {idc:.2e}   readout {rdo:.2e}   "
          f"-> {'PASS' if idc < 1e-2 and rdo < 1e-2 else '*** FAIL ***'}")
    patch = {"n_layers": len(taps) - 1, "l0_curve": curve.tolist(),
             "l0_sem": (rec.std(1) / np.sqrt(rec.shape[1])).tolist(),
             "l0_peak_layer": int(np.argmax(curve)),
             "l0_peak_frac": float(np.argmax(curve) / (len(taps) - 1)),
             "l0_peak_recovery": float(curve.max()),
             "localization": localization(curve),
             "check_identity": idc, "check_readout": rdo,
             "check_pass": bool(idc < 1e-2 and rdo < 1e-2),
             "n_pairs_l0": int(rec.shape[1]),
             "patch_region": "centre", "patch_frac": args.patch_frac,
             "border_curve": curve_b.tolist(),
             "border_localization": localization(curve_b),
             "border_peak_recovery": float(curve_b.max())}
    print(f"    A3 localization {patch['localization']:.3f}  "
          f"(centre peak rec {curve.max():.3f})")
    print(f"    border patch    A3 {patch['border_localization']:.3f}  "
          f"(border peak rec {curve_b.max():.3f})  <- shortcut diagnostic")

    # ---- steer -----------------------------------------------------------
    dp, dr = steer_curve(model, taps, Xte, dirs, dev, args.alpha,
                         args.n_rand, args.n_steer, args.bs, args.seed)
    a2 = np.clip(1.0 - dr / np.maximum(dp, 1e-9), 0, 1)
    a2[dp <= 0] = 0.0
    masked = np.where(dp >= 0.10 * dp.max(), a2, -1.0)
    pk = int(np.argmax(masked))
    steer = {"n_layers": len(taps) - 1, "alpha": args.alpha,
             "dLD_probe_by_layer": dp.tolist(),
             "dLD_random_by_layer": dr.tolist(),
             "dLD_ablate_by_layer": [float("nan")] * len(taps),
             "steer_peak_layer": pk,
             "dLD_probe_at_peak": float(dp[pk]),
             "dLD_random_at_peak": float(dr[pk]),
             "dLD_ablate_at_peak": float("nan"),
             "A2_causal": float(a2[pk]), "probe_peak_layer": peak}
    print(f"    A2 causal       {steer['A2_causal']:.3f}")

    gate = {"spec": {"key": name, "taps": taps},
            "token_ids": {"yes_form": "malignant", "no_form": "benign",
                          "yes_id": 1, "no_id": 0,
                          "top20_mean_logit_tokens": ["malignant", "benign"]},
            "stats": {"n": int(len(yte)), "n_layers": len(taps) - 1,
                      "d_model": int(f_te[-1].shape[1]),
                      "auroc_ld": gate_auc,
                      "accuracy": float(((ld_te > 0) == yte).mean()),
                      "post_span_ok": 1.0, "ld_std": float(ld_te.std()),
                      "n_usable_at_1sd": int(len(pairs))}}

    for obj, pat in ((gate, "gate"), (probe, "probe"), (patch, "patch"),
                     (steer, "steer")):
        with open(os.path.join(args.out, f"{pat}_{name}.json"), "w") as f:
            json.dump(obj, f, indent=2)
    np.save(os.path.join(args.out, f"probe_dirs_{name}.npy"),
            np.array(dirs, dtype=object), allow_pickle=True)
    del model; torch.cuda.empty_cache()
    return probe, patch, steer, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(TAPS))
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2010)
    ap.add_argument("--n-pairs", type=int, default=96)
    ap.add_argument("--patch-frac", type=float, default=0.25,
                    help="area fraction of the feature map replaced per patch")
    ap.add_argument("--n-steer", type=int, default=128)
    ap.add_argument("--n-rand", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_manifest()
    Xtr, ytr = load_images(df[df["split"] == "train"], args.n_train)
    Xte, yte = load_images(df[df["split"] == "eval"], args.n_test)
    print(f"train={len(ytr)} (bal {ytr.mean():.3f})  "
          f"eval={len(yte)} (bal {yte.mean():.3f})", flush=True)

    rows = {}
    for m in args.models:
        rows[m] = run(m, args, (Xtr, ytr, Xte, yte))

    print("\n" + "=" * 90)
    print(f"{'model':16} {'probeAUROC':>11} {'A1':>8} {'A2':>8} {'A3':>8} "
          f"{'gateAUROC':>10} {'ident':>8} {'readout':>8}")
    print("-" * 90)
    for m, (pr, pa, st, g) in rows.items():
        print(f"{m:16} {max(pr['probe_auroc_by_layer']):11.3f} "
              f"{pr['A1_readability']:+8.3f} {st['A2_causal']:8.3f} "
              f"{pa['localization']:8.3f} {g['stats']['auroc_ld']:10.3f} "
              f"{pa['check_identity']:8.1e} {pa['check_readout']:8.1e}")
    print("=" * 90)


if __name__ == "__main__":
    main()
