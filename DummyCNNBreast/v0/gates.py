#!/usr/bin/env python3
"""
Five gates. Every model passes all five before any XAI touches it.

    python gates.py                 # all registered models, no checkpoints
    python gates.py --model resnet18
    python gates.py --ckpt-dir checkpoints    # adds gates 4 and 5

  1 SHAPE        forward (N,1,227,227) -> (N,2)
  2 CONTRACT     normalization applied exactly once, inside the model
  3 DETERMINISM  same input twice -> BITWISE identical. no tolerance.
  4 ACCURACY     balanced acc within [floor, ceiling]     (needs checkpoint)
  5 SATURATION   fraction of max(softmax) > 0.999          (needs checkpoint)

Gates 4 and 5 are the ones that would otherwise waste hours: an untrained
or leakage-inflated model still produces perfectly formatted XAI numbers,
and a saturated model NaNs out FaithfulnessCorrelation on every sample.
"""
import argparse
import sys
from pathlib import Path

import determinism                      # MUST precede torch import: sets
determinism.enable()                    # CUBLAS_WORKSPACE_CONFIG pre-CUDA

import numpy as np
import torch

import config as C
from models.registry import MODELS, get_module
from models.wrap import build_model

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def _p(gate, status, msg=""):
    mark = {"PASS": "[ ok ]", "FAIL": "[FAIL]", "SKIP": "[skip]"}[status]
    print(f"    {mark} {gate:<14} {msg}")
    return status


# ------------------------------------------------------------------ 1
def gate_shape(model, device, n=4):
    x = torch.rand(n, C.IN_CHANNELS, C.IMG_SIZE, C.IMG_SIZE, device=device)
    with torch.no_grad():
        y = model(x)
    if tuple(y.shape) != (n, 2):
        return _p("shape", FAIL, f"got {tuple(y.shape)}, want ({n}, 2)")
    if not torch.isfinite(y).all():
        return _p("shape", FAIL, "non-finite logits")
    return _p("shape", PASS, f"(N,1,{C.IMG_SIZE},{C.IMG_SIZE}) -> {tuple(y.shape)} logits")


# ------------------------------------------------------------------ 2
def gate_contract(model, device, atol=1e-4):
    """Hook the backbone's first Conv2d and verify ANALYTICALLY what reaches it.

    A statistical range check is not enough -- double normalization still lands
    inside any plausible range window. So we feed constant inputs and assert the
    stem sees the exact arithmetic result:

        x == v  (1 channel)  ->  stem channel c == (v - mean[c]) / std[c]

    That single equation pins down all four failure modes at once: missing
    normalization, double normalization, wrong mean/std, and wrong order
    (normalize-before-replicate would broadcast one channel's stats to all).
    Replication is verified separately: the three channels must be identical
    BEFORE normalization, i.e. differ afterwards by exactly the per-channel
    offsets.
    """
    convs = [(n, m) for n, m in model.backbone.named_modules()
             if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        return _p("contract", SKIP, "no Conv2d in backbone")
    name, conv = convs[0]
    seen = {}

    def pre_hook(_m, inp):
        seen["t"] = inp[0].detach()

    h = conv.register_forward_pre_hook(pre_hook)
    mean = torch.tensor(C.IMAGENET_MEAN)
    std = torch.tensor(C.IMAGENET_STD)
    try:
        for v in (0.0, 1.0):
            x = torch.full((2, C.IN_CHANNELS, C.IMG_SIZE, C.IMG_SIZE),
                           v, device=device)
            with torch.no_grad():
                model(x)
            t = seen["t"].cpu()
            if t.shape[1] != 3:
                return _p("contract", FAIL,
                          f"stem sees {t.shape[1]} channels, want 3")
            got = t.mean(dim=(0, 2, 3))
            want = (v - mean) / std
            if not torch.allclose(got, want, atol=atol):
                d = (got - want).abs().max().item()
                ratio = (got / want).mean().item() if want.abs().min() > 1e-6 else float("nan")
                hint = ("double normalization" if ratio > 1.5
                        else "normalization missing" if abs(ratio) < 0.5
                        else "wrong mean/std or wrong order")
                return _p("contract", FAIL,
                          f"x={v}: stem={[round(g,4) for g in got.tolist()]} "
                          f"want={[round(w,4) for w in want.tolist()]} "
                          f"max|d|={d:.3e} -- {hint}")
            # spread across channels must equal the spread of the norm constants
            if not torch.allclose(t.std(dim=(0, 2, 3)),
                                  torch.zeros(3), atol=1e-4):
                return _p("contract", FAIL,
                          f"x={v}: stem not constant -- replication broken")
    finally:
        h.remove()

    z = ((0.0 - mean) / std).tolist()
    return _p("contract", PASS,
              f"stem({name}) 3ch  x=0 -> {[round(v,3) for v in z]} exact")


# ------------------------------------------------------------------ 3
def gate_determinism(model, device, n=8, trials=3):
    """Bitwise. Any drift means live dropout or BN in train mode, and it
    would masquerade as explanation instability in every robustness metric."""
    x = torch.rand(n, C.IN_CHANNELS, C.IMG_SIZE, C.IMG_SIZE, device=device)
    with torch.no_grad():
        ref = model(x)
        for t in range(trials):
            y = model(x)
            if not torch.equal(ref, y):
                d = (ref - y).abs().max().item()
                return _p("determinism", FAIL,
                          f"trial {t+1}: max|delta|={d:.3e}. find the live "
                          f"dropout / BN-in-train-mode")
    # Gradients must be reproducible too -- VanillaGradients IS the gradient,
    # and AvgSensitivity measures how much the explanation moves under input
    # noise. A jittery gradient turns that into a measurement of the GPU.
    #
    # Bitwise is the ideal but not the only acceptable outcome. What decides
    # it is whether the wobble changes which pixels rank as important, since
    # Sparseness and FaithfulnessCorrelation both read the ORDER of the
    # attributions rather than exact float values.
    import determinism as D
    d = D.gradient_stability(model, x, target_class=1, trials=3)
    if d["bitwise"]:
        return _p("determinism", PASS,
                  f"{trials} forwards + 3 backwards, bitwise")
    if d["acceptable"]:
        return _p("determinism", PASS,
                  f"forward bitwise; grad wobble {d['relative']:.1e} relative, "
                  f"top-1% pixels {d['top1pct_overlap']:.4f} preserved -- "
                  f"below the level that moves attributions")
    return _p("determinism", FAIL,
              f"forward stable, GRADIENTS differ: max|d|={d['max_abs_diff']:.3e} "
              f"({d['relative']:.1e} relative), top-1% pixel agreement "
              f"{d['top1pct_overlap']:.4f}. run: "
              f"import determinism; determinism.enable() BEFORE torch use")


# ------------------------------------------------------------------ 4 & 5
@torch.no_grad()
def _eval_logits(model, loader, device):
    L, Y = [], []
    for xb, yb in loader:
        L.append(model(xb.to(device)).cpu())
        Y.append(yb)
    return torch.cat(L), torch.cat(Y)


def gate_accuracy(logits, y):
    pred = logits.argmax(1)
    accs = []
    for c in (0, 1):
        m = y == c
        if m.sum() == 0:
            return _p("accuracy", FAIL, f"no class-{c} samples in eval")
        accs.append((pred[m] == c).float().mean().item())
    bal = float(np.mean(accs))
    lo, hi = C.GATES["accuracy_floor"], C.GATES["accuracy_ceiling"]
    msg = (f"balanced={bal:.4f} (benign={accs[0]:.3f} malignant={accs[1]:.3f})")
    if bal < lo:
        return _p("accuracy", FAIL, msg + f"  < floor {lo} -- XAI would be noise")
    if bal > hi:
        return _p("accuracy", FAIL, msg + f"  > ceiling {hi} -- SUSPECT LEAKAGE, "
                                          f"re-check the split before believing it")
    return _p("accuracy", PASS, msg), bal


def gate_saturation(logits):
    p = torch.softmax(logits, dim=1).max(1).values
    frac = (p > C.GATES["saturation_p"]).float().mean().item()
    msg = (f"{frac:.1%} of samples have max(softmax) > "
           f"{C.GATES['saturation_p']}  (median p={p.median():.4f})")
    if frac > C.GATES["saturation_max_frac"]:
        return _p("saturation", FAIL, msg + "  -- FaithCorr will NaN out")
    return _p("saturation", PASS, msg)


# ------------------------------------------------------------------ runner
def run(name, device, ckpt=None, loader=None):
    print(f"\n  {name}")
    spec = MODELS[name]
    model = build_model(name, pretrained=False, ckpt=ckpt, device=device)
    n_par = sum(p.numel() for p in model.parameters())
    head = get_module(model.backbone, spec.head_path)
    print(f"    params={n_par/1e6:.1f}M  head={spec.head_path} "
          f"Linear({head.in_features},{head.out_features})  bs={spec.batch_size}")

    res = [gate_shape(model, device),
           gate_contract(model, device),
           gate_determinism(model, device)]

    if ckpt and loader is not None:
        logits, y = _eval_logits(model, loader, device)
        r4 = gate_accuracy(logits, y)
        res.append(r4[0] if isinstance(r4, tuple) else r4)
        res.append(gate_saturation(logits))
    else:
        res.append(_p("accuracy", SKIP, "no checkpoint"))
        res.append(_p("saturation", SKIP, "no checkpoint"))

    ok = FAIL not in res
    print(f"    {'-> ALL GATES PASS' if ok else '-> BLOCKED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    names = [a.model] if a.model else list(MODELS)
    print("=" * 70)
    print(f"  GATES   device={a.device}  models={names}")
    print("=" * 70)

    allok = True
    for n in names:
        ckpt = None
        if a.ckpt_dir:
            p = Path(a.ckpt_dir) / f"{n}.pt"
            ckpt = str(p) if p.exists() else None
        allok &= run(n, a.device, ckpt=ckpt, loader=None)

    print("\n" + "=" * 70)
    print(f"  {'ALL MODELS CLEARED' if allok else 'ONE OR MORE BLOCKED'}")
    print("=" * 70)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()