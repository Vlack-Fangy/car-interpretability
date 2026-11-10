#!/usr/bin/env python3
"""
Compute the three XAI metrics for every (model, method) pair and emit the sheet.

    python run_metrics.py --tag probe
    python run_metrics.py --tag probe --models resnet18 --n 64

Explainers are implemented directly on torch autograd -- no captum dependency.
Both are deterministic given a fixed baseline, which is why GradientSHAP was
dropped: its random baselines would confound AvgSensitivity.

Output: results/metrics_long.csv (one row per model x method x metric)
        results/sheet.csv        (the M1..M8 table)
"""
import argparse
import json
import time
from pathlib import Path

import determinism
determinism.enable()

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import torch                                                # noqa: E402
import quantus                                              # noqa: E402

import config as C                                          # noqa: E402
from data_loader import load_batch, xai_eval_subset         # noqa: E402
from models.registry import MODELS                          # noqa: E402
from models.wrap import build_model                         # noqa: E402


# ------------------------------------------------------------- explainers
def _post(a: torch.Tensor) -> torch.Tensor:
    """ONE policy for every method, or the comparison is invalid.
    abs -> per-sample max-normalise. No channel reduction needed: the
    grayscale->3ch replication is linear, so the 1-channel gradient is already
    the sum over the replicated channels."""
    if C.ATTR_POST["abs"]:
        a = a.abs()
    if C.ATTR_POST["normalise"]:
        m = a.flatten(1).max(1).values.view(-1, 1, 1, 1).clamp_min(1e-12)
        a = a / m
    return a


def vanilla_gradients(model, x, y):
    x = x.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    out = model(x)
    sel = out.gather(1, y.view(-1, 1)).sum()
    g = torch.autograd.grad(sel, x)[0]
    return _post(g.detach())


def integrated_gradients(model, x, y, steps=C.IG_STEPS):
    """Riemann-trapezoid along a straight line from a black baseline.
    Baseline is fixed zeros in [0,1] space -- an empty breast, and the same
    value Quantus uses for perturb_baseline='black'."""
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for k in range(1, steps + 1):
        alpha = k / steps
        xi = (base + alpha * (x - base)).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        sel = model(xi).gather(1, y.view(-1, 1)).sum()
        total = total + torch.autograd.grad(sel, xi)[0].detach()
    return _post((x - base) * total / steps)


EXPLAINERS = {
    "vanilla_gradients": vanilla_gradients,
    "integrated_gradients": integrated_gradients,
}


def make_explain_func(method: str, device: str):
    """Quantus calls explain_func(model, inputs, targets, **kw) with numpy.
    Written once; every future method reuses it."""
    fn = EXPLAINERS[method]

    def explain_func(model, inputs, targets, **kwargs):
        x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        y = torch.as_tensor(targets, dtype=torch.long, device=device)
        return fn(model, x, y).cpu().numpy()

    return explain_func


# ---------------------------------------------------------------- metrics
def run_pair(model, x, y, a, method, device):
    """Three metrics, three different Y-sources -- deliberately not redundant.
    FaithCorr -> model behaviour. AvgSens -> input variation. Sparseness ->
    the map alone."""
    common = dict(channel_first=True, softmax=False, device=device)
    xn, yn, an = x.cpu().numpy(), y.cpu().numpy(), a.cpu().numpy()
    out = {}

    t = time.time()
    fc = quantus.FaithfulnessCorrelation(
        disable_warnings=True, **C.METRICS["faithfulness_correlation"]
    )(model=model, x_batch=xn, y_batch=yn, a_batch=an, **common)
    fc = np.asarray(fc, dtype=float).ravel()
    out["faithfulness_correlation"] = dict(
        mean=float(np.nanmean(fc)), std=float(np.nanstd(fc)),
        median=float(np.nanmedian(fc)),
        nan=int(np.isnan(fc).sum()), n=int(fc.size), secs=time.time() - t)

    t = time.time()
    sp = quantus.Sparseness(
        disable_warnings=True, **C.METRICS["sparseness"]
    )(model=model, x_batch=xn, y_batch=yn, a_batch=an, **common)
    sp = np.asarray(sp, dtype=float).ravel()
    out["sparseness"] = dict(
        mean=float(np.nanmean(sp)), std=float(np.nanstd(sp)),
        median=float(np.nanmedian(sp)),
        nan=int(np.isnan(sp).sum()), n=int(sp.size), secs=time.time() - t)

    t = time.time()
    av = quantus.AvgSensitivity(
        disable_warnings=True, **C.METRICS["avg_sensitivity"]
    )(model=model, x_batch=xn, y_batch=yn, a_batch=an,
      explain_func=make_explain_func(method, device), **common)
    av = np.asarray(av, dtype=float).ravel()
    out["avg_sensitivity"] = dict(
        mean=float(np.nanmean(av)), std=float(np.nanstd(av)),
        median=float(np.nanmedian(av)),
        nan=int(np.isnan(av).sum()), n=int(av.size), secs=time.time() - t)
    return out


@torch.no_grad()
def model_health(model, x, y):
    logits = model(x)
    pred = logits.argmax(1)
    accs = [(pred[y == c] == c).float().mean().item() for c in (0, 1)
            if (y == c).sum() > 0]
    p = torch.softmax(logits, 1).max(1).values
    return dict(balanced_accuracy=float(np.mean(accs)),
                saturated_frac=float((p > C.GATES["saturation_p"]).float().mean()),
                median_max_prob=float(p.median()))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=list(C.XAI_METHODS))
    ap.add_argument("--n", type=int, default=C.N_EVAL)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    sel = xai_eval_subset(n=a.n)
    x, y = load_batch(sel, a.device)
    print(f"eval batch {tuple(x.shape)}  independent view-groups="
          f"{sel.groupby(['case','side','view']).ngroups}  "
          f"labels={torch.bincount(y).tolist()}")

    names = a.models or list(MODELS)
    rows, health = [], {}
    for name in names:
        ck = C.CKPT_DIR / f"{name}_{a.tag}.pt"
        if not ck.exists():
            print(f"  skip {name}: no {ck.name}")
            continue
        model = build_model(name, pretrained=False, ckpt=str(ck), device=a.device)
        h = model_health(model, x, y)
        health[name] = h
        print(f"\n{name}  bal_acc={h['balanced_accuracy']:.4f}  "
              f"saturated={h['saturated_frac']:.1%}  "
              f"median max-prob={h['median_max_prob']:.4f}")

        for method in a.methods:
            att = EXPLAINERS[method](model, x, y)
            r = run_pair(model, x, y, att, method, a.device)
            for metric, d in r.items():
                rows.append(dict(model=name, xai_method=method, metric=metric,
                                 **d, **h))
                print(f"    {method:<22}{metric:<26}"
                      f"{d['mean']:>9.4f} +-{d['std']:<8.4f}"
                      f"NaN={d['nan']}/{d['n']}  ({d['secs']:.0f}s)")

    if not rows:
        print("\nno checkpoints found -- train first")
        return

    long = pd.DataFrame(rows)
    long.to_csv(C.RESULTS_DIR / "metrics_long.csv", index=False)

    piv = long.pivot_table(index=["model", "xai_method"], columns="metric",
                           values="mean").reset_index()
    piv["M1_task"] = C.TASK_NAME
    piv["M2_dataset"] = C.DATASET_NAME
    piv = piv.rename(columns={
        "model": "M3_model", "xai_method": "M4_xai_method",
        "faithfulness_correlation": "M5_faithfulness_correlation",
        "avg_sensitivity": "M6_avg_sensitivity",
        "sparseness": "M7_sparseness"})
    piv["M8_balanced_accuracy"] = piv["M3_model"].map(
        lambda m: health[m]["balanced_accuracy"])
    piv["notes"] = piv["M3_model"].map(
        lambda m: f"saturated={health[m]['saturated_frac']:.0%}")
    piv = piv[C.SHEET_COLUMNS + ["notes"]]
    piv.to_csv(C.RESULTS_DIR / "sheet.csv", index=False)

    print("\n" + "=" * 78)
    print(piv.drop(columns=["M1_task", "M2_dataset"]).to_string(index=False))
    print("=" * 78)
    print("M5 higher=better | M6 LOWER=better | M7 descriptive, not a score")
    print(f"wrote {C.RESULTS_DIR}/metrics_long.csv and sheet.csv")
    (C.RESULTS_DIR / "health.json").write_text(json.dumps(health, indent=2))


if __name__ == "__main__":
    main()