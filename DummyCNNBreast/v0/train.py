#!/usr/bin/env python3
"""
Train one or all models. IDENTICAL PROTOCOL ACROSS MODELS.

    python train.py --model resnet18
    python train.py --all
    python train.py --model resnet18 --epochs 40 --tag budget_probe

Same split, same epoch budget, same optimiser, same schedule, same
augmentation for every model. Only the learning rate may differ, and only
within LR_SWEEP_GRID in config. Anything else and you are comparing training
runs rather than architectures, and the three-model story collapses.

NO EARLY STOPPING. A fixed budget keeps the protocol identical. Pick the
budget once from a val curve (run ResNet18 with a generous --epochs, read the
plateau), then hold it for all three.

Writes per model:
    checkpoints/<name>.pt          weights + the config it was trained under
    results/<name>_curve.csv       per-epoch train/val loss and balanced acc
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import time
from pathlib import Path

import determinism                     # before torch: sets CUBLAS env pre-CUDA
determinism.enable()

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
import torch.nn as nn                                         # noqa: E402

import config as C                                            # noqa: E402
from data_loader import make_loader, describe                 # noqa: E402
from models.registry import MODELS                            # noqa: E402
from models.wrap import build_model                           # noqa: E402


def balanced_accuracy(pred, y):
    accs = []
    for c in (0, 1):
        m = y == c
        if m.sum() == 0:
            return float("nan")
        accs.append((pred[m] == c).float().mean().item())
    return float(np.mean(accs))


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    tot, n, P, Y = 0.0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        out = model(xb)
        tot += criterion(out, yb).item() * len(yb)
        n += len(yb)
        P.append(out.argmax(1).cpu())
        Y.append(yb.cpu())
    P, Y = torch.cat(P), torch.cat(Y)
    return tot / n, balanced_accuracy(P, Y), (P == Y).float().mean().item()


def make_scheduler(opt, epochs, steps_per_epoch, warmup_epochs):
    total = epochs * steps_per_epoch
    warm = warmup_epochs * steps_per_epoch

    def fn(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        p = (step - warm) / max(total - warm, 1)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train_one(name, device, epochs, lr, tag=""):
    spec = MODELS[name]
    bs = spec.batch_size
    print(f"\n{'='*70}\n  {name}   bs={bs}  lr={lr:g}  epochs={epochs}  "
          f"device={device}\n{'='*70}")

    tr = make_loader("train", bs, augment=True)
    va = make_loader("val", bs, augment=False)
    print(f"    train batches={len(tr)}  val batches={len(va)}")

    model = build_model(name, pretrained=True, device=device)
    model.train()

    criterion = nn.CrossEntropyLoss(
        label_smoothing=C.TRAIN["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=C.TRAIN["weight_decay"])
    sched = make_scheduler(opt, epochs, len(tr), C.TRAIN["warmup_epochs"])

    rows, best = [], -1.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run, n, P, Y = 0.0, 0, [], []
        for xb, yb in tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(yb)
            n += len(yb)
            P.append(out.argmax(1).detach().cpu())
            Y.append(yb.cpu())
        tr_loss = run / n
        tr_bal = balanced_accuracy(torch.cat(P), torch.cat(Y))
        va_loss, va_bal, va_acc = evaluate(model, va, device, criterion)

        rows.append(dict(epoch=ep, lr=sched.get_last_lr()[0],
                         train_loss=tr_loss, train_bal_acc=tr_bal,
                         val_loss=va_loss, val_bal_acc=va_bal, val_acc=va_acc,
                         elapsed_s=time.time() - t0))
        flag = ""
        if va_bal > C.GATES["accuracy_ceiling"]:
            flag = "  <-- above leakage ceiling, check the split"
        print(f"    ep {ep:>3}/{epochs}  train {tr_loss:.4f}/{tr_bal:.4f}   "
              f"val {va_loss:.4f}/{va_bal:.4f}{flag}")
        best = max(best, va_bal)

    # Fixed budget: the FINAL epoch is the checkpoint, not the best epoch.
    # Selecting the best epoch on val is a form of early stopping and would
    # break protocol identity across models.
    ck = C.CKPT_DIR / f"{name}{('_' + tag) if tag else ''}.pt"
    torch.save(dict(model=model.state_dict(), name=name, epochs=epochs, lr=lr,
                    train_cfg=C.TRAIN, augment=C.AUGMENT, seed=C.SEED,
                    final_val_bal_acc=rows[-1]["val_bal_acc"],
                    best_val_bal_acc=best), ck)

    import pandas as pd
    cv = C.RESULTS_DIR / f"{name}{('_' + tag) if tag else ''}_curve.csv"
    pd.DataFrame(rows).to_csv(cv, index=False)

    fin = rows[-1]["val_bal_acc"]
    lo, hi = C.GATES["accuracy_floor"], C.GATES["accuracy_ceiling"]
    verdict = ("BELOW FLOOR - XAI would be noise" if fin < lo else
               "ABOVE CEILING - suspect leakage" if fin > hi else
               "in range - reportable")
    print(f"\n    final val balanced acc : {fin:.4f}   [{verdict}]")
    print(f"    best  val balanced acc : {best:.4f}")
    print(f"    saved {ck.name}, {cv.name}   ({(time.time()-t0)/60:.1f} min)")
    return fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=C.TRAIN["epochs"])
    ap.add_argument("--lr", type=float, default=C.TRAIN["lr"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    print("=" * 70)
    print("  DATA")
    print("=" * 70)
    describe()

    names = list(MODELS) if a.all else [a.model or "resnet18"]
    out = {n: train_one(n, a.device, a.epochs, a.lr, a.tag) for n in names}

    print("\n" + "=" * 70)
    print("  SUMMARY  (final-epoch val balanced accuracy)")
    print("=" * 70)
    for n, v in out.items():
        print(f"    {n:<20} {v:.4f}")
    (C.RESULTS_DIR / "train_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()