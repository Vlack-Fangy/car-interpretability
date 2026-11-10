#!/usr/bin/env python3
"""
Settle the epoch budget: are the 6 crops per view-group geometric transforms
of one patch, or genuinely different image regions?

    python dihedral_check.py --manifest manifest.csv --groups 400

Raw pixel correlation (0.209 median, from build_split.py) is NOT invariant to
rotation, flip, or translation. Rotate a patch 90 degrees and its correlation
with the original collapses toward zero while the content is identical. So a
low raw correlation does not by itself prove the crops are independent.

This computes three quantities per pair of crops:

    raw       plain zero-mean pixel correlation
    dihedral  max over the 8 flip/rotation transforms
    dih+shift max over 8 transforms AND all translations
              (FFT normalised cross-correlation -- "randomly cropped" means
               translation, so this is the one that matters most)

Verdict:
    dih+shift >> raw   -> crops are geometric variants. EFFECTIVE N = groups.
    dih+shift ~= raw   -> crops are distinct regions.   EFFECTIVE N = images.

Also writes contact sheets so the answer is visible, not just statistical.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config as C

SIDE = 64          # working resolution; enough for structure, cheap for FFT


# ------------------------------------------------------------ transforms
def dihedral8(a):
    """The 8 elements of the dihedral group D4."""
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def zmn(a):
    a = a.astype(np.float32)
    a = a - a.mean()
    n = np.linalg.norm(a)
    return a / (n if n > 0 else 1.0)


def ncc_shift(A, B):
    """Max normalised cross-correlation over ALL circular shifts, via FFT."""
    F = np.fft.rfft2(A) * np.conj(np.fft.rfft2(B))
    return float(np.fft.irfft2(F, s=A.shape).max())


# ------------------------------------------------------------ loading
def load(path, side=SIDE):
    im = Image.open(path).convert("L").resize((side, side), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def group_key(r):
    return (r.case, r.side, r.view)


# ------------------------------------------------------------ analysis
def analyse(df, n_groups, seed):
    groups = defaultdict(list)
    for r in df.itertuples():
        groups[group_key(r)].append(r.path)
    groups = {k: v for k, v in groups.items() if len(v) > 1}
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    if len(keys) > n_groups:
        keys = [keys[i] for i in rng.choice(len(keys), n_groups, replace=False)]

    raw, dih, dsh = [], [], []
    for gi, k in enumerate(keys):
        if gi % 25 == 0:
            print(f"      {gi}/{len(keys)} groups", end="\r", flush=True)
        imgs = [zmn(load(p)) for p in groups[k]]
        variants = [[zmn(t) for t in dihedral8(a)] for a in imgs]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                A = imgs[i]
                raw.append(float((A * imgs[j]).sum()))
                dih.append(max(float((A * v).sum()) for v in variants[j]))
                dsh.append(max(ncc_shift(A, v) for v in variants[j]))
    print(f"      {len(keys)}/{len(keys)} groups done   ")
    return (np.array(raw), np.array(dih), np.array(dsh),
            len(keys), sum(len(groups[k]) for k in keys))


def report(raw, dih, dsh, n_groups, n_imgs, df):
    def line(name, s):
        print(f"    {name:<12} mean={s.mean():+.4f}  median={np.median(s):+.4f}  "
              f"p95={np.percentile(s,95):+.4f}  >0.9={100*(s>0.9).mean():5.1f}%  "
              f">0.95={100*(s>0.95).mean():5.1f}%")

    print(f"    pairs compared: {len(raw):,}  "
          f"from {n_groups:,} view-groups / {n_imgs:,} images\n")
    line("raw", raw)
    line("dihedral", dih)
    line("dih+shift", dsh)

    # Near-exact duplication is the signal for "same patch, transformed".
    # Median level is a separate signal for "overlapping windows". Keying the
    # verdict on the median alone conflates the two: crops sampled from
    # overlapping regions of one ROI legitimately score ~0.4-0.7 on shifted
    # correlation without being duplicates of each other.
    f95 = float((dsh > 0.95).mean())
    f90 = float((dsh > 0.90).mean())
    med = float(np.median(dsh))
    lift = med - float(np.median(raw))
    print(f"\n    fraction dih+shift > 0.95 : {f95:.3f}   <- near-exact duplication")
    print(f"    median dih+shift          : {med:+.4f}   <- overlap degree")
    print(f"    median lift over raw      : {lift:+.4f}")

    print("\n" + "-" * 66)
    n_all = len(df)
    n_vg = df.groupby(["case", "side", "view"]).ngroups
    if f95 > 0.50:
        print("    VERDICT: GEOMETRIC VARIANTS of one patch.")
        print(f"    EFFECTIVE N = {n_vg:,} view-groups, not {n_all:,} images.")
        print(f"    -> size the epoch budget against {n_vg:,}.")
        print("    -> DO NOT add flip/rot90 augmentation: the dataset already")
        print("       contains the orbit. You would only re-sample it.")
        verdict = "geometric"
    elif f95 > 0.15 or f90 > 0.30:
        print("    VERDICT: MIXED -- a substantial minority are near-duplicates.")
        print(f"    -> effective N between {n_vg:,} and {n_all:,}. Be")
        print("       conservative on epochs; watch for early val plateau.")
        verdict = "mixed"
    elif med > 0.35:
        print("    VERDICT: OVERLAPPING WINDOWS, not duplicates.")
        print("    Crops share image content but none is a transform of another.")
        print(f"    -> effective N well above {n_vg:,}, below {n_all:,}.")
        print("    -> geometric augmentation still safe. Keep the group split:")
        print("       overlap is exactly why random splitting would leak.")
        verdict = "overlapping"
    else:
        print("    VERDICT: GENUINELY DIFFERENT REGIONS.")
        print(f"    EFFECTIVE N ~ {n_all:,} images.")
        print("    -> geometric augmentation is safe and useful.")
        verdict = "independent"
    print("-" * 66)
    return verdict


# ------------------------------------------------------------ contact sheet
def contact_sheet(df, out, n_groups=6, seed=C.SEED):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for r in df.itertuples():
        groups[group_key(r)].append((r.crop, r.path, r.cls))
    keys = sorted(k for k, v in groups.items() if len(v) >= 6)
    rng = np.random.default_rng(seed)
    keys = [keys[i] for i in rng.choice(len(keys),
                                        min(n_groups, len(keys)), replace=False)]

    ncol = max(len(groups[k]) for k in keys)
    fig, axes = plt.subplots(len(keys), ncol,
                             figsize=(2.0 * ncol, 2.15 * len(keys)))
    axes = np.atleast_2d(axes)
    for r, k in enumerate(keys):
        for c in range(ncol):
            ax = axes[r, c]
            ax.axis("off")
            items = sorted(groups[k])
            if c < len(items):
                crop, path, cls = items[c]
                ax.imshow(load(path, 227), cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"crop {crop}", fontsize=8)
        axes[r, 0].set_ylabel("y")
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"{k[0]}\n{k[1]}_{k[2]}", fontsize=7)
    fig.suptitle("Six crops per view-group -- are these transforms of one patch?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"    wrote {out}")


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(C.MANIFEST))
    ap.add_argument("--groups", type=int, default=400)
    ap.add_argument("--sheets", type=int, default=6)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    print("=" * 70)
    print(f"  DIHEDRAL / TRANSLATION REDUNDANCY TEST")
    print(f"  manifest={a.manifest}  images={len(df):,}")
    print("=" * 70)

    print("\n  CONTACT SHEETS")
    out = Path(C.FIG_DIR) / "crop_contact_sheet.png"
    contact_sheet(df, out, a.sheets, a.seed)
    print("    look at it. if they are rotations you will see it instantly.")

    print("\n  CORRELATION UNDER TRANSFORM GROUPS")
    raw, dih, dsh, ng, ni = analyse(df, a.groups, a.seed)
    verdict = report(raw, dih, dsh, ng, ni, df)

    np.savez(Path(C.RESULTS_DIR) / "dihedral_check.npz",
             raw=raw, dihedral=dih, dih_shift=dsh, verdict=verdict)
    print(f"\n    saved {C.RESULTS_DIR}/dihedral_check.npz\n")


if __name__ == "__main__":
    main()