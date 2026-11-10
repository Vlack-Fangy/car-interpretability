#!/usr/bin/env python3
"""
STEP 1b - group structure analysis + frozen split manifest.

    python build_split.py ~/Desktop/CAR/data/ddsm_sanchris --out manifest.csv

Observed filename grammar (sanchris DDSM):
    D1_A_1177_1.RIGHT_CC (2).png
    <-- case --> <side>_<view> <crop>

What this does
  1. parse every filename -> (case, side, view, crop_idx); report coverage
  2. two grouping keys (full case vs. case with trailing _N stripped) so the
     conservative choice is made on evidence, not vibes
  3. case/class overlap check
  4. crop redundancy: within-view-group pixel correlation  -> effective N
  5. chunked cosine NN scan at 32x32 -> cross-class near-duplicates that
     byte-hashing cannot see
  6. emit a FROZEN, case-disjoint, class-stratified train/val/eval manifest

Everything downstream reads the manifest. It is the single source of truth.
Requires: numpy, pillow, pandas.
"""
import argparse, os, re, sys, json, hashlib
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
THUMB = 32                      # side length for the similarity scan
SEED = 1337
SPLIT_FRAC = (0.70, 0.15, 0.15)  # train / val / eval, applied at CASE level

# D1_A_1177_1.RIGHT_CC (2)   ->  case | side | view | crop
FNAME_RE = re.compile(
    r"^(?P<case>.+?)\.(?P<side>LEFT|RIGHT)[_-](?P<view>CC|MLO)"
    r"(?:\s*\((?P<crop>\d+)\))?$",
    re.IGNORECASE,
)
TRAILING_N = re.compile(r"_\d+$")


def rule(t=""):
    print("\n" + "=" * 70)
    if t:
        print(f"  {t}")
        print("=" * 70)


# ------------------------------------------------------------------ 1. parse
def scan(root):
    rows, unparsed = [], []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in IMG_EXT:
                continue
            cls = os.path.basename(dirpath)
            m = FNAME_RE.match(stem)
            if not m:
                unparsed.append(f)
                rows.append(dict(path=os.path.join(dirpath, f), cls=cls,
                                 case=stem, case_coarse=stem, side="?",
                                 view="?", crop=1, parsed=False))
                continue
            case = m.group("case")
            rows.append(dict(
                path=os.path.join(dirpath, f), cls=cls,
                case=case,
                case_coarse=TRAILING_N.sub("", case),
                side=m.group("side").upper(),
                view=m.group("view").upper(),
                crop=int(m.group("crop") or 1),
                parsed=True,
            ))
    return rows, unparsed


def report_parse(rows, unparsed):
    n = len(rows)
    ok = sum(r["parsed"] for r in rows)
    print(f"    images            : {n:,}")
    print(f"    parsed            : {ok:,}  ({100*ok/n:.2f}%)")
    print(f"    UNPARSED          : {len(unparsed):,}")
    for f in unparsed[:10]:
        print(f"        {f}")
    if unparsed and len(unparsed) > 0.01 * n:
        print("    !! >1% unparsed -- fix the regex before trusting the split")

    print(f"\n    unique case       : {len({r['case'] for r in rows}):,}")
    print(f"    unique case_coarse: {len({r['case_coarse'] for r in rows}):,}")
    print(f"    sides             : {dict(Counter(r['side'] for r in rows))}")
    print(f"    views             : {dict(Counter(r['view'] for r in rows))}")
    cr = Counter(r["crop"] for r in rows)
    print(f"    crop indices      : {dict(sorted(cr.items())[:12])}")

    per_case = Counter(r["case"] for r in rows)
    v = np.array(list(per_case.values()))
    print(f"\n    images per case   : mean={v.mean():.1f} median={np.median(v):.0f} "
          f"min={v.min()} max={v.max()}")
    vg = Counter((r["case"], r["side"], r["view"]) for r in rows)
    w = np.array(list(vg.values()))
    print(f"    view-groups       : {len(vg):,}")
    print(f"    crops per view    : mean={w.mean():.1f} median={np.median(w):.0f} "
          f"min={w.min()} max={w.max()}")


def report_key_choice(rows):
    full = {r["case"] for r in rows}
    coarse = {r["case_coarse"] for r in rows}
    print(f"    case        -> {len(full):,} groups")
    print(f"    case_coarse -> {len(coarse):,} groups   "
          f"(merges {len(full)-len(coarse):,})")
    if len(full) == len(coarse):
        print("    identical: trailing _N carries no grouping information.")
        return "case"
    print("    trailing _N DOES merge groups.")
    print("    -> using case_coarse (coarser = strictly safer against leakage)")
    return "case_coarse"


# ------------------------------------------------------- 2. class purity
def report_overlap(rows, key):
    by = defaultdict(set)
    for r in rows:
        by[r[key]].add(r["cls"])
    mixed = {k: v for k, v in by.items() if len(v) > 1}
    print(f"    groups spanning >1 class: {len(mixed):,} / {len(by):,}")
    for k in list(mixed)[:8]:
        print(f"        {k}  -> {sorted(mixed[k])}")
    if mixed:
        print("    note: group-splitting still holds. Each image keeps its own")
        print("          label; the whole group lands in one split. Stratify on")
        print("          the group's majority class.")
    else:
        print("    every group is class-pure -> clean stratification")
    return mixed


# ------------------------------------------------------- 3. thumbnails
def load_thumbs(rows, size=THUMB):
    n = len(rows)
    X = np.zeros((n, size * size), dtype=np.float32)
    bad = []
    for i, r in enumerate(rows):
        if i % 2000 == 0:
            print(f"      {i:,}/{n:,}", end="\r", flush=True)
        try:
            im = Image.open(r["path"]).convert("L").resize(
                (size, size), Image.BILINEAR)
            X[i] = np.asarray(im, dtype=np.float32).ravel()
        except Exception as e:
            bad.append((r["path"], str(e)))
    print(f"      {n:,}/{n:,} done      ")
    if bad:
        print(f"      !! {len(bad)} unreadable")
    X -= X.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm                       # rows are unit, mean-centred


# --------------------------------------------- 4. within-group redundancy
def report_redundancy(rows, X):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["case"], r["side"], r["view"])].append(i)
    sims = []
    rng = np.random.default_rng(SEED)
    keys = [k for k, v in groups.items() if len(v) > 1]
    for k in (keys if len(keys) <= 3000 else
              [keys[i] for i in rng.choice(len(keys), 3000, replace=False)]):
        idx = groups[k]
        G = X[idx] @ X[idx].T
        iu = np.triu_indices(len(idx), k=1)
        sims.append(G[iu])
    if not sims:
        print("    no multi-crop groups found")
        return
    s = np.concatenate(sims)
    print(f"    pairs compared          : {len(s):,}")
    print(f"    correlation of crops    : mean={s.mean():.4f} "
          f"median={np.median(s):.4f} p05={np.percentile(s,5):.4f}")
    for t in (0.90, 0.95, 0.99, 0.999):
        print(f"      fraction > {t:<6}      : {(s > t).mean():.3f}")
    n_img, n_grp = len(rows), len(groups)
    print(f"\n    images                  : {n_img:,}")
    print(f"    view-groups             : {n_grp:,}")
    if np.median(s) > 0.95:
        print(f"    -> crops are near-redundant. EFFECTIVE N ~ {n_grp:,},")
        print(f"       not {n_img:,}. Size the epoch budget against that.")
    else:
        print(f"    -> crops carry real variation; treat as strong augmentation")


# --------------------------------------------- 5. cross-class near-dupes
def report_cross_class(rows, X, thresh=0.99, block=512):
    cls = np.array([r["cls"] for r in rows])
    case = np.array([r["case"] for r in rows])
    n = len(rows)
    hits, nn_same_case = [], 0
    for a in range(0, n, block):
        b = min(a + block, n)
        G = X[a:b] @ X.T                       # (block, n)
        for i in range(b - a):
            gi = a + i
            G[i, gi] = -2.0                    # mask self
            j = int(np.argmax(G[i]))
            if case[j] == case[gi]:
                nn_same_case += 1
            if G[i, j] > thresh and cls[j] != cls[gi] and gi < j:
                hits.append((gi, j, float(G[i, j])))
        print(f"      {b:,}/{n:,}", end="\r", flush=True)
    print(f"      {n:,}/{n:,} done      ")
    print(f"    nearest neighbour is same case : "
          f"{nn_same_case:,}/{n:,}  ({100*nn_same_case/n:.1f}%)")
    print(f"    cross-class pairs  cos>{thresh}  : {len(hits):,}")
    for gi, j, v in sorted(hits, key=lambda t: -t[2])[:8]:
        print(f"        {v:.4f}  {os.path.basename(rows[gi]['path'])[:34]:<36}"
              f"{rows[gi]['cls'][:9]:<10} <-> "
              f"{os.path.basename(rows[j]['path'])[:34]:<36}{rows[j]['cls'][:9]}")
    if hits:
        print("    !! near-identical images with OPPOSITE labels.")
        print("       Either label noise or duplicated source lesions.")
        print("       These cap achievable accuracy and poison XAI eval.")
    else:
        print("    clean: no near-duplicate label conflicts")
    return hits


# ------------------------------------------------------- 6. frozen split
def build_split(rows, key, fracs=SPLIT_FRAC, seed=SEED):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r[key]].append(i)
    maj = {g: Counter(rows[i]["cls"] for i in idx).most_common(1)[0][0]
           for g, idx in by.items()}
    rng = np.random.default_rng(seed)
    assign = {}
    for c in sorted(set(maj.values())):
        gs = sorted(g for g in by if maj[g] == c)
        rng.shuffle(gs)
        n = len(gs)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        for g in gs[:n_tr]:
            assign[g] = "train"
        for g in gs[n_tr:n_tr + n_va]:
            assign[g] = "val"
        for g in gs[n_tr + n_va:]:
            assign[g] = "eval"
    for r in rows:
        r["split"] = assign[r[key]]
    return assign


def report_split(rows, key, assign):
    tab = defaultdict(Counter)
    for r in rows:
        tab[r["split"]][r["cls"]] += 1
    gtab = Counter(assign.values())
    print(f"    {'split':<8}{'groups':>9}{'images':>10}   class breakdown")
    for s in ("train", "val", "eval"):
        c = tab[s]
        tot = sum(c.values())
        bd = "  ".join(f"{k}={v:,}" for k, v in sorted(c.items()))
        print(f"    {s:<8}{gtab[s]:>9,}{tot:>10,}   {bd}")
    ev = tab["eval"]
    tot = sum(ev.values())
    if tot:
        mn = min(ev.values()) / tot
        print(f"\n    eval balance    : minority {mn:.1%}")
        print(f"    eval pool size  : {tot:,}  "
              f"({'OK' if tot >= 128 else 'TOO SMALL'} for N=128)")
    # the assertion that matters
    seen = defaultdict(set)
    for r in rows:
        seen[r[key]].add(r["split"])
    bad = [g for g, s in seen.items() if len(s) > 1]
    print(f"\n    GROUP DISJOINTNESS: "
          f"{'PASS - no group spans splits' if not bad else f'FAIL - {len(bad)} leak'}")
    return not bad


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--fast", action="store_true",
                    help="skip the similarity scans")
    a = ap.parse_args()

    rule("1. FILENAME PARSE")
    rows, unparsed = scan(os.path.abspath(a.root))
    if not rows:
        print("    no images found"); sys.exit(1)
    report_parse(rows, unparsed)

    rule("2. GROUPING KEY")
    key = report_key_choice(rows)

    rule("3. CLASS PURITY PER GROUP")
    report_overlap(rows, key)

    if not a.fast:
        rule("4. LOADING THUMBNAILS  (32x32, for similarity)")
        X = load_thumbs(rows)

        rule("5. CROP REDUNDANCY  -> effective sample size")
        report_redundancy(rows, X)

        rule("6. CROSS-CLASS NEAR-DUPLICATE SCAN")
        report_cross_class(rows, X)

    rule("7. FROZEN SPLIT  (case-disjoint, class-stratified)")
    assign = build_split(rows, key)
    ok = report_split(rows, key, assign)

    import pandas as pd
    df = pd.DataFrame(rows)[
        ["path", "cls", "case", "case_coarse", "side", "view",
         "crop", "parsed", "split"]]
    df["label"] = (df["cls"].str.lower().str.contains("malignant")).astype(int)
    df.to_csv(a.out, index=False)

    meta = dict(root=os.path.abspath(a.root), group_key=key, seed=SEED,
                fracs=SPLIT_FRAC, n_images=len(df),
                n_groups=len(assign), disjoint=bool(ok),
                label_map={"0": "benign", "1": "malignant"},
                manifest_sha=hashlib.sha256(
                    open(a.out, "rb").read()).hexdigest()[:16])
    with open(os.path.splitext(a.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    rule("WROTE")
    print(f"    {a.out}")
    print(f"    {os.path.splitext(a.out)[0]}_meta.json")
    print(f"    manifest sha256[:16] = {meta['manifest_sha']}")
    print("\n    label: 0=benign 1=malignant")
    print("    every downstream script reads THIS FILE. Do not re-split.\n")


if __name__ == "__main__":
    main()