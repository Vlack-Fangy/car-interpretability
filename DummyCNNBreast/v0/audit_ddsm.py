#!/usr/bin/env python3
"""
STEP 1 - format-agnostic dataset audit. Read-only, writes nothing.

    python audit_dataset.py /path/to/extracted/dataset

Makes no assumption about layout. Reports:
  1. directory tree + file-type census
  2. any CSV: columns, dtypes, low-cardinality value counts, label/patient candidates
  3. class-folder layout: per-folder counts
  4. image sample: shape, mode, dtype, pixel range, TRUE-grayscale check
  5. filename pattern mining -> can we recover patient/case IDs?  (grouping!)
  6. exact-duplicate hash scan -> train/test contamination detector

Requires: numpy, pillow, pandas.
"""
import sys, os, csv, hashlib, re
from collections import Counter, defaultdict

MAX_TREE_ENTRIES = 12
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}
ARR_EXT = {".npy", ".npz"}
TAB_EXT = {".csv", ".tsv"}


def rule(t=""):
    print("\n" + "=" * 70)
    if t:
        print(f"  {t}")
        print("=" * 70)


# ------------------------------------------------------------------ tree
def walk_tree(root):
    print(f"  {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        depth = dirpath[len(root):].count(os.sep)
        if depth > 3:
            dirnames[:] = []
            continue
        indent = "    " * (depth + 1)
        rel = os.path.relpath(dirpath, root)
        if rel != ".":
            print(f"{indent}{os.path.basename(dirpath)}/   "
                  f"[{len(filenames)} files, {len(dirnames)} dirs]")
        ext_c = Counter(os.path.splitext(f)[1].lower() for f in filenames)
        if ext_c and depth <= 2:
            shown = ", ".join(f"{e or '<none>'}:{n}" for e, n in ext_c.most_common(6))
            print(f"{indent}    -> {shown}")
        for f in sorted(filenames)[:MAX_TREE_ENTRIES]:
            if os.path.splitext(f)[1].lower() in TAB_EXT | ARR_EXT:
                p = os.path.join(dirpath, f)
                print(f"{indent}    * {f}  ({os.path.getsize(p)/1e6:.1f} MB)")


def census(root):
    ext_c, total = Counter(), 0
    files = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            e = os.path.splitext(f)[1].lower()
            ext_c[e] += 1
            total += 1
            files[e].append(os.path.join(dirpath, f))
    print(f"\n  total files: {total:,}")
    for e, n in ext_c.most_common(15):
        print(f"    {e or '<no ext>':<12} {n:>8,}")
    return files


# ------------------------------------------------------------------ csv
LABEL_HINTS = ("pathology", "label", "class", "target", "diagnosis",
               "malignant", "benign", "cancer", "y")
ID_HINTS = ("patient", "case", "subject", "study", "id", "image_id",
            "ref", "series", "uid")


def audit_csv(path):
    import pandas as pd
    print(f"\n  --- {os.path.basename(path)} ---")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"    unreadable: {e}")
        return None
    print(f"    shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"    columns: {list(df.columns)}")

    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        lc = c.lower()
        tagl = "  <-- LABEL?" if any(h in lc for h in LABEL_HINTS) else ""
        tagi = "  <-- ID / GROUPING KEY?" if any(h in lc for h in ID_HINTS) else ""
        if nun <= 12:
            vc = df[c].value_counts(dropna=False)
            print(f"    {c:<28} nuniq={nun:<7}{tagl}{tagi}")
            for v, n in vc.items():
                print(f"        {str(v)[:44]:<46} {n:>7,}")
        else:
            print(f"    {c:<28} nuniq={nun:<7} "
                  f"ex={str(df[c].dropna().iloc[0])[:40] if len(df[c].dropna()) else '-':<42}"
                  f"{tagl}{tagi}")
    return df


# ------------------------------------------------------------------ folders
def audit_class_folders(root):
    """Any directory whose children are mostly images = a class folder."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        imgs = [f for f in filenames
                if os.path.splitext(f)[1].lower() in IMG_EXT]
        if len(imgs) >= 5 and not dirnames:
            found.append((os.path.relpath(dirpath, root), len(imgs), dirpath))
    if not found:
        print("    no image class-folders detected")
        return []
    found.sort()
    total = sum(n for _, n, _ in found)
    for rel, n, _ in found:
        print(f"    {rel:<52} {n:>7,}  ({100*n/total:5.1f}%)")
    print(f"    {'TOTAL':<52} {total:>7,}")

    if len(found) == 2:
        a, b = found[0][1], found[1][1]
        r = min(a, b) / (a + b)
        print(f"\n    binary balance: minority = {r:.1%}")
        if not 0.15 <= r <= 0.85:
            print("      !! imbalanced -> force-balance the XAI eval split")
    return found


# ------------------------------------------------------------------ images
def audit_images(paths, k=6):
    import numpy as np
    from PIL import Image
    print(f"    sampling {min(k, len(paths))} of {len(paths):,} images\n")
    shapes, modes = Counter(), Counter()
    for p in paths[:k]:
        try:
            im = Image.open(p)
            a = np.array(im)
        except Exception as e:
            print(f"    {os.path.basename(p)[:44]:<46} unreadable: {e}")
            continue
        shapes[a.shape] += 1
        modes[im.mode] += 1
        gray = "n/a"
        if a.ndim == 3 and a.shape[-1] >= 3:
            gray = bool(np.array_equal(a[..., 0], a[..., 1])
                        and np.array_equal(a[..., 1], a[..., 2]))
        print(f"    {os.path.basename(p)[:44]:<46} {str(a.shape):<18}"
              f"{str(a.dtype):<9} mode={im.mode:<5} "
              f"range=[{a.min()},{a.max()}] chans_identical={gray}")
    print(f"\n    shapes seen : {dict(shapes)}")
    print(f"    PIL modes   : {dict(modes)}")
    if len(shapes) > 1:
        print("    !! heterogeneous sizes -> resize in the loader, fixed policy")
    if any(m in ("RGB", "RGBA") for m in modes):
        print("    !! stored as RGB. If channels are identical it is fake-RGB:")
        print("       collapse to 1 channel at load. Replication happens INSIDE")
        print("       the model, so Quantus perturbs real grayscale pixels.")
    if any(str(s) for s in shapes) and all(
            np.dtype(np.uint8) for _ in [0]):
        print("    -> scale uint8 to [0,1]; normalization stays inside the model")


# ------------------------------------------------------------------ filenames
CBIS_PAT = re.compile(r"(P[_-]?\d{4,6})", re.I)
GENERIC_ID = re.compile(r"(\d{3,8})")


def mine_filenames(paths, sample=4000):
    """Can we recover a grouping key? This decides whether patient-level
    splitting is possible at all."""
    names = [os.path.basename(p) for p in paths[:sample]]
    print(f"    examples:")
    for n in names[:8]:
        print(f"      {n}")

    hits = [CBIS_PAT.search(n) for n in names]
    n_hit = sum(1 for h in hits if h)
    print(f"\n    CBIS patient pattern (P_00001): "
          f"{n_hit:,}/{len(names):,} filenames matched")
    if n_hit > 0.5 * len(names):
        pid = Counter(h.group(1).upper().replace("-", "_") for h in hits if h)
        print(f"      -> {len(pid):,} unique patient IDs recovered")
        print(f"      -> images per patient: mean="
              f"{sum(pid.values())/len(pid):.2f} max={max(pid.values())}")
        print("      ** PATIENT-LEVEL SPLITTING IS POSSIBLE. Use it. **")
        return "cbis"

    gid = [GENERIC_ID.search(n) for n in names]
    n_gid = sum(1 for g in gid if g)
    if n_gid > 0.5 * len(names):
        ids = Counter(g.group(1) for g in gid if g)
        rep = sum(1 for v in ids.values() if v > 1)
        print(f"    generic numeric ID: {len(ids):,} unique, "
              f"{rep:,} appear more than once")
        if rep > 0.05 * len(ids):
            print("      -> repeated IDs = likely multiple crops/views per case.")
            print("         Candidate grouping key. VERIFY before trusting it.")
            return "generic"
        print("      -> looks like a plain running index, NOT a grouping key")
    else:
        print("    no usable ID pattern in filenames")
    print("    ** NO GROUPING KEY -> patient-level splitting impossible. **")
    print("       Treat any eval accuracy > 0.95 as leakage, not skill.")
    return None


# ------------------------------------------------------------------ dupes
def dupe_scan(groups, limit=30000):
    """groups: {name: [paths]}.  Hash raw bytes; report within & across."""
    seen, within, across = {}, Counter(), []
    for gname, paths in groups.items():
        local = set()
        for p in paths[:limit]:
            try:
                with open(p, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
            except Exception:
                continue
            if h in local:
                within[gname] += 1
            local.add(h)
            if h in seen and seen[h] != gname:
                across.append((seen[h], gname, os.path.basename(p)))
            seen[h] = gname
    print(f"    unique image hashes : {len(seen):,}")
    for g, n in within.items():
        print(f"    duplicates within {g:<30} {n:,}")
    if across:
        print(f"\n    !! {len(across):,} IDENTICAL images across groups:")
        for a, b, f in across[:8]:
            print(f"        {a}  <->  {b}   {f}")
        print("    This is direct contamination. Dedupe before training.")
    else:
        print("    no byte-identical images across groups")


# ------------------------------------------------------------------ main
def main(root):
    root = os.path.abspath(root)
    rule("DIRECTORY TREE")
    walk_tree(root)

    rule("FILE-TYPE CENSUS")
    files = census(root)

    csvs = [p for e in TAB_EXT for p in files.get(e, [])]
    if csvs:
        rule("TABULAR METADATA")
        for p in sorted(csvs)[:6]:
            audit_csv(p)
        print("\n    -> if a patient/case column exists, that IS your split key.")

    npys = [p for e in ARR_EXT for p in files.get(e, [])]
    if npys:
        import numpy as np
        rule("NUMPY ARRAYS")
        for p in sorted(npys)[:10]:
            try:
                a = np.load(p, mmap_mode="r", allow_pickle=False)
                print(f"    {os.path.relpath(p, root):<44} shape={a.shape} "
                      f"dtype={a.dtype}")
                if a.ndim == 1 and a.size < 200000:
                    print(f"        values: {dict(Counter(np.asarray(a).ravel().tolist()).most_common(8))}")
            except Exception as e:
                print(f"    {os.path.relpath(p, root):<44} {e}")

    imgs = [p for e in IMG_EXT for p in files.get(e, [])]
    if imgs:
        rule("CLASS-FOLDER LAYOUT")
        folders = audit_class_folders(root)

        rule("IMAGE PROPERTIES")
        audit_images(sorted(imgs))

        rule("FILENAME ID MINING  (decides split strategy)")
        mine_filenames(sorted(imgs))

        rule("DUPLICATE / CONTAMINATION SCAN")
        if folders:
            groups = {rel: sorted(
                os.path.join(dp, f) for f in os.listdir(dp)
                if os.path.splitext(f)[1].lower() in IMG_EXT)
                for rel, _, dp in folders}
        else:
            groups = {"all": sorted(imgs)}
        dupe_scan(groups)

    rule("DECISIONS THIS AUDIT FEEDS")
    print("""    1. split key      -> patient col / filename ID / none (leakage risk)
    2. eval pool size -> is N=128 balanced reachable?
    3. channel policy -> real grayscale, or fake-RGB to collapse
    4. resize policy  -> heterogeneous sizes?
    5. accuracy floor -> if >0.95 is easy, suspect contamination first
""")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])