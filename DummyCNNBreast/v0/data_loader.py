"""
Data loading. Everything comes from the frozen manifest -- nothing re-splits.

Output contract, matching what the model wrapper expects:
    (1, 227, 227) float32 in [0,1]     -- grayscale, unnormalised
    label 0 = benign, 1 = malignant

Normalisation and grayscale->3ch replication happen INSIDE the model
(models/wrap.py), never here. That keeps "zero == black" true so Quantus's
black-baseline perturbation actually erases pixels.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import config as C
from determinism import seed_worker


class DDSMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False,
                 seed: int = C.SEED):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("L")
        if im.size != (C.IMG_SIZE, C.IMG_SIZE):
            im = im.resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0        # -> [0,1]

        if self.augment:
            # Flips only. NO rot90: dihedral_check found the crops are already
            # rotated relative to each other (median correlation 0.19 raw ->
            # 0.65 after allowing rotation). Adding rotation would re-sample
            # variation the dataset already contains.
            if self.rng.random() < C.AUGMENT["hflip"]:
                a = np.ascontiguousarray(a[:, ::-1])
            if self.rng.random() < C.AUGMENT["vflip"]:
                a = np.ascontiguousarray(a[::-1, :])

        x = torch.from_numpy(a).unsqueeze(0)                # (1,H,W)
        return x, int(self.labels[i])


def reroot(df: pd.DataFrame, max_depth: int = 6) -> pd.DataFrame:
    """Make the manifest portable across machines.

    build_split.py wrote absolute paths from whatever box it ran on; moving to
    a server breaks all of them. Rather than regenerate the manifest (which
    risks a different split), find how many trailing path components to keep
    so that C.DATA_ROOT/<suffix> exists, then apply that depth to every row.
    Which case lands in which fold is untouched.
    """
    paths = df["path"].tolist()
    if Path(paths[0]).exists():
        return df

    new_root = Path(C.DATA_ROOT)
    probe = Path(paths[0]).parts
    depth = None
    for k in range(1, min(max_depth, len(probe)) + 1):
        if (new_root.joinpath(*probe[-k:])).exists():
            depth = k                       # keep searching for the longest
    if depth is None:
        raise FileNotFoundError(
            f"re-rooting failed.\n  manifest path : {paths[0]}\n"
            f"  DDSM_ROOT     : {new_root}\n"
            f"set DDSM_ROOT to the folder CONTAINING 'DDSM Dataset/', e.g.\n"
            f"  export DDSM_ROOT=$HOME/CAR/data/ddsm_sanchris")

    fixed = [str(new_root.joinpath(*Path(p).parts[-depth:])) for p in paths]
    missing = [f for f in fixed[:50] if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"re-rooted but missing: {missing[0]}")
    print(f"    [reroot] kept last {depth} components -> {new_root}")
    df = df.copy()
    df["path"] = fixed
    return df


def load_manifest(path: Optional[Path] = None) -> pd.DataFrame:
    df = pd.read_csv(path or C.MANIFEST)
    need = {"path", "label", "split", "case", "side", "view", "crop"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    return reroot(df)


def make_loader(split: str, batch_size: int, augment: bool = False,
                shuffle: Optional[bool] = None, df: Optional[pd.DataFrame] = None):
    df = load_manifest() if df is None else df
    sub = df[df["split"] == split].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"no rows for split={split}")
    g = torch.Generator()
    g.manual_seed(C.SEED)
    return DataLoader(
        DDSMDataset(sub, augment=augment),
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=C.TRAIN["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )


def xai_eval_subset(n: int = C.N_EVAL, seed: int = C.SEED,
                    df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The N images XAI metrics are computed on -- ONE CROP PER VIEW-GROUP.

    Why not just sample N rows at random: the eval split holds ~2,010 images
    but only ~335 view-groups, and crops within a group overlap heavily
    (median correlation 0.75 once rotation and shift are allowed). Sampling
    rows directly would draw ~128 crops from ~60 distinct lesions. The mean
    would survive that, but the standard deviation and NaN rate would be
    wrong, because the samples are not independent.

    One crop per group, class-balanced, gives N independent lesions.
    """
    df = load_manifest() if df is None else df
    ev = df[df["split"] == "eval"]
    rng = np.random.default_rng(seed)

    # one row per (case, side, view)
    picked = (ev.groupby(["case", "side", "view"], sort=True)
                .apply(lambda g: g.iloc[rng.integers(len(g))],
                       include_groups=False)
                .reset_index())
    if "label" not in picked.columns:
        picked = picked.merge(ev[["path", "label", "cls"]], on="path", how="left")

    per_class = n // 2
    out = []
    for lab in (0, 1):
        pool = picked[picked["label"] == lab]
        if len(pool) < per_class:
            raise ValueError(
                f"only {len(pool)} independent view-groups for label={lab}, "
                f"need {per_class}. lower N_EVAL in config.")
        out.append(pool.sample(n=per_class, random_state=seed))
    sel = pd.concat(out).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sel


def load_batch(df: pd.DataFrame, device: str = "cpu"):
    """Materialise a dataframe as one tensor batch, for Quantus."""
    ds = DDSMDataset(df, augment=False)
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    return torch.stack(xs).to(device), torch.tensor(ys).to(device)


def describe(df: Optional[pd.DataFrame] = None) -> None:
    df = load_manifest() if df is None else df
    print(f"    {'split':<8}{'images':>9}{'cases':>9}{'view-grps':>11}"
          f"{'benign':>9}{'malig':>8}")
    for s in ("train", "val", "eval"):
        d = df[df["split"] == s]
        print(f"    {s:<8}{len(d):>9,}{d['case'].nunique():>9,}"
              f"{d.groupby(['case','side','view']).ngroups:>11,}"
              f"{(d['label']==0).sum():>9,}{(d['label']==1).sum():>8,}")