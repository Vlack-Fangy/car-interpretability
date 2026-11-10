#!/usr/bin/env python3
"""
01_data.py -- Suicide Detection (Kaggle rvarun11/suicidal-ideation-reddit-dataset)

Owns EVERYTHING task-specific about the data. 00_gate.py and 04_cache.py both call
load_rows(split), so the two loaders cannot drift -- the whole pipeline depends on
them emitting identically-ordered rows (see 06 check in the plan's verification list).

Emits the Dreaddit row schema so nothing downstream changes:

  text       the post
  label      1 = suicidal, 0 = not
  conf       PROXY. Dreaddit ships annotator agreement; this dataset does not.
             Five bagged lexical annotators, out-of-fold, agreeing with the gold
             label -> {0, .2, .4, .6, .8, 1.0}. Same value grid as Dreaddit, so the
             `conf >= 1.0` filters in 00_gate.py and 05_patch.py need NO code change.
             Independent of the three LLMs under study, so it puts no circularity
             into the patching pairs (an |LD|-percentile proxy would).
  subreddit  PROXY. No subreddit column exists and it is not recoverable. Holds a
             TF-IDF/SVD/KMeans topic cluster id ("topic_03"), k=10 to match
             Dreaddit's 10 subreddits, fit on TRAIN only. 06_probe.subreddit_control
             then works verbatim.
  wc         len(text.split()) -- exact analogue of Dreaddit's lex_liwc_WC.

The `conf` and `subreddit` keys are ALIASES, not misnomers. Keeping the names is
what lets 04/05/06/07/08 stay untouched.

usage:
  python 01_data.py --build          # download, filter, split, write jsonl
  python 01_data.py --stats          # report an existing build
"""

import argparse
import hashlib
import io
import json
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SUICIDE_DATA",
                          os.path.join(_here, "..", "data", "suicide_kaggle"))
DATA_DIR = os.path.abspath(DATA_DIR)

URL = ("https://www.kaggle.com/api/v1/datasets/download/"
       "rvarun11/suicidal-ideation-reddit-dataset")
CSV_NAME = "suicidal_ideation_reddit_annotated.csv"

# Measured on the raw file -- assert these so a silently changed upstream is caught.
RAW_SHAPE = (12656, 2)
RAW_BALANCE = {1: 6609, 0: 6047}

WC_MIN, WC_MAX = 10, 400   # posts run to 3868 words; cap so prompts stay tractable
TEST_FRAC = 0.20
SEED = 0
N_TOPICS = 10              # matches Dreaddit's 10 subreddits
N_ANNOTATORS = 5           # matches Dreaddit's conf grid granularity


# ------------------------------------------------------------------- download

def fetch_csv():
    """Download + extract once. Public dataset: no Kaggle credentials needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, CSV_NAME)
    if os.path.exists(dest):
        print(f"  using cached {dest}")
        return dest
    print(f"  downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    blob = urllib.request.urlopen(req, timeout=300).read()
    print(f"  got {len(blob) / 1e6:.1f} MB")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extract(CSV_NAME, DATA_DIR)
    return dest


# ------------------------------------------------------------------ proxies

def bagged_conf(texts, labels, seed=SEED):
    """
    conf PROXY -- five simulated lexical annotators.

    Dreaddit's conf is the fraction of human annotators agreeing with the gold
    label; its function in the pipeline is to select items whose ground truth is
    unambiguous, so the clean/corrupt patching contrast is a real contrast in the
    construct rather than annotator noise.

    Each "annotator" is one repeat of a stratified 5-fold TF-IDF + logistic
    regression, scored OUT OF FOLD, so no row is ever predicted by a model that
    saw it. conf = (# of the 5 agreeing with gold) / 5.

    The annotators are deliberately DECORRELATED -- each sees a 60% bootstrap of
    its training folds and a different feature space (n-gram range, vocabulary
    size, regularisation), because five identical strong classifiers agree with
    each other almost always.

    MEASURED: conf==1.0 on 0.82 of rows, against Dreaddit's 0.37. Decorrelating
    moved it only 0.88 -> 0.82, so this is a property of the DATA, not of the
    proxy: a TF-IDF baseline scores 0.878 accuracy / 0.948 AUROC on this task
    versus 0.715 on Dreaddit. Most items here really are lexically unambiguous
    ("I want to kill myself"), so the filter is permissive by necessity. It was
    NOT tuned down to match Dreaddit -- that would be manufacturing a number.
    The `ld[clean] - ld[corrupt] > sd` filter in 05_patch.build_pairs is
    model-specific and does the real selection work.

    Computed over train + test. That is acceptable ONLY because conf is used
    solely for ITEM SELECTION in patching -- never as a feature and never as a
    label. Say so in the report or a reader will assume leakage.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline

    # one config per annotator -- different views of the same text
    CONFIGS = [
        dict(max_features=3000,  ngram_range=(1, 1), min_df=2, C=0.1),
        dict(max_features=1500,  ngram_range=(1, 2), min_df=5, C=1.0),
        dict(max_features=5000,  ngram_range=(1, 1), min_df=3, C=0.03),
        dict(max_features=2000,  ngram_range=(2, 2), min_df=3, C=1.0),
        dict(max_features=4000,  ngram_range=(1, 2), min_df=2, C=0.3),
    ]
    assert len(CONFIGS) == N_ANNOTATORS

    y = np.asarray(labels)
    agree = np.zeros(len(y), dtype=int)
    for r, cfg in enumerate(CONFIGS):
        rng = np.random.default_rng(seed + r)
        C = cfg.pop("C")
        oof = np.zeros(len(y), dtype=int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + r)
        for tr, te in skf.split(np.zeros(len(y)), y):
            boot = rng.choice(tr, int(len(tr) * 0.60), replace=False)
            clf = make_pipeline(TfidfVectorizer(**cfg),
                                LogisticRegression(C=C, max_iter=2000))
            clf.fit([texts[i] for i in boot], y[boot])
            oof[te] = clf.predict([texts[i] for i in te])
        cfg["C"] = C
        agree += (oof == y).astype(int)
        print(f"    annotator {r + 1}/{N_ANNOTATORS} {cfg}: "
              f"oof acc {(oof == y).mean():.3f}", flush=True)
    return agree / N_ANNOTATORS


def topic_clusters(train_texts, test_texts, k=N_TOPICS, seed=SEED):
    """
    subreddit PROXY -- unsupervised topic id, fit on TRAIN ONLY.

    Caveat for the report: the clusters are DEFINED lexically, so a probe on a
    contextualised representation finds them more easily than it would find true
    metadata. The control therefore OVER-estimates topic encodability, which is
    the conservative direction for a leakage bound.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 1), min_df=3)
    Xtr = vec.fit_transform(train_texts)
    svd = TruncatedSVD(n_components=64, random_state=seed)
    Ztr = svd.fit_transform(Xtr)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Ztr)
    Zte = svd.transform(vec.transform(test_texts))
    return ([f"topic_{c:02d}" for c in km.labels_],
            [f"topic_{c:02d}" for c in km.predict(Zte)])


def mutual_info(clusters, labels):
    """MI(cluster; label) in bits -- bounds how much the topic proxy can explain."""
    from sklearn.metrics import mutual_info_score
    return float(mutual_info_score(clusters, labels) / np.log(2))


# --------------------------------------------------------------------- build

def build():
    csv = fetch_csv()
    df = pd.read_csv(csv)
    print(f"  raw {df.shape} {dict(df.label.value_counts())}")
    assert df.shape == RAW_SHAPE, f"upstream changed: {df.shape} != {RAW_SHAPE}"
    assert dict(df.label.value_counts()) == RAW_BALANCE, "upstream balance changed"

    df["text"] = df.usertext.astype(str).str.strip()
    df = df[df.text.str.len() > 0]

    n0 = len(df)
    key = df.text.str.lower().str.replace(r"\s+", " ", regex=True)
    df = df[~key.duplicated()]
    print(f"  exact dedup: {len(df)} (dropped {n0 - len(df)})")

    df["wc"] = df.text.str.split().str.len()
    n1 = len(df)
    df = df[(df.wc >= WC_MIN) & (df.wc <= WC_MAX)].reset_index(drop=True)
    print(f"  wc in [{WC_MIN},{WC_MAX}]: {len(df)} (dropped {n1 - len(df)}), "
          f"balance {df.label.mean():.3f}")

    # stratified split on label x wc-quartile, mirroring 00_gate.load_eval's own
    # length-stratified subsample so the two never fight each other
    rng = np.random.default_rng(SEED)
    q = np.digitize(df.wc, np.quantile(df.wc, [.25, .5, .75]))
    test_idx = []
    for lab in (0, 1):
        for b in range(4):
            pool = np.flatnonzero((df.label.values == lab) & (q == b))
            take = int(round(len(pool) * TEST_FRAC))
            test_idx += list(rng.choice(pool, take, replace=False))
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[test_idx] = True

    print("  conf proxy (5 bagged lexical annotators, out-of-fold)", flush=True)
    df["conf"] = bagged_conf(df.text.tolist(), df.label.values)

    print("  topic proxy (TF-IDF -> SVD64 -> KMeans10, fit on train)", flush=True)
    tr_topics, te_topics = topic_clusters(
        df.text[~test_mask].tolist(), df.text[test_mask].tolist())
    df.loc[~test_mask, "subreddit"] = tr_topics
    df.loc[test_mask, "subreddit"] = te_topics

    cols = ["text", "label", "conf", "subreddit", "wc"]
    out = {}
    for split, mask in [("train", ~test_mask), ("test", test_mask)]:
        sub = df.loc[mask, cols].reset_index(drop=True)
        sub["label"] = sub.label.astype(int)
        sub["wc"] = sub.wc.astype(float)
        path = os.path.join(DATA_DIR, f"split_{split}.jsonl")
        sub.to_json(path, orient="records", lines=True)
        out[split] = sub
        print(f"  -> {path}  n={len(sub)}  balance={sub.label.mean():.3f}  "
              f"conf==1.0 {(sub.conf >= 1.0).mean():.3f}")

    man = {
        "source": URL, "seed": SEED, "wc_range": [WC_MIN, WC_MAX],
        "test_frac": TEST_FRAC, "n_topics": N_TOPICS, "n_annotators": N_ANNOTATORS,
        "n_train": len(out["train"]), "n_test": len(out["test"]),
        "sha256": {s: hashlib.sha256(
            open(os.path.join(DATA_DIR, f"split_{s}.jsonl"), "rb").read()
        ).hexdigest() for s in ("train", "test")},
        "mi_topic_label_bits": {
            s: mutual_info(out[s].subreddit.tolist(), out[s].label.tolist())
            for s in ("train", "test")},
    }
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"\n  MI(topic; label) train {man['mi_topic_label_bits']['train']:.4f} bits, "
          f"test {man['mi_topic_label_bits']['test']:.4f} bits")
    return out


# ---------------------------------------------------------------------- load

def load_rows(split):
    """The one entry point 00_gate.py and 04_cache.py both use."""
    path = os.path.join(DATA_DIR, f"split_{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing -- run `python 01_data.py --build`")
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"text": r["text"], "label": int(r["label"]),
                         "conf": float(r["conf"]), "subreddit": r["subreddit"],
                         "wc": float(r["wc"])})
    return rows


def stats():
    for split in ("train", "test"):
        rows = load_rows(split)
        wc = np.array([r["wc"] for r in rows])
        conf = np.array([r["conf"] for r in rows])
        lab = np.array([r["label"] for r in rows])
        print(f"{split:6s} n={len(rows):6d} balance={lab.mean():.3f} "
              f"conf==1.0 {(conf >= 1.0).mean():.3f} "
              f"wc med={np.median(wc):.0f} p95={np.quantile(wc, .95):.0f} "
              f"max={wc.max():.0f}")
        print(f"       conf grid {dict(zip(*np.unique(conf, return_counts=True)))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    print(f"data dir -> {DATA_DIR}")
    if args.build:
        build()
    if args.stats or not args.build:
        stats()


if __name__ == "__main__":
    main()
