#!/usr/bin/env python3
"""
01_data.py -- Answer-generation / hallucination detection (UTAustin-AIHealth/MedHallu,
pqa_labeled config).

TASK. Each source row carries one Question, one human-written `Ground Truth`
answer, and one `Hallucinated Answer`. Both are rendered against the SAME
question, giving a counterfactually paired binary task:

    label 1 = the shown answer is the hallucinated one
    label 0 = the shown answer is the ground truth
    LD = logit("Yes") - logit("No")

Pairing both answers to one question means 05_patch gets true minimal pairs
(clean and corrupt differ only in the answer body) and the topic is identical
across the two labels, so topic cannot leak.

THE LENGTH SHORTCUT -- the reason this file filters at all.
Ground-truth answers in this corpus run 37 words at the median; hallucinated
answers run 23. That +11.9-word mean gap is an artifact of how the hallucinations
were generated, and it is enormous: a classifier that only counts words scores
AUROC 0.740, i.e. most of the "hallucination detection" signal available in the
raw corpus is just length. Measured length-only AUROC by tolerance:

    |gap| <= 3   166 items   AUROC 0.523
    |gap| <= 5   233 items   AUROC 0.534     <- used
    |gap| <= 8   340 items   AUROC 0.584
    |gap| <= 10  397 items   AUROC 0.602
    unfiltered  1000 items   AUROC 0.740

WC_TOL = 5 is the loosest setting that keeps length near chance while still
leaving a 200-row test split. It costs 77% of the corpus, which is the right
trade: without it every downstream number would be a statement about answer
length, not about hallucination.

Emits the Dreaddit row schema so nothing downstream changes:
  text / label / conf / subreddit / wc / qid   (`conf` and `subreddit` are
  ALIASES carrying task-specific proxies -- see below, and the docstring in
  ../Suicide/01_data.py for why the names are kept.)

usage:
  python 01_data.py --build
  python 01_data.py --audit
"""

import argparse
import hashlib
import json
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.environ.get(
    "MEDHALLU_DATA", os.path.join(_here, "..", "data", "medhallu")))

HF_ID, HF_CONFIG = "UTAustin-AIHealth/MedHallu", "pqa_labeled"
N_TEST_ROWS = 200          # the evaluated set, as specified
WC_TOL = 5                 # see the length-shortcut note above
SEED = 0
N_TOPICS = 10

# Difficulty Level is a human annotation of how hard the item is to call -- the
# closest thing this corpus has to Dreaddit's `confidence`. It is RETAINED per row
# (and in the manifest) as a covariate, but it is NOT used as conf: mapping
# easy->1.0 / medium->0.6 / hard->0.2 left only 26% of items at conf == 1.0, i.e.
# ~26 matched pairs out of a 100-item test split, which is too few for a stable
# patching recovery curve. conf is set to 1.0 throughout, matching MedQA and
# MedLFQA, and the model-specific |dLD| > 1sd filter in 05_patch.build_pairs does
# the real pair selection.
DIFFICULTY_CONF = {"easy": 1.0, "medium": 1.0, "hard": 1.0}


def content_words(t):
    import re
    return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 2}


def build():
    from datasets import load_dataset
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    os.makedirs(DATA_DIR, exist_ok=True)
    d = load_dataset(HF_ID, HF_CONFIG)["train"]
    print(f"  corpus n={len(d)}")

    q = [str(x).strip() for x in d["Question"]]
    gt = [str(x).strip() for x in d["Ground Truth"]]
    ha = [str(x).strip() for x in d["Hallucinated Answer"]]
    diff = [str(x).strip() for x in d["Difficulty Level"]]

    gap = np.array([len(a.split()) - len(b.split()) for a, b in zip(gt, ha)])
    keep = np.flatnonzero(np.abs(gap) <= WC_TOL)
    print(f"  |wc(gt) - wc(hall)| <= {WC_TOL}: {len(keep)} items kept "
          f"(dropped {len(d) - len(keep)}), mean signed gap "
          f"{gap[keep].mean():+.2f} words")

    rng = np.random.default_rng(SEED)
    keep = rng.permutation(keep)

    # topic proxy: KMeans over question text (no native topic field here)
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2)
    X = vec.fit_transform([q[i] for i in keep])
    Z = TruncatedSVD(n_components=min(64, X.shape[1] - 1),
                     random_state=SEED).fit_transform(X)
    km = KMeans(n_clusters=N_TOPICS, random_state=SEED, n_init=10).fit_predict(Z)
    topic = {int(i): f"topic_{c:02d}" for i, c in zip(keep, km)}

    n_test_items = N_TEST_ROWS // 2
    rows = {"test": [], "train": []}
    for rank, i in enumerate(keep):
        i = int(i)
        split = "test" if rank < n_test_items else "train"
        conf = DIFFICULTY_CONF.get(diff[i], 1.0)
        for label, ans in ((1, ha[i]), (0, gt[i])):
            rows[split].append({
                "text": f"Question: {q[i]}\n\nAnswer: {ans}",
                "label": label, "conf": conf, "subreddit": topic[i],
                "wc": float(len(q[i].split()) + len(ans.split())),
                "qid": i, "difficulty": diff[i]})

    for split in ("train", "test"):
        p = os.path.join(DATA_DIR, f"split_{split}.jsonl")
        with open(p, "w") as f:
            for r in rows[split]:
                f.write(json.dumps(r) + "\n")
        y = np.array([r["label"] for r in rows[split]])
        c = np.array([r["conf"] for r in rows[split]])
        print(f"  -> {p}  n={len(y)} ({len(y)//2} items)  balance={y.mean():.3f}  "
              f"conf==1.0 {(c >= 1.0).mean():.3f}")

    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump({"source": f"{HF_ID}:{HF_CONFIG}", "seed": SEED,
                   "wc_tol": WC_TOL, "n_test_rows": N_TEST_ROWS,
                   "n_items_kept": len(keep),
                   "difficulty_counts": {k: int(sum(1 for i in keep
                                                    if diff[int(i)] == k))
                                         for k in ("easy", "medium", "hard")},
                   "mean_signed_wc_gap": float(gap[keep].mean()),
                   "sha256": {s: hashlib.sha256(open(
                       os.path.join(DATA_DIR, f"split_{s}.jsonl"), "rb").read()
                   ).hexdigest() for s in ("train", "test")}}, f, indent=2)
    audit(rows["test"])


def audit(rows=None):
    """Both shortcuts a bag-of-words probe could exploit instead of the construct."""
    from sklearn.metrics import roc_auc_score
    if rows is None:
        rows = load_rows("test")
    y = np.array([r["label"] for r in rows])
    ans = [r["text"].split("\n\nAnswer: ", 1)[1] for r in rows]
    qs = [r["text"].split("\n\nAnswer: ", 1)[0] for r in rows]
    a_len = roc_auc_score(y, [-len(a.split()) for a in ans])
    a_ov = roc_auc_score(y, [len(content_words(a) & content_words(q))
                             for a, q in zip(ans, qs)])
    print(f"\n  AUDIT  shorter-answer-is-hallucinated  AUROC {a_len:.3f}  "
          f"(want 0.50 +/- 0.05)")
    print(f"  AUDIT  question-overlap                AUROC {a_ov:.3f}  (want < 0.65)")
    if abs(a_len - 0.5) > 0.05:
        print("  *** length shortcut present -- lower WC_TOL ***")
    if a_ov >= 0.65:
        print("  *** question-overlap shortcut present ***")
    return a_len, a_ov


def load_rows(split):
    """The one entry point 00_gate.py and 04_cache.py both use."""
    p = os.path.join(DATA_DIR, f"split_{split}.jsonl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing -- run `python 01_data.py --build`")
    out = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            out.append({"text": r["text"], "label": int(r["label"]),
                        "conf": float(r["conf"]), "subreddit": r["subreddit"],
                        "wc": float(r["wc"]), "qid": int(r["qid"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    print(f"data dir -> {DATA_DIR}")
    if a.build:
        build()
    if a.audit:
        audit()


if __name__ == "__main__":
    main()
