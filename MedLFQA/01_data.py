#!/usr/bin/env python3
"""
01_data.py -- Long-form medical QA (dmis-lab/MedLFQA).

TASK. MedLFQA has no native binary label: each row is a Question, a
Free_form_answer, and two expert-annotated statement lists -- `Must_have` (facts a
correct answer MUST contain) and `Nice_to_have` (facts that are useful but
optional). The essentiality judgement between those two lists IS the decision
variable:

    label 1 = the shown statement came from Must_have   (essential)
    label 0 = the shown statement came from Nice_to_have (optional)
    LD = logit("Yes") - logit("No")

One must-have and one nice-to-have statement are drawn from the SAME question, so
the two labels share a question, a topic and a vocabulary. That is what makes this
framing safe: the alternative (negatives borrowed from other questions) would make
the negatives off-topic and hand a bag-of-words probe the answer for free -- the
failure that took several rebuilds to neutralise on the psychology MCQ task. Here
topic cannot leak by construction, and 05_patch gets true minimal pairs.

Statement lengths are near-balanced in the source (must 16 words median vs nice
17), but that is not relied on -- WC_TOL enforces per-item length parity and the
audit reports the residual.

Emits the Dreaddit row schema so nothing downstream changes:
  text / label / conf / subreddit / wc / qid   (`conf` and `subreddit` are
  ALIASES carrying task-specific proxies.)

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
    "MEDLFQA_DATA", os.path.join(_here, "..", "data", "medlfqa")))

HF_ID = "dmis-lab/MedLFQA"
N_TEST_ROWS = 200          # the evaluated set, as specified
N_TRAIN_ITEMS = 900        # disjoint probe-training pool
WC_TOL = 6                 # per-item statement length parity
MIN_WORDS = 5
SEED = 0
N_TOPICS = 10


def content_words(t):
    import re
    return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 2}


def build():
    from datasets import load_dataset
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    os.makedirs(DATA_DIR, exist_ok=True)
    d = load_dataset(HF_ID)["test"]
    print(f"  corpus n={len(d)}")

    rng = np.random.default_rng(SEED)
    items, drop_len, drop_empty = [], 0, 0
    for i in range(len(d)):
        q = str(d["Question"][i]).strip()
        must = [s.strip() for s in d["Must_have"][i]
                if len(str(s).split()) >= MIN_WORDS]
        nice = [s.strip() for s in d["Nice_to_have"][i]
                if len(str(s).split()) >= MIN_WORDS]
        if not q or not must or not nice:
            drop_empty += 1
            continue
        # pick the must/nice pair with the closest word counts, then require the
        # residual gap to be small -- otherwise "the longer statement is essential"
        # becomes a free shortcut
        best = min(((m, n) for m in must for n in nice),
                   key=lambda p: abs(len(p[0].split()) - len(p[1].split())))
        if abs(len(best[0].split()) - len(best[1].split())) > WC_TOL:
            drop_len += 1
            continue
        items.append({"qid": i, "q": q, "must": best[0], "nice": best[1]})
    print(f"  usable items: {len(items)} (dropped {drop_empty} missing a list, "
          f"{drop_len} failing |wc| <= {WC_TOL} parity)")

    order = rng.permutation(len(items))
    n_test_items = N_TEST_ROWS // 2
    sel = order[: n_test_items + N_TRAIN_ITEMS]

    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2)
    X = vec.fit_transform([items[i]["q"] for i in sel])
    Z = TruncatedSVD(n_components=min(64, X.shape[1] - 1),
                     random_state=SEED).fit_transform(X)
    km = KMeans(n_clusters=N_TOPICS, random_state=SEED, n_init=10).fit_predict(Z)
    topic = {int(i): f"topic_{c:02d}" for i, c in zip(sel, km)}

    rows = {"test": [], "train": []}
    for rank, i in enumerate(sel):
        it = items[int(i)]
        split = "test" if rank < n_test_items else "train"
        for label, st in ((1, it["must"]), (0, it["nice"])):
            rows[split].append({
                "text": f"Question: {it['q']}\n\nStatement: {st}",
                "label": label,
                # no per-item human confidence in this corpus; both statements come
                # from the same expert annotation pass, so every item is treated as
                # unambiguous and the |dLD| > 1sd filter in 05_patch does the real
                # pair selection
                "conf": 1.0,
                "subreddit": topic[int(i)],
                "wc": float(len(it["q"].split()) + len(st.split())),
                "qid": it["qid"]})

    for split in ("train", "test"):
        p = os.path.join(DATA_DIR, f"split_{split}.jsonl")
        with open(p, "w") as f:
            for r in rows[split]:
                f.write(json.dumps(r) + "\n")
        y = np.array([r["label"] for r in rows[split]])
        print(f"  -> {p}  n={len(y)} ({len(y)//2} items)  balance={y.mean():.3f}")

    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump({"source": HF_ID, "seed": SEED, "wc_tol": WC_TOL,
                   "n_test_rows": N_TEST_ROWS, "n_train_items": N_TRAIN_ITEMS,
                   "n_usable_items": len(items),
                   "sha256": {s: hashlib.sha256(open(
                       os.path.join(DATA_DIR, f"split_{s}.jsonl"), "rb").read()
                   ).hexdigest() for s in ("train", "test")}}, f, indent=2)
    audit(rows["test"])


def audit(rows=None):
    from sklearn.metrics import roc_auc_score
    if rows is None:
        rows = load_rows("test")
    y = np.array([r["label"] for r in rows])
    st = [r["text"].split("\n\nStatement: ", 1)[1] for r in rows]
    qs = [r["text"].split("\n\nStatement: ", 1)[0] for r in rows]
    a_len = roc_auc_score(y, [len(s.split()) for s in st])
    a_ov = roc_auc_score(y, [len(content_words(s) & content_words(q))
                             for s, q in zip(st, qs)])
    print(f"\n  AUDIT  longer-statement-is-essential  AUROC {a_len:.3f}  "
          f"(want 0.50 +/- 0.05)")
    print(f"  AUDIT  question-overlap               AUROC {a_ov:.3f}  (want < 0.65)")
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
