#!/usr/bin/env python3
"""
01_data.py -- Medical MCQ (dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust).

TASK. The corpus is native 4-option USMLE multiple choice: `options` {A,B,C,D}
plus `answer_idx` giving the correct letter. It is reduced to the same
counterfactually paired two-option forced choice used for the psychology MCQ task,
so the four metrics stay definitionally identical across tasks:

    the gold option and ONE distractor are shown as A and B
    every question is rendered TWICE -- gold in A, then gold in B
    label 1 = gold is in slot A;   LD = logit("A") - logit("B")

Rendering both slot orders is what makes the layer-0 probe floor PROVABLY 0.500:
04_cache.py reads rep[0] from a pre-hook on layers[0], i.e. the raw embedding
output before any attention, and these are RoPE models with no additive positional
embedding -- so the mean-pool over the option span is a pure bag-of-words centroid
and swapping the two option bodies leaves the token multiset identical. A layer-0
probe therefore cannot beat chance, and anything above ~0.55 is a bug rather than
a finding. It also drives the TF-IDF floor to ~0.50 and MI(topic; label) to exactly
0, and hands 05_patch true minimal pairs.

Unlike the psychology task the distractors are NOT synthesized -- they are the
exam's own hand-written options, so there is no risk of a distractor secretly
being correct. The remaining risk is that the gold and the chosen distractor
differ systematically in surface form, so the distractor is picked for length and
question-overlap parity and the residual is audited.

`subreddit` carries the corpus's own `usmle_system` field (15 medical systems) --
a REAL topic label rather than a clustering proxy, which makes the topic control
sharper here than on any other task in this study.

The adversarial context columns (misleading_context / alternative_context /
edge_case_context) are deliberately NOT used: they change the prompt rather than
the decision variable, and mixing them in would confound the interpretability
metrics with robustness. The clean question is what gets rendered.

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
    "MEDQA_DATA", os.path.join(_here, "..", "data", "medqa")))

HF_ID = "dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust"
N_TEST_ROWS = 200          # the evaluated set, as specified
WC_TOL = 5                 # gold/distractor length parity
SEED = 0


def content_words(t):
    import re
    return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 2}


def build():
    from datasets import load_dataset

    os.makedirs(DATA_DIR, exist_ok=True)
    d = load_dataset(HF_ID)["train"]
    print(f"  corpus n={len(d)}")

    rng = np.random.default_rng(SEED)
    items, dropped, bal = [], 0, 0
    for i in range(len(d)):
        q = str(d["question"][i]).strip()
        opts = d["options"][i]
        gold_letter = str(d["answer_idx"][i]).strip()
        if not q or gold_letter not in opts:
            dropped += 1
            continue
        gold = str(opts[gold_letter]).strip()
        others = [str(v).strip() for k, v in opts.items()
                  if k != gold_letter and str(v).strip()]
        if not others:
            dropped += 1
            continue
        gw, qcw = len(gold.split()), content_words(q)
        g_ov = len(content_words(gold) & qcw)
        # Pick the distractor closest to the gold on BOTH surface statistics, so
        # neither "the longer option" nor "the option sharing more words with the
        # question" identifies the answer.
        #
        # Closest-on-average is not enough on its own: 65% of items tie exactly on
        # length, and the non-tied remainder skewed one way, leaving a +0.19-word
        # mean gap and a length AUROC of 0.579 on the 1898-row train split -- small
        # but systematic across both splits, so noise is ruled out. `bal` carries
        # the running signed gap and, among candidates that are equally good on the
        # primary criterion, breaks the tie toward whichever sign pulls the dataset
        # back to zero. The residual then cancels ACROSS items rather than needing
        # a tighter per-item tolerance that would cost yield.
        def _key(o, _bal=None):
            dw = len(o.split()) - gw
            prim = abs(dw) + abs(len(content_words(o) & qcw) - g_ov)
            return (prim, dw * np.sign(bal) if bal else abs(dw))

        pick = min(others, key=_key)
        if abs(len(pick.split()) - gw) > WC_TOL:
            dropped += 1
            continue
        bal += len(pick.split()) - gw
        items.append({"qid": i, "q": q, "gold": gold, "dist": pick,
                      "system": str(d["usmle_system"][i]).strip() or "unknown"})
    print(f"  usable items: {len(items)} (dropped {dropped})")

    order = rng.permutation(len(items))
    n_test_items = N_TEST_ROWS // 2

    rows = {"test": [], "train": []}
    for rank, i in enumerate(order):
        it = items[int(i)]
        split = "test" if rank < n_test_items else "train"
        for label, (oa, ob) in ((1, (it["gold"], it["dist"])),
                                (0, (it["dist"], it["gold"]))):
            rows[split].append({
                "text": f"Question: {it['q']}\n\nA. {oa}\nB. {ob}",
                "label": label,
                # the options are exam-authored, so every retained item is treated
                # as unambiguous; the |dLD| > 1sd filter in 05_patch.build_pairs
                # does the real pair selection
                "conf": 1.0,
                "subreddit": it["system"],      # REAL topic label, not a proxy
                "wc": float(len(it["q"].split()) + len(oa.split()) + len(ob.split())),
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
                   "n_test_rows": N_TEST_ROWS, "n_usable_items": len(items),
                   "sha256": {s: hashlib.sha256(open(
                       os.path.join(DATA_DIR, f"split_{s}.jsonl"), "rb").read()
                   ).hexdigest() for s in ("train", "test")}}, f, indent=2)
    audit(rows["test"])


def audit(rows=None):
    from sklearn.metrics import roc_auc_score
    if rows is None:
        rows = load_rows("test")
    y = np.array([r["label"] for r in rows])

    def opts(t):
        a, b = t.split("\nA. ", 1)[1].split("\nB. ", 1)
        return a, b

    lg, ov = [], []
    for r in rows:
        oa, ob = opts(r["text"])
        q = r["text"].split("\n\nA. ")[0]
        cq = content_words(q)
        lg.append(len(oa.split()) - len(ob.split()))
        ov.append(len(content_words(oa) & cq) - len(content_words(ob) & cq))
    a_len, a_ov = roc_auc_score(y, lg), roc_auc_score(y, ov)
    sgn = lambda v: float(np.mean([x if yy == 1 else -x for x, yy in zip(v, y)]))
    print(f"\n  AUDIT  longer-option-is-gold   AUROC {a_len:.3f}  (want 0.50 +/- 0.05)"
          f"  signed gap {sgn(lg):+.2f} words")
    print(f"  AUDIT  question-overlap        AUROC {a_ov:.3f}  (want < 0.65)"
          f"  signed gap {sgn(ov):+.2f} words")
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
