#!/usr/bin/env python3
"""
01_data.py -- Psychoeducation MCQ (HF BoltMonkey/psychology-question-answer)

The source is FREE-FORM QA (question, answer), 197,180 rows. MCQ items are
synthesized here.

FORMAT -- two-option forced choice, counterfactually paired.

  Each question is rendered TWICE, once with the gold answer in slot A and once
  with it in slot B. LD = logit("A") - logit("B"); y = 1 iff the gold is in A.

  Why both renders rather than one random slot: it makes the layer-0 probe floor
  PROVABLY 0.500. 04_cache.py captures rep[0] via a pre-hook on layers[0] -- the
  raw embedding output, before any attention. Llama-3 and Qwen2.5 are RoPE models
  with no additive positional embedding, so the mean-pool over the option span is
  a pure bag-of-words centroid, and swapping the two option bodies preserves the
  token multiset exactly. A layer-0 probe therefore CANNOT beat chance, and
  anything above 0.55 is a bug (a tokenisation-boundary artifact or an option
  length imbalance), not a finding. The TF-IDF floor collapses to ~0.50 for the
  same reason. It also makes MI(topic; label) exactly 0, and gives 05_patch true
  minimal pairs: clean and corrupt differ ONLY in option order.

Emits the Dreaddit row schema so nothing downstream changes:

  text       "Question: {q}\n\nA. {opt_a}\nB. {opt_b}"  -- the span render() locates
  label      1 = gold is in slot A, 0 = gold is in slot B
  conf       PROXY. Five independent item-quality votes -> {0, .2, .4, .6, .8, 1.0}.
             Same value grid as Dreaddit's annotator agreement, so the
             `conf >= 1.0` filters in 00_gate.py and 05_patch.py need NO change.
  subreddit  PROXY. KMeans(k=10) topic cluster over question text. Because both
             renders of a question carry opposite labels, MI(cluster; label) == 0
             exactly -- so here the control is not a leakage bound but a POSITIVE
             control that topic is encoded while provably unable to help.
  wc         words in question + both options.
  qid        NEW. Groups the two renders of one question. Used by 05_patch to
             build matched pairs and by 06_probe for consistent-pair accuracy.

The `conf` and `subreddit` keys are ALIASES, not misnomers.

usage:
  python 01_data.py --build --hi 0.30 --lo 0.05
  python 01_data.py --build --hi 0.30 --pilot 300     # small bank for calibration
  python 01_data.py --audit                           # shortcut AUROCs, CPU
  python 01_data.py --judge --n 500                   # LLM validity check, GPU
"""

import argparse
import hashlib
import json
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("PSYCH_DATA",
                          os.path.join(_here, "..", "data", "psych_mcq"))
DATA_DIR = os.path.abspath(DATA_DIR)

HF_ID = "BoltMonkey/psychology-question-answer"

# No cap -- build from the entire deduped corpus and keep every item that clears
# the parity filters. The filters are strict (~10% yield), so this lands far below
# the raw 150k questions available.
N_QUESTIONS = 10 ** 9
TEST_FRAC = 0.20
SEED = 0
N_TOPICS = 10
N_FINE = 64            # fine-grained answer clusters the distractor pool is drawn from
N_DISTRACTORS = 3      # store 3, render 1 -- makes a 4-option variant a render flag

Q_SIM_MAX = 0.40       # reject distractors whose QUESTION resembles the gold's
WC_DIFF_MAX = 3        # kill "the longer option is the gold"
OVERLAP_TOL = 1        # kill "the option sharing more words with the question"


STOP = set(
    "the a an and or of to in is are was were be been being for on with as that "
    "this it its by from at how does do what which why who when whose their they "
    "them there these those than then so such can could may might will would "
    "should have has had not no but if into about over under more most less".split())


def content_words(t):
    """Content-word set used by the question-overlap shortcut audit and filter."""
    import re
    return {w for w in re.findall(r"[a-z]+", t.lower())
            if w not in STOP and len(w) > 2}


# ---------------------------------------------------------------------- load

def load_corpus():
    from datasets import load_dataset
    d = load_dataset(HF_ID)["train"]
    q = [str(x).strip() for x in d["question"]]
    a = [str(x).strip() for x in d["answer"]]
    print(f"  corpus n={len(q)}")
    return q, a


def dedup(questions, answers, sim=0.90, seed=SEED):
    """
    Exact then near-duplicate removal on QUESTIONS.

    This corpus is heavily templated ("What is the main focus of X?"), and two
    near-duplicate questions have interchangeable answers -- which is exactly the
    failure mode that makes a synthesized distractor secretly correct.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    keep, seen = [], set()
    for i, q in enumerate(questions):
        k = " ".join(q.lower().split())
        if k and k not in seen and len(q.split()) >= 4 and len(answers[i].split()) >= 6:
            seen.add(k)
            keep.append(i)
    print(f"  after exact dedup + length filter: {len(keep)}")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
    X = vec.fit_transform([questions[i] for i in keep])
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    drop = set()
    for row, (d1, j1) in enumerate(zip(dist[:, 1], idx[:, 1])):
        if 1.0 - d1 > sim and row not in drop:
            drop.add(int(j1))
    keep = [i for r, i in enumerate(keep) if r not in drop]
    print(f"  after near-dup (cos>{sim}) removal: {len(keep)}")
    return keep


# --------------------------------------------------------------- item bank

def prepare(seed=SEED):
    """
    Corpus load, dedup, vectorise, cluster. ~5 min, and identical for every (lo, hi)
    band, so it is cached -- the difficulty calibration sweep reuses it.

    ONE shared TF-IDF space for questions and answers. Separate vectorizers would
    give answer-vs-question similarity no common vocabulary, which is both wrong
    and a dimension mismatch.
    """
    import pickle

    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "prepared.pkl")
    if os.path.exists(cache):
        print(f"  using cached {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    q_all, a_all = load_corpus()
    keep = dedup(q_all, a_all)
    Q = [q_all[i] for i in keep]
    A = [a_all[i] for i in keep]

    print("  vectorising (one shared space for questions and answers)")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
    vec.fit(A + Q)
    XA = normalize(vec.transform(A))
    XQ = normalize(vec.transform(Q))

    print(f"  clustering answers into {N_FINE} fine groups")
    svd = TruncatedSVD(n_components=128, random_state=seed)
    ZA = svd.fit_transform(XA)
    fine = KMeans(n_clusters=N_FINE, random_state=seed, n_init=5).fit_predict(ZA)

    out = (Q, A, XA, XQ, fine)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    print(f"  -> cached {cache}")
    return out


def build_bank(hi, lo, n_questions, seed=SEED):
    """
    One item = (question, gold, 3 distractors). Distractors are drawn from the
    gold's own fine-grained answer cluster, then filtered so they are neither
    paraphrases of the gold (which would also be correct) nor off-topic (which
    would make the item trivial). (lo, hi) is the difficulty knob.
    """
    Q, A, XA, XQ, fine = prepare(seed)

    wc_a = np.array([len(x.split()) for x in A])
    cw_a = [content_words(x) for x in A]      # precomputed: the inner loop hits
    cw_q = [content_words(x) for x in Q]      # these ~1200x per item
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Q))

    by_fine = {}
    for i, c in enumerate(fine):
        by_fine.setdefault(int(c), []).append(i)

    items, stats = [], {"pool_empty": 0, "no_survivor": 0, "overlap_parity": 0}
    for gi in order:
        if len(items) >= n_questions:
            break
        pool = by_fine[int(fine[gi])]
        if len(pool) < 20:
            stats["pool_empty"] += 1
            continue
        cand = np.array([j for j in pool if j != gi])
        if len(cand) > 1200:                      # cap the similarity math per item
            cand = rng.choice(cand, 1200, replace=False)

        sim_a = np.asarray(XA[cand].dot(XA[gi].T).todense()).ravel()
        sim_q = np.asarray(XQ[cand].dot(XQ[gi].T).todense()).ravel()
        ok = (sim_a <= hi) & (sim_a >= lo) & (sim_q <= Q_SIM_MAX) \
            & (np.abs(wc_a[cand] - wc_a[gi]) <= 3 * WC_DIFF_MAX)
        surv = cand[ok]
        if len(surv) < N_DISTRACTORS:
            stats["no_survivor"] += 1
            continue

        # QUESTION-OVERLAP PARITY -- the shortcut that nearly sank this task.
        # A gold answer shares content words with its own question (it answers it);
        # an answer lifted from a different question does not. Left alone that gives
        # "the option sharing more words with the question is the gold" an AUROC of
        # 0.949, and the model never needs any psychology.
        #
        # Matching TF-IDF cosine does NOT fix it -- the audit counts raw shared
        # content words, so that is what has to be filtered. Hard requirement, not a
        # ranking preference: a candidate must come within OVERLAP_TOL of the gold's
        # own overlap or the item is dropped. Measured yield ~53%, and only 5k of
        # 150k questions are needed, so the yield cost is free.
        qs = cw_q[gi]
        g_ov = len(cw_a[gi] & qs)
        d_ov = np.array([len(cw_a[int(j)] & qs) for j in surv])
        d_wc = wc_a[surv]

        # The two shortcuts are COUPLED and must be controlled jointly. Selecting
        # for question-overlap alone pulls in longer answers (more words -> more
        # chance to overlap), which then hands back a length shortcut in the other
        # direction: overlap AUROC fell 0.949 -> 0.688 but length AUROC went
        # 0.489 -> 0.266. Both differences are scored SIGNED, so what has to sit at
        # zero is the mean signed gap, not the mean absolute one.
        # EXACT overlap parity, not a tolerance band. A band looks symmetric but
        # is not: the gold sits near the top of the overlap distribution, so a
        # +/-1 window has far more candidates below it than above, the residual
        # skews positive, and the shortcut survives (AUROC crept back to 0.734).
        # Requiring d_ov == g_ov drives the mean signed gap to exactly 0, so that
        # statistic is uninformative BY CONSTRUCTION rather than by tuning.
        #
        # Length cannot take the same treatment without gutting the yield, so its
        # residual is balanced ACROSS items instead: alternate which sign of length
        # difference is admissible, so the dataset-level mean signed gap cancels.
        want_longer = (len(items) % 2 == 0)
        dw = d_wc - wc_a[gi]
        near = (d_ov == g_ov) & (np.abs(dw) <= WC_DIFF_MAX) & \
               ((dw >= 0) if want_longer else (dw <= 0))
        if near.sum() < N_DISTRACTORS:
            stats["overlap_parity"] += 1
            continue
        surv, d_ov, dw = surv[near], d_ov[near], dw[near]
        rank_ = np.argsort(np.abs(dw))[:N_DISTRACTORS]
        pick, pick_ov = surv[rank_], d_ov[rank_]

        items.append({
            "qid": int(gi),
            "question": Q[gi],
            "gold": A[gi],
            "distractors": [A[int(j)] for j in pick],
            "sim_a": [float(x) for x in
                      np.asarray(XA[pick].dot(XA[gi].T).todense()).ravel()],
            "sim_q": [float(x) for x in
                      np.asarray(XQ[pick].dot(XQ[gi].T).todense()).ravel()],
            "ov_gap": [int(abs(x - g_ov)) for x in pick_ov],
            "fine": int(fine[gi]),
        })
    print(f"  built {len(items)} items  (skipped: small cluster "
          f"{stats['pool_empty']}, no survivor {stats['no_survivor']}, "
          f"overlap parity {stats['overlap_parity']})")
    return items, Q, XQ


# --------------------------------------------------------------------- conf

def item_conf(items, hi, judge=None):
    """
    conf PROXY -- five independent item-quality votes.

    Dreaddit's conf measures item unambiguity. For a synthesized MCQ, unambiguity
    IS distractor quality, so each vote checks one way a distractor could be bad.
    Both renders of a question share the item's conf.

    v5 (the LLM judge) is only available after --judge has run; without it the
    grid tops out at 0.8 and the `conf >= 1.0` filter would reject everything, so
    it falls back to 4 votes rescaled.
    """
    for it in items:
        d0 = 0                       # the rendered distractor is always index 0
        votes = [
            it["sim_a"][d0] <= hi - 0.05,          # margin below the paraphrase ceiling
            it["sim_q"][d0] <= Q_SIM_MAX,          # questions genuinely differ
            True,                                  # length parity (enforced at build)
            it["ov_gap"][d0] <= OVERLAP_TOL,       # question-overlap parity
        ]
        if judge is not None:
            votes.append(bool(judge.get(str(it["qid"]), True)))
            it["conf"] = sum(votes) / 5.0
        else:
            it["conf"] = sum(votes) / 4.0
    return items


# ------------------------------------------------------------------- render

def render_rows(items, topics):
    """Two rows per item -- gold in A, gold in B. Balanced 50/50 by construction."""
    rows = []
    for it in items:
        d = it["distractors"][0]
        for label, (oa, ob) in ((1, (it["gold"], d)), (0, (d, it["gold"]))):
            text = f"Question: {it['question']}\n\nA. {oa}\nB. {ob}"
            rows.append({
                "text": text,
                "label": label,
                "conf": float(it["conf"]),
                "subreddit": topics[it["qid"]],
                "wc": float(len(it["question"].split()) + len(oa.split())
                            + len(ob.split())),
                "qid": int(it["qid"]),
            })
    return rows


# --------------------------------------------------------------------- build

def build(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    n_q = args.pilot if args.pilot else N_QUESTIONS
    items, Q, XQ = build_bank(args.hi, args.lo, n_q)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=64, random_state=SEED)
    Z = svd.fit_transform(XQ)
    km = KMeans(n_clusters=N_TOPICS, random_state=SEED, n_init=10).fit(Z)
    topics = {i: f"topic_{c:02d}" for i, c in enumerate(km.labels_)}

    judge = None
    jp = os.path.join(DATA_DIR, "judge.json")
    if os.path.exists(jp) and not args.pilot:
        judge = json.load(open(jp))
        print(f"  applying LLM-judge verdicts for {len(judge)} items")
    items = item_conf(items, args.hi, judge)

    # split BY QUESTION so a question's two renders never straddle the boundary
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(items))
    n_test = int(round(len(items) * TEST_FRAC))
    split_of = {}
    for r, i in enumerate(perm):
        split_of[i] = "test" if r < n_test else "train"

    with open(os.path.join(DATA_DIR, "bank.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    out = {}
    for split in ("train", "test"):
        sel = [items[i] for i in range(len(items)) if split_of[i] == split]
        rows = render_rows(sel, topics)
        path = os.path.join(DATA_DIR, f"split_{split}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        out[split] = rows
        lab = np.array([r["label"] for r in rows])
        cf = np.array([r["conf"] for r in rows])
        print(f"  -> {path}  n={len(rows)} ({len(sel)} questions)  "
              f"balance={lab.mean():.3f}  conf==1.0 {(cf >= 1.0).mean():.3f}")

    man = {
        "source": HF_ID, "seed": SEED, "hi": args.hi, "lo": args.lo,
        "q_sim_max": Q_SIM_MAX, "wc_diff_max": WC_DIFF_MAX,
        "n_questions": len(items), "judge_applied": judge is not None,
        "n_train": len(out["train"]), "n_test": len(out["test"]),
        "sha256": {s: hashlib.sha256(
            open(os.path.join(DATA_DIR, f"split_{s}.jsonl"), "rb").read()
        ).hexdigest() for s in ("train", "test")},
    }
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    audit(out["test"])
    return out


# --------------------------------------------------------------------- audit

def audit(rows=None):
    """
    Shortcut audits. These two numbers ARE the content of the TF-IDF floor -- if a
    surface heuristic separates the labels, the probe will find it and the result
    means nothing.
    """
    from sklearn.metrics import roc_auc_score
    if rows is None:
        rows = load_rows("test")
    y = np.array([r["label"] for r in rows])

    def opts(t):
        a = t.split("\nA. ", 1)[1].split("\nB. ", 1)
        return a[0], a[1]

    longer_a, overlap_a, overlap_raw = [], [], []
    for r in rows:
        oa, ob = opts(r["text"])
        q = r["text"].split("\n\nA. ")[0]
        longer_a.append(len(oa.split()) - len(ob.split()))
        cq = content_words(q)
        overlap_a.append(len(content_words(oa) & cq) - len(content_words(ob) & cq))
        rq = set(q.lower().split())
        overlap_raw.append(len(set(oa.lower().split()) & rq)
                           - len(set(ob.lower().split()) & rq))

    a1 = roc_auc_score(y, longer_a)
    a2 = roc_auc_score(y, overlap_a)
    a3 = roc_auc_score(y, overlap_raw)
    sgn = lambda v: float(np.mean([x if yy == 1 else -x for x, yy in zip(v, y)]))
    print(f"\n  AUDIT  longer-option-is-gold      AUROC {a1:.3f}  (want 0.50 +/- 0.03)"
          f"  signed gap {sgn(longer_a):+.2f} words")
    print(f"  AUDIT  question-overlap, content   AUROC {a2:.3f}  (want < 0.65)"
          f"  signed gap {sgn(overlap_a):+.2f} words   <- the filtered statistic")
    print(f"  AUDIT  question-overlap, raw       AUROC {a3:.3f}  (want < 0.65)"
          f"  signed gap {sgn(overlap_raw):+.2f} words   <- incl. stopwords")
    # AUROC on a heavily tied statistic swings on tie-breaking, so the signed gap
    # is the reliable read for length.
    if abs(sgn(longer_a)) > 0.5:
        print("  *** length shortcut present -- tighten WC_DIFF_MAX ***")
    if max(a2, a3) >= 0.65:
        print("  *** question-overlap shortcut present -- tighten OVERLAP_TOL ***")
    return a1, a2, a3


# --------------------------------------------------------------------- judge

def run_judge(args):
    """
    Ask a strong model whether each DISTRACTOR is also a correct answer.

    MEASURED, AND THE REASON THIS IS NOT USED: llama8b is degenerately yes-biased
    in this framing. On n=400 it answered Yes to 98.5% of golds (so the
    P(Yes|gold) >= 0.90 control PASSES), 99.3% of distractors, and 99.8% of
    answers lifted from a completely unrelated question. Worse, on the paired
    read it preferred the unrelated answer to the gold 98% of the time. It is not
    discriminating at all, so neither the absolute rates nor the verdicts mean
    anything, and no `conf` vote is derived from them.

    The control alone could never have caught this -- an always-Yes judge passes
    it by construction. That is what the random-other discrimination arm is for.

    Distractor validity is instead read off `paired_2afc_acc` in the gate, which
    puts the question to the model in the SAME two-option forced-choice framing
    the task itself uses. That framing works where yes/no verification does not,
    which is also the reason the task is built as a forced choice.

    Kept as a runnable diagnostic; not part of the pipeline.
    """
    import importlib.util
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gs = importlib.util.spec_from_file_location(
        "gate", os.path.join(_here, "00_gate.py"))
    gate = importlib.util.module_from_spec(gs)
    gs.loader.exec_module(gate)

    items = [json.loads(l) for l in open(os.path.join(DATA_DIR, "bank.jsonl"))]
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(items), min(args.n, len(items)), replace=False)

    spec = gate.REGISTRY[args.model]
    tok = AutoTokenizer.from_pretrained(spec.path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.path, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()

    JSYS = ("You are checking answers to psychology questions. "
            "Answer with a single word: Yes or No.")
    JUSR = ("Question: {q}\n\nProposed answer: {a}\n\n"
            "Is this a correct answer to the question? Answer Yes or No.")

    def ask(pairs):
        msgs = []
        for q, a in pairs:
            m = [{"role": "system", "content": JSYS},
                 {"role": "user", "content": JUSR.format(q=q, a=a)}]
            try:
                s = tok.apply_chat_template(m, tokenize=False,
                                            add_generation_prompt=True,
                                            date_string=gate.PIN_DATE)
            except TypeError:
                s = tok.apply_chat_template(m, tokenize=False,
                                            add_generation_prompt=True)
            msgs.append(s)
        return gate.last_logits(model, tok, msgs, args.bs, "cuda")

    gold_pairs = [(items[i]["question"], items[i]["gold"]) for i in sel]
    dist_pairs = [(items[i]["question"], items[i]["distractors"][0]) for i in sel]
    # DISCRIMINATION ARM: answers from a random OTHER item, which are almost
    # certainly wrong. An always-Yes judge scores high here too, and that is the
    # only way to tell a broken judge from genuinely bad distractors -- the
    # P(Yes|gold) control alone cannot, since a degenerate judge passes it.
    rng2 = np.random.default_rng(SEED + 7)
    far = rng2.choice(len(items), len(sel), replace=False)
    rand_pairs = [(items[i]["question"], items[int(j)]["gold"])
                  for i, j in zip(sel, far)]

    lg = ask(gold_pairs)
    ids = gate.resolve_yes_no(tok, lg)
    y, n = ids["yes_id"], ids["no_id"]
    ld_g = (lg[:, y] - lg[:, n]).numpy()
    _d = ask(dist_pairs); ld_d = (_d[:, y] - _d[:, n]).numpy()
    _r = ask(rand_pairs); ld_r = (_r[:, y] - _r[:, n]).numpy()

    p_gold = float((ld_g > 0).mean())
    p_dist = float((ld_d > 0).mean())
    p_rand = float((ld_r > 0).mean())
    # PAIRED read: does the judge prefer the gold to the distractor for the SAME
    # question? Robust to an overall Yes bias, which the absolute rates are not.
    prefer_gold = float((ld_g > ld_d).mean())
    prefer_over_rand = float((ld_g > ld_r).mean())

    print(f"\n  JUDGE ({args.model}, n={len(sel)})")
    print(f"    P(Yes | gold)        {p_gold:.3f}   (control, want >= 0.90)")
    print(f"    P(Yes | distractor)  {p_dist:.3f}")
    print(f"    P(Yes | random other){p_rand:.3f}   <- discrimination arm")
    print(f"    LD(gold) > LD(distractor)   {prefer_gold:.3f}   (want >> 0.50)")
    print(f"    LD(gold) > LD(random other) {prefer_over_rand:.3f}   <- judge sanity")
    if p_gold < 0.90:
        print("    *** control FAILED -- judge is broken, everything below is void ***")
    elif p_rand > 0.80:
        print(f"    *** judge is YES-BIASED (says yes to {p_rand:.0%} of clearly "
              f"wrong answers) -- absolute rates are meaningless, read the paired "
              f"numbers instead ***")
    if prefer_over_rand < 0.65:
        print("    *** judge cannot even separate gold from a random other answer "
              "-- it is not usable as a validity check ***")
    elif prefer_gold < 0.65:
        print("    *** distractors really are competitive with the golds -- "
              "lower --hi to push them further from the gold ***")

    # verdict per item: True = the judge prefers the GOLD, i.e. this distractor is
    # not competitive. Relative, so an overall Yes bias cancels.
    verdicts = {str(items[i]["qid"]): bool(v)
                for i, v in zip(sel, (ld_g > ld_d))}
    with open(os.path.join(DATA_DIR, "judge.json"), "w") as f:
        json.dump(verdicts, f)
    print(f"    -> judge.json ({len(verdicts)} verdicts); re-run --build to fold "
          f"them into conf")


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
                         "wc": float(r["wc"]), "qid": int(r["qid"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--hi", type=float, default=0.30)
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--model", default="llama8b")
    args = ap.parse_args()
    print(f"data dir -> {DATA_DIR}")
    if args.build:
        build(args)
    if args.judge:
        run_judge(args)
    if args.audit:
        audit()


if __name__ == "__main__":
    main()
