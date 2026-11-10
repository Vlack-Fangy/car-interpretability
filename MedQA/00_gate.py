#!/usr/bin/env python3
"""
00_gate.py

Gate checks. Nothing downstream is interpretable until these pass.

Per model:
  1. resolve yes/no token IDs empirically (vocabs differ across families)
  2. render + verify chat template, locate the post token span
  3. logit-diff distribution -> AUROC, accuracy, prediction balance
  4. pair viability: how many clean/corrupted pairs survive |dLD| > tau

Writes gate_<key>.json and ld_<key>.npz. Prints a summary table.

usage:
  python 00_gate.py                          # all models, full test split
  python 00_gate.py --models llama1b         # one model
  python 00_gate.py --n-eval 300             # subsample lever
"""

import os

# Must run BEFORE torch is imported -- once CUDA is initialized the mask is ignored.
# setdefault so a shell override still wins: CUDA_VISIBLE_DEVICES=2 python 00_gate.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import importlib.util
import json
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# 01_data.py owns everything task-specific about the data. Loaded by path because
# the module name starts with a digit. 04_cache.py imports the SAME function, so
# the two loaders cannot drift.
_dh = os.path.dirname(os.path.abspath(__file__))
_ds = importlib.util.spec_from_file_location("task_data",
                                             os.path.join(_dh, "01_data.py"))
data = importlib.util.module_from_spec(_ds)
_ds.loader.exec_module(data)


# ------------------------------------------------------------------ registry

MODEL_ROOT = os.environ.get("MODEL_ROOT", "./models")


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    layer_path: str = "model.layers"   # attribute chain to the decoder block list

    @property
    def path(self):
        """Local weights if present under MODEL_ROOT, else fall back to the hub."""
        local = os.path.join(MODEL_ROOT, self.hf_id.split("/")[-1])
        return local if os.path.isdir(local) else self.hf_id


REGISTRY = {
    "llama1b":  ModelSpec("llama1b",  "meta-llama/Llama-3.2-1B-Instruct"),
    "llama3b":  ModelSpec("llama3b",  "meta-llama/Llama-3.2-3B-Instruct"),
    "qwen1.5b": ModelSpec("qwen1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    # Phi-3.5-mini is a Phi3ForCausalLM: decoder blocks still live at model.layers
    # and it has a plain lm_head, so the hooks in 04/05/07 need no change.
    "phi35":    ModelSpec("phi35",    "microsoft/Phi-3.5-mini-instruct"),
    "qwen3b":   ModelSpec("qwen3b",   "Qwen/Qwen2.5-3B-Instruct"),
    "llama8b":  ModelSpec("llama8b",  "meta-llama/Llama-3.1-8B-Instruct"),
    # add here. same three fields. nothing else in the pipeline knows the model.
}


def get_layers(model, spec: ModelSpec):
    """Resolve the decoder block ModuleList. Used by every later job's hooks."""
    obj = model
    for part in spec.layer_path.split("."):
        obj = getattr(obj, part)
    return obj


# -------------------------------------------------------------------- prompt

SYSTEM = ("You are answering United States Medical Licensing Examination "
          "multiple-choice questions. Answer with a single letter: A or B.")
USER = "{post}\n\nWhich option is correct? Answer A or B."


PIN_DATE = "26 Jul 2024"   # Llama-3.2 template calls strftime_now() otherwise -> prompt
                           # changes every run. Pinned for reproducibility + Llama parity.


def render(tok, post):
    """Rendered prompt string + char span of the post inside it."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(post=post)}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True, date_string=PIN_DATE)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    start = s.find(post)
    if start < 0:
        raise RuntimeError("post text not found in rendered prompt -- template mangled it")
    return s, (start, start + len(post))


def post_token_span(tok, rendered, char_span):
    """Map post char span -> [lo, hi) token indices. Needed for probe pooling."""
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    lo, hi = char_span
    idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > lo and a < hi]
    if not idx:
        return None
    return (min(idx), max(idx) + 1)


# ----------------------------------------------------------------- yes/no ids

# The readout is the letter pair, not yes/no. The VARIABLE names stay yes_*/no_*
# on purpose: 04/05/06/07/08 read `token_ids.yes_id` / `no_id` out of the gate
# JSON, and renaming would fork the artifact schema for no benefit.
# LD = logit("A") - logit("B");  label 1 = the gold answer sits in slot A.
YES_CANDS = ["A", " A"]
NO_CANDS  = ["B", " B"]


def single_token_ids(tok, cands):
    out = {}
    for c in cands:
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            out[c] = ids[0]
    return out


def resolve_yes_no(tok, logits):
    """
    logits: [N, V] final-position logits over a probe sample.
    Pick the surface form the model actually puts mass on.
    """
    y_cand = single_token_ids(tok, YES_CANDS)
    n_cand = single_token_ids(tok, NO_CANDS)
    if not y_cand or not n_cand:
        raise RuntimeError("no single-token yes/no surface form -- inspect vocab by hand")

    y_scores = {c: logits[:, i].mean().item() for c, i in y_cand.items()}
    n_scores = {c: logits[:, i].mean().item() for c, i in n_cand.items()}
    y_best = max(y_scores, key=y_scores.get)
    n_best = max(n_scores, key=n_scores.get)

    top = logits.mean(0).topk(20)
    top_tokens = [tok.decode([i]) for i in top.indices.tolist()]

    assert y_cand[y_best] != n_cand[n_best], "yes and no resolved to the same token id"

    return {
        "yes_form": y_best, "yes_id": y_cand[y_best],
        "no_form":  n_best, "no_id":  n_cand[n_best],
        "yes_candidates": y_scores,
        "no_candidates":  n_scores,
        "top20_mean_logit_tokens": top_tokens,
    }


# ------------------------------------------------------------------- forward

@torch.no_grad()
def last_logits(model, tok, prompts, bs, device):
    """Final-position logits. LEFT padding so index -1 is the real last token."""
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        outs.append(model(**enc).logits[:, -1, :].float().cpu())
        if i % (bs * 20) == 0:
            print(f"    {i}/{len(prompts)}", flush=True)
    return torch.cat(outs)


# ---------------------------------------------------------------------- data

def load_eval(n_eval, seed=0):
    rows = data.load_rows("test")
    print(f"using split 'test', n={len(rows)}")

    if n_eval is not None and n_eval < len(rows):
        # Subsample by QUESTION, not by row: both renders of a question must stay
        # together or the label balance skews and paired_2afc_acc loses its pairs.
        rng = np.random.default_rng(seed)
        by_q = {}
        for i, r in enumerate(rows):
            by_q.setdefault(r["qid"], []).append(i)
        qids = sorted(by_q)
        take = min(len(qids), n_eval // 2)
        sel = rng.choice(len(qids), take, replace=False)
        keep = sorted(i for q in sel for i in by_q[qids[q]])
        rows = [rows[i] for i in keep]
        print(f"subsampled to n={len(rows)} ({take} questions)")
    return rows


# ---------------------------------------------------------------------- gate

def run_gate(spec: ModelSpec, rows, bs, tau_grid, device, out_dir="."):
    print(f"\n=== {spec.key} :: {spec.path} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(spec.path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.path, dtype=torch.bfloat16, device_map={"": device})
    model.eval()

    n_layers = len(get_layers(model, spec))
    d_model = model.config.hidden_size

    prompts, spans = [], []
    for r in rows:
        s, cspan = render(tok, r["text"])
        prompts.append(s)
        spans.append(post_token_span(tok, s, cspan))

    print("--- rendered prompt (example 0) ---")
    print(prompts[0])
    print("--- post token span (example 0):", spans[0], flush=True)

    probe = last_logits(model, tok, prompts[:min(50, len(prompts))], bs, device)
    ids = resolve_yes_no(tok, probe)
    print("yes:", ids["yes_form"], ids["yes_id"], "| no:", ids["no_form"], ids["no_id"])
    print("top-20 by mean logit:", ids["top20_mean_logit_tokens"], flush=True)

    lg = last_logits(model, tok, prompts, bs, device)
    ld = (lg[:, ids["yes_id"]] - lg[:, ids["no_id"]]).numpy()
    y = np.array([r["label"] for r in rows])
    pred = (ld > 0).astype(int)

    stats = {
        "n": len(rows),
        "n_layers": n_layers,
        "d_model": d_model,
        "label_balance": float(y.mean()),
        "accuracy": float((pred == y).mean()),
        "f1": float(f1_score(y, pred)),
        "pct_pred_yes": float(pred.mean()),
        "auroc_ld": float(roc_auc_score(y, ld)),
        "ld_mean_pos": float(ld[y == 1].mean()),
        "ld_mean_neg": float(ld[y == 0].mean()),
        "ld_std": float(ld.std()),
        "post_span_ok": float(np.mean([s is not None for s in spans])),
    }

    # 2AFC: does LD move the right way when the SAME question has its options
    # swapped? Threshold-free and free of position bias, so unlike raw accuracy
    # (which sits near 0.5 whenever the model has an A-or-B preference) this is
    # an honest read on whether the model knows the answer.
    by_q = {}
    for i, r in enumerate(rows):
        by_q.setdefault(r["qid"], {})[r["label"]] = i
    both = [v for v in by_q.values() if 0 in v and 1 in v]
    stats["n_questions"] = len(both)
    stats["paired_2afc_acc"] = float(
        np.mean([ld[v[1]] > ld[v[0]] for v in both])) if both else float("nan")

    # pair viability: confidence==1.0, one pos one neg.
    # `conf` here is the five-vote item-quality proxy from 01_data.py -- same value
    # grid as Dreaddit's human agreement, so this threshold is unchanged.
    pos = [i for i, r in enumerate(rows) if r["label"] == 1 and r["conf"] >= 1.0]
    neg = [i for i, r in enumerate(rows) if r["label"] == 0 and r["conf"] >= 1.0]
    print(f"conf==1.0 pool: {len(pos)} pos / {len(neg)} neg "
          f"-> {len(pos) * len(neg)} possible pairs")
    rng = np.random.default_rng(0)
    # cross-product, not 1:1 -- permutation matching caps at min(|pos|,|neg|) for no reason
    n_draw = min(1000, len(pos) * len(neg))
    pairs = [(int(rng.choice(pos)), int(rng.choice(neg))) for _ in range(n_draw)]
    dld = np.abs(ld[[p for p, _ in pairs]] - ld[[n for _, n in pairs]])
    stats["n_conf1_pos"] = len(pos)
    stats["n_conf1_neg"] = len(neg)
    stats["n_pairs_sampled"] = len(pairs)
    stats["dld_median"] = float(np.median(dld))
    # absolute grid (numerical sanity) + model-relative grid (cross-model comparability;
    # recovery is scale-invariant so what matters is dLD vs this model's own LD spread)
    stats["pair_survival_abs"] = {str(t): float((dld > t).mean()) for t in tau_grid}
    sd = ld.std()
    stats["pair_survival_rel"] = {
        f"{k}sd": float((dld > k * sd).mean()) for k in (0.25, 0.5, 1.0, 2.0)}
    stats["n_usable_at_1sd"] = int((dld > sd).sum())

    # calibration gap: best achievable accuracy if the LD threshold were tuned
    order = np.argsort(ld)
    ys = y[order]
    best = max(((ys[:i] == 0).sum() + (ys[i:] == 1).sum()) / len(ys)
               for i in range(len(ys) + 1))
    stats["accuracy_tuned_threshold"] = float(best)

    np.savez(os.path.join(out_dir, f"ld_{spec.key}.npz"), ld=ld, label=y, pred=pred,
             spans=np.array([s if s else (-1, -1) for s in spans]))
    with open(os.path.join(out_dir, f"gate_{spec.key}.json"), "w") as f:
        json.dump({"spec": {**spec.__dict__, "resolved_path": spec.path},
                   "prompt": {"system": SYSTEM, "user": USER, "pin_date": PIN_DATE,
                              "rendered_example_0": prompts[0]},
                   "token_ids": ids, "stats": stats}, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["llama1b", "llama3b", "qwen1.5b", "phi35"])
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"artifacts -> {os.path.abspath(args.out)}")

    if torch.cuda.is_available():
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"| visible={torch.cuda.device_count()} "
              f"| device 0 -> {torch.cuda.get_device_name(0)} "
              f"| {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    else:
        raise RuntimeError("no CUDA device visible -- check CUDA_VISIBLE_DEVICES")

    tau_grid = [0.5, 1.0, 2.0, 4.0]
    rows = load_eval(args.n_eval)

    table = {}
    for key in args.models:
        table[key] = run_gate(REGISTRY[key], rows, args.bs, tau_grid, args.device, args.out)

    print("\n" + "=" * 96)
    hdr = (f"{'model':10} {'L':>3} {'d':>5} {'acc':>6} {'tuned':>6} {'%A':>6} "
           f"{'auroc':>6} {'2afc':>6} {'|LD|sd':>7} {'pairs@1sd':>10} {'n_pairs':>8}")
    print(hdr)
    print("-" * 96)
    for k, s in table.items():
        print(f"{k:10} {s['n_layers']:3d} {s['d_model']:5d} {s['accuracy']:6.3f} "
              f"{s['accuracy_tuned_threshold']:6.3f} {s['pct_pred_yes']:6.3f} "
              f"{s['auroc_ld']:6.3f} {s['paired_2afc_acc']:6.3f} {s['ld_std']:7.2f} "
              f"{s['pair_survival_rel']['1.0sd']:10.3f} {s['n_usable_at_1sd']:8d}")
    print("=" * 96)


if __name__ == "__main__":
    main()