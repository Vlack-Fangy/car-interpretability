#!/usr/bin/env python3
"""
04_cache.py

One forward per example, caching the residual stream at every layer.
Feeds 06_probe.py and 07_steer.py. Also computes the logit lens for free.

Cached per split, per model:
  mean : [N, L+1, D]   mean-pooled over the POST tokens only (not the template)
  last : [N, L+1, D]   final position -- the site the decision is read from
  lens : [N, L+1]      logit-lens LD at each depth

Index convention: rep[0] is the embedding output (input to layer 0), rep[l+1] is
the output of layer l. rep[0] doubles as the layer-0 / bag-of-words control.

usage:
  python 04_cache.py                       # train + test, all models
  python 04_cache.py --randinit            # ALSO cache a random-init control
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import argparse
import importlib.util
import json

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("gate", os.path.join(_here, "00_gate.py"))
gate = importlib.util.module_from_spec(_s)
_s.loader.exec_module(gate)


def load_split(split):
    """Same function 00_gate.load_eval uses -- guarantees identical row ordering."""
    return gate.data.load_rows(split)


@torch.no_grad()
def cache_split(model, layers, tok, rows, yes_id, no_id, dev, bs):
    """Returns mean [N,L+1,D], last [N,L+1,D], lens [N,L+1]."""
    L, D = len(layers), model.config.hidden_size
    N = len(rows)
    out_mean = np.zeros((N, L + 1, D), dtype=np.float16)
    out_last = np.zeros((N, L + 1, D), dtype=np.float16)

    # logit-diff direction in unembedding space. RMSNorm first, then one dot
    # product -- far cheaper than unembedding the whole vocab per layer.
    w_diff = (model.lm_head.weight[yes_id] - model.lm_head.weight[no_id]).float()
    norm = model.model.norm

    buf = {}

    def mk(i):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            buf[i + 1] = h.detach()
        return fn

    def pre(module, args, kwargs):
        buf[0] = (args[0] if args else kwargs["hidden_states"]).detach()

    hooks = [layers[i].register_forward_hook(mk(i)) for i in range(L)]
    hooks.append(layers[0].register_forward_pre_hook(pre, with_kwargs=True))

    n_span_miss = 0

    try:
        for b in range(0, N, bs):
            chunk = rows[b:b + bs]
            prompts, spans = [], []
            for r in chunk:
                s, cspan = gate.render(tok, r["text"])
                prompts.append(s)
                spans.append(gate.post_token_span(tok, s, cspan))
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(dev)
            buf.clear()
            model(**enc, use_cache=False)

            # left padding shifts every index right by (S - real_len)
            S = enc["input_ids"].shape[1]
            shift = (S - enc["attention_mask"].sum(1)).cpu().numpy()

            for l in range(L + 1):
                h = buf[l].float()                       # [B, S, D]
                out_last[b:b + len(chunk), l] = h[:, -1, :].cpu().numpy().astype(np.float16)
                for j, sp in enumerate(spans):
                    lo, hi = (sp[0] + shift[j], sp[1] + shift[j]) if sp else (0, S)
                    out_mean[b + j, l] = h[j, lo:hi, :].mean(0).cpu().numpy().astype(np.float16)
            n_span_miss += sum(sp is None for sp in spans)
            if b % (bs * 10) == 0:
                print(f"    {b}/{N}", flush=True)
    finally:
        for h in hooks:
            h.remove()

    # The sp-is-None branch pools over (0, S), i.e. across left-padding. If it ever
    # fires, `mean` is silently corrupted and the provable layer-0 floor is void.
    assert n_span_miss == 0, f"{n_span_miss} rows had no post span"

    lens = np.zeros((N, L + 1), dtype=np.float32)
    for l in range(L + 1):
        h = torch.from_numpy(out_last[:, l]).float().to(dev)
        lens[:, l] = (norm(h) @ w_diff).cpu().numpy()
    return out_mean, out_last, lens


def run(key, args):
    spec = gate.REGISTRY[key]
    dev = args.device
    print(f"\n{'=' * 70}\n{key} :: {spec.path}\n{'=' * 70}", flush=True)

    with open(os.path.join(args.out, f"gate_{key}.json")) as f:
        g = json.load(f)
    yes_id, no_id = g["token_ids"]["yes_id"], g["token_ids"]["no_id"]

    tok = AutoTokenizer.from_pretrained(spec.path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for tag, init in [("", False)] + ([("_randinit", True)] if args.randinit else []):
        if init:
            cfg = AutoConfig.from_pretrained(spec.path)
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_config(cfg).to(dev).to(torch.bfloat16)
            print("  [random-init control]")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                spec.path, dtype=torch.bfloat16, device_map={"": dev})
        model.eval()
        layers = gate.get_layers(model, spec)

        for split in args.splits:
            rows = load_split(split)
            print(f"  {split}: n={len(rows)}", flush=True)
            m, l_, lens = cache_split(model, layers, tok, rows, yes_id, no_id,
                                      dev, args.bs)
            np.savez_compressed(
                os.path.join(args.out, f"cache_{key}{tag}_{split}.npz"),
                mean=m, last=l_, lens=lens,
                label=np.array([r["label"] for r in rows]),
                conf=np.array([r["conf"] for r in rows]),
                subreddit=np.array([r["subreddit"] for r in rows]),
                qid=np.array([r["qid"] for r in rows]),
                text=np.array([r["text"] for r in rows], dtype=object))
            print(f"    -> cache_{key}{tag}_{split}.npz  {m.shape}")
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["llama1b", "llama3b", "qwen1.5b", "phi35"])
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--randinit", action="store_true")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"| {torch.cuda.get_device_name(0)}")
    for k in args.models:
        run(k, args)


if __name__ == "__main__":
    main()