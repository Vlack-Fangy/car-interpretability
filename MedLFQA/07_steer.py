#!/usr/bin/env python3
"""
07_steer.py

Does the model USE the direction the probe found, or is it merely readable?

A probe scoring 0.85 proves a linear readout exists. It does not prove the model
consumes that direction. This is the causal test: push the residual stream along
the probe direction and watch the decision move.

  steer     h[:, -1, :] += alpha * ||h|| * w_hat        -> dLD_probe
  random    same alpha, same norm, random unit vector   -> dLD_random   (control)
  ablate    h -= (h . w_hat) w_hat                      -> dLD_ablate

  A2 = 1 - |dLD_random| / |dLD_probe|      clipped to [0, 1]

  A2 near 0 -> the probe direction is no more causal than a random push of the
  same magnitude. Readable but inert. This is the failure mode that most probing
  papers never check for.

alpha is scaled by the residual-stream norm at the target layer, never absolute.
Logit scales differ ~16x across these models (qwen3b |LD|sd = 10.4 vs llama1b
0.61), so a fixed alpha would nudge one model and shove another.

usage:
  python 07_steer.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import argparse
import importlib.util
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("gate", os.path.join(_here, "00_gate.py"))
gate = importlib.util.module_from_spec(_s)
_s.loader.exec_module(gate)


def edit_hook(vec=None, alpha=0.0, ablate=None):
    """Additive steer (vec, alpha) or projection-removal (ablate). Final position."""
    def fn(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = (output[0] if is_tuple else output).clone()
        x = h[:, -1, :].float()
        if ablate is not None:
            d = ablate.to(x.device)
            x = x - (x @ d).unsqueeze(-1) * d
        else:
            d = vec.to(x.device)
            scale = x.norm(dim=-1, keepdim=True)
            x = x + alpha * scale * d
        h[:, -1, :] = x.to(h.dtype)
        return ((h,) + tuple(output[1:])) if is_tuple else h
    return fn


@torch.no_grad()
def ld_with(model, layers, ell, hook, enc, yes_id, no_id):
    hk = layers[ell].register_forward_hook(hook) if hook else None
    try:
        lg = model(**enc, use_cache=False).logits[:, -1, :].float()
    finally:
        if hk:
            hk.remove()
    return (lg[:, yes_id] - lg[:, no_id]).cpu().numpy()


def run(key, rows, args):
    spec = gate.REGISTRY[key]
    dev = args.device
    print(f"\n{'=' * 70}\n{key} :: {spec.path}\n{'=' * 70}", flush=True)

    with open(os.path.join(args.out, f"gate_{key}.json")) as f:
        tid = json.load(f)["token_ids"]
    yes_id, no_id = tid["yes_id"], tid["no_id"]
    with open(os.path.join(args.out, f"probe_{key}.json")) as f:
        pr = json.load(f)
    dirs = np.load(os.path.join(args.out, f"probe_dirs_{key}.npy"))   # [L+1, D]

    tok = AutoTokenizer.from_pretrained(spec.path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.path, dtype=torch.bfloat16, device_map={"": dev})
    model.eval()
    layers = gate.get_layers(model, spec)
    L = len(layers)

    # sample whole question pairs so the steered set is label-balanced by
    # construction -- steering an all-gold-in-A sample would confound the effect
    # with the model's own slot preference
    rng = np.random.default_rng(args.seed)
    by_q = {}
    for i, r in enumerate(rows):
        by_q.setdefault(r["qid"], []).append(i)
    qids = sorted(by_q)
    pick = rng.choice(len(qids), min(args.n // 2, len(qids)), replace=False)
    sel = sorted(i for j in pick for i in by_q[qids[j]])
    prompts = [gate.render(tok, rows[i]["text"])[0] for i in sel]

    d_probe = np.zeros(L)
    d_rand = np.zeros(L)
    d_abl = np.zeros(L)

    for b in range(0, len(prompts), args.bs):
        enc = tok(prompts[b:b + args.bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        base = ld_with(model, layers, 0, None, enc, yes_id, no_id)

        for ell in range(L):
            # probe direction for layer l is dirs[l+1] (dirs[0] = embeddings)
            w = torch.from_numpy(dirs[ell + 1]).to(dev)

            # DEGENERATE-DIRECTION GUARD. On this task the layer-0/1 probes sit at
            # exactly chance by construction (option-swap + RoPE -> the mean-pooled
            # embedding is identical for a question's two renders), so regularised
            # logistic regression shrinks every coefficient to 0 and ||w|| == 0.
            # w / ||w|| is then 0/0 = nan, which propagates into d_probe, makes
            # argmax select that layer, and turns A2 into nan. Leave those layers
            # at zero effect instead: argmax cannot pick them, and a2_specific's
            # `p >= floor_frac * p.max()` mask already excludes them.
            if not torch.isfinite(w).all() or w.norm() < 1e-8:
                continue
            w = w / w.norm()

            pos = ld_with(model, layers, ell, edit_hook(w, +args.alpha),
                          enc, yes_id, no_id)
            neg = ld_with(model, layers, ell, edit_hook(w, -args.alpha),
                          enc, yes_id, no_id)
            d_probe[ell] += np.abs(pos - neg).sum()

            r_acc = 0.0
            for _ in range(args.n_rand):
                rv = torch.randn_like(w)
                rv = rv / rv.norm()
                rp = ld_with(model, layers, ell, edit_hook(rv, +args.alpha),
                             enc, yes_id, no_id)
                rn = ld_with(model, layers, ell, edit_hook(rv, -args.alpha),
                             enc, yes_id, no_id)
                r_acc += np.abs(rp - rn).sum()
            d_rand[ell] += r_acc / args.n_rand

            ab = ld_with(model, layers, ell, edit_hook(ablate=w),
                         enc, yes_id, no_id)
            d_abl[ell] += np.abs(ab - base).sum()
        print(f"  {min(b + args.bs, len(prompts))}/{len(prompts)}", flush=True)

    n = len(prompts)
    d_probe, d_rand, d_abl = d_probe / n, d_rand / n, d_abl / n

    peak = int(np.argmax(d_probe))
    ratio = float(d_rand[peak] / (d_probe[peak] + 1e-9))
    res = {
        "n_layers": L, "alpha": args.alpha, "n_examples": n,
        "dLD_probe_by_layer": d_probe.tolist(),
        "dLD_random_by_layer": d_rand.tolist(),
        "dLD_ablate_by_layer": d_abl.tolist(),
        "steer_peak_layer": peak,
        "steer_peak_frac": float(peak / (L - 1)),
        "dLD_probe_at_peak": float(d_probe[peak]),
        "dLD_random_at_peak": float(d_rand[peak]),
        "dLD_ablate_at_peak": float(d_abl[peak]),
        "causal_ratio_rand_over_probe": ratio,
        "A2_causal": float(np.clip(1.0 - ratio, 0, 1)),
        "probe_peak_layer": pr["probe_peak_layer"],
    }
    with open(os.path.join(args.out, f"steer_{key}.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n  steer peak layer {peak}/{L - 1}  (probe peak was {pr['probe_peak_layer']})")
    print(f"  dLD probe  {d_probe[peak]:8.3f}")
    print(f"  dLD random {d_rand[peak]:8.3f}   <- control")
    print(f"  dLD ablate {d_abl[peak]:8.3f}")
    print(f"  A2 causal  {res['A2_causal']:8.3f}")

    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["llama1b", "qwen1.5b", "phi35"])
    ap.add_argument("--n", type=int, default=128, help="examples to steer")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="steer magnitude as a FRACTION of the residual norm")
    ap.add_argument("--n-rand", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"| {torch.cuda.get_device_name(0)}")
    rows = gate.load_eval(None)
    table = {k: run(k, rows, args) for k in args.models}

    print("\n" + "=" * 88)
    print(f"{'model':10} {'steer l':>8} {'l/L':>6} {'probe l':>8} "
          f"{'dLD prb':>8} {'dLD rnd':>8} {'dLD abl':>8} {'A2':>7}")
    print("-" * 88)
    for k, r in table.items():
        print(f"{k:10} {r['steer_peak_layer']:8d} {r['steer_peak_frac']:6.2f} "
              f"{r['probe_peak_layer']:8d} {r['dLD_probe_at_peak']:8.3f} "
              f"{r['dLD_random_at_peak']:8.3f} {r['dLD_ablate_at_peak']:8.3f} "
              f"{r['A2_causal']:7.3f}")
    print("=" * 88)


if __name__ == "__main__":
    main()