#!/usr/bin/env python3
"""
05_patch.py

Activation patching. Clean (stressed) residual stream -> corrupted (non-stressed) run.

  L0: final position only, sweep layers      -> recovery curve    [L]
  L1: last K positions x layers              -> recovery heatmap  [L, K]

  recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)
             0.0 = site irrelevant   1.0 = site fully restored the clean answer

TWO PLUMBING CHECKS run automatically before any result is reported. Both are exact,
not approximate:

  identity : patch corrupted's own activations back into corrupted -> MUST be 0.0
             (proves the hook writes without corrupting anything else)
  readout  : patch layer L-1 at the final position                 -> MUST be 1.0
             (layer L-1 output at pos -1 feeds norm -> lm_head directly, so the
              logits become exactly the clean logits. any other value = bad indexing)

If either fails, stop. Nothing downstream is real.

usage:
  python 05_patch.py                          # all models, L0 + L1
  python 05_patch.py --models llama1b --no-l1 # quick
"""

import os

# Must run BEFORE torch. setdefault so a shell override still wins.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import importlib.util
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- reuse the gate's prompt + registry so prompts are byte-identical --------
_here = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("gate", os.path.join(_here, "00_gate.py"))
gate = importlib.util.module_from_spec(_s)
_s.loader.exec_module(gate)


# ------------------------------------------------------------------ patching

def make_hook(src, positions):
    """
    src       : [B, P, D] values to write
    positions : list of P NEGATIVE indices (left padding -> -1 is the last real token)
    """
    def fn(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = (output[0] if is_tuple else output).clone()
        for j, p in enumerate(positions):
            h[:, p, :] = src[:, j, :].to(h.dtype)
        return ((h,) + tuple(output[1:])) if is_tuple else h
    return fn


@torch.no_grad()
def forward_cache(model, layers, tok, prompts, yes_id, no_id, device, K):
    """
    One forward, caching each layer's RAW output via read hooks.

    Do NOT use output_hidden_states=True for this. HF appends the final entry
    AFTER model.norm(), so hidden_states[L] is post-RMSNorm and does not equal
    the output of layers[L-1]. Read hooks give cache[l] == exactly what a write
    hook on layers[l] replaces, for every l including the last.

    returns cache (list of [B, K, D]), ld [B], enc
    """
    cache = [None] * len(layers)

    def mk(i):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            cache[i] = h[:, -K:, :].detach().clone()
        return fn

    hooks = [layers[i].register_forward_hook(mk(i)) for i in range(len(layers))]
    try:
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        lg = model(**enc, use_cache=False).logits[:, -1, :].float()
    finally:
        for h in hooks:
            h.remove()
    return cache, lg[:, yes_id] - lg[:, no_id], enc


@torch.no_grad()
def ld_under_hook(model, layers, ell, src, positions, enc, yes_id, no_id):
    hk = layers[ell].register_forward_hook(make_hook(src, positions))
    try:
        lg = model(**enc, use_cache=False).logits[:, -1, :].float()
    finally:
        hk.remove()
    return lg[:, yes_id] - lg[:, no_id]


def recovery(ld_patched, ld_clean, ld_corr):
    return ((ld_patched - ld_corr) / (ld_clean - ld_corr)).cpu().numpy()


# -------------------------------------------------------------------- scores

def localization(curve):
    """
    A3 = 1 - normalized entropy of the per-layer MARGINAL contribution.

    Uses diff(curve), not the curve itself. Recovery curves are near-monotone --
    once the decision is assembled, every later layer also recovers it. Entropy of
    the raw curve therefore mostly measures how LATE the rise happens, not how
    localized the computation is. The increments are where the work actually gets
    done, so that is what gets scored.

    1.0 = one layer transition does everything. 0.0 = smeared evenly.
    Normalized by log(L-1) so different depths are comparable.
    """
    d = np.clip(np.diff(np.clip(curve, 0, None)), 0, None)
    if d.sum() <= 0 or len(d) < 2:
        return 0.0
    p = d / d.sum()
    nz = p[p > 0]
    return float(1.0 - (-(nz * np.log(nz)).sum()) / np.log(len(d)))


def localization_cumulative(curve):
    """Old definition, kept as a secondary diagnostic. Entropy of the raw curve."""
    w = np.clip(curve, 0, None)
    if w.sum() <= 0:
        return 0.0
    p = w / w.sum()
    nz = p[p > 0]
    return float(1.0 - (-(nz * np.log(nz)).sum()) / np.log(len(curve)))


# ---------------------------------------------------------------------- main

def build_pairs(rows, ld, sd, n_pairs, seed, min_len=None, tok_lens=None):
    """conf==1.0, correctly-ranked (ld_clean > ld_corrupt), separated by > 1 sd."""
    pos = [i for i, r in enumerate(rows) if r["label"] == 1 and r["conf"] >= 1.0]
    neg = [i for i, r in enumerate(rows) if r["label"] == 0 and r["conf"] >= 1.0]
    rng = np.random.default_rng(seed)
    cand = [(c, k) for c in pos for k in neg]
    rng.shuffle(cand)
    out = []
    for c, k in cand:
        if ld[c] - ld[k] <= sd:            # ranked correctly AND separated
            continue
        if min_len is not None and (tok_lens[c] < min_len or tok_lens[k] < min_len):
            continue
        out.append((c, k))
        if len(out) >= n_pairs:
            break
    return out


def build_pairs_matched(rows, ld, sd, n_pairs, seed, min_len=None, tok_lens=None):
    """
    MCQ pairing: clean and corrupt are the two renders of the SAME question, so
    they differ only in which slot holds the gold. That is a far cleaner
    corruption than Dreaddit's "a different post" -- the token multiset is
    preserved, so the L1 position index refers to comparable content across both
    members of the pair.

    Same filters as build_pairs: conf==1.0, correctly ranked, separated by > 1 sd.
    """
    by_q = {}
    for i, r in enumerate(rows):
        by_q.setdefault(r["qid"], {})[r["label"]] = i
    cand = [(v[1], v[0]) for q, v in by_q.items()
            if 0 in v and 1 in v and rows[v[1]]["conf"] >= 1.0]
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)

    def _take(pool, out, seen):
        for c, k in pool:
            if (c, k) in seen:
                continue
            if ld[c] - ld[k] <= sd:        # ranked correctly AND separated
                continue
            if min_len is not None and (tok_lens[c] < min_len
                                        or tok_lens[k] < min_len):
                continue
            out.append((c, k))
            seen.add((c, k))
            if len(out) >= n_pairs:
                return True
        return False

    out, seen = [], set()
    _take(cand, out, seen)
    n_matched = len(out)

    # CROSS-ITEM TOP-UP. Same-question pairs are the better contrast, but they cap
    # at one per item and the `> 1 sd` separation filter culls most of those when
    # the model is weak on the task -- on MedHallu and MedLFQA that left ~32-40
    # pairs out of a requested 96, and a recovery curve averaged over 32 pairs has
    # roughly 3x the standard error of Dreaddit's 256-pair curve.
    #
    # So top up with cross-item pairs: clean is any label-1 row, corrupt any
    # label-0 row from a DIFFERENT item. That is exactly the pairing Dreaddit uses
    # (a pos x neg cross-product), so it is the original validated contrast rather
    # than a new one -- it just does not hold the question fixed. Matched pairs are
    # always consumed first, and the split is printed so the mix is visible.
    if len(out) < n_pairs:
        pos = [i for i, r in enumerate(rows) if r["label"] == 1 and r["conf"] >= 1.0]
        neg = [i for i, r in enumerate(rows) if r["label"] == 0 and r["conf"] >= 1.0]
        qid = [r["qid"] for r in rows]
        cross = [(c, k) for c in pos for k in neg if qid[c] != qid[k]]
        rng.shuffle(cross)
        _take(cross, out, seen)
    print(f"    pairs: {n_matched} matched + {len(out) - n_matched} cross-item "
          f"= {len(out)}", flush=True)
    return out


def run_model(key, rows, args):
    spec = gate.REGISTRY[key]
    dev = args.device
    print(f"\n{'=' * 70}\n{key} :: {spec.path}\n{'=' * 70}", flush=True)

    with open(os.path.join(args.out, f"gate_{key}.json")) as f:
        g = json.load(f)
    yes_id, no_id = g["token_ids"]["yes_id"], g["token_ids"]["no_id"]
    z = np.load(os.path.join(args.out, f"ld_{key}.npz"))
    ld_all, sd = z["ld"], z["ld"].std()
    # ld_all is indexed by position in `rows`. If 00_gate.py was run with --n-eval
    # it wrote a SHORTER ld than load_eval(None) returns here, and every pair index
    # would silently point at the wrong item.
    assert len(rows) == ld_all.shape[0], (
        f"rows={len(rows)} but ld_{key}.npz has {ld_all.shape[0]} -- "
        "00_gate.py was run with a different --n-eval; re-run it without one")

    tok = AutoTokenizer.from_pretrained(spec.path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.path, dtype=torch.bfloat16, device_map={"": dev})
    model.eval()
    layers = gate.get_layers(model, spec)
    L = len(layers)

    prompts = [gate.render(tok, r["text"])[0] for r in rows]
    tok_lens = np.array([len(tok(p, add_special_tokens=False)["input_ids"])
                         for p in prompts])

    res = {"n_layers": L}

    # ---------------------------------------------------------------- L0 ----
    pairs = build_pairs_matched(rows, ld_all, sd, args.n_pairs, args.seed)
    print(f"L0: {len(pairs)}/{args.n_pairs} matched pairs", flush=True)
    if len(pairs) < args.n_pairs:
        print(f"  WARNING: short -- conf>=1.0 pool or the |dLD| > {sd:.2f} "
              f"separation filter is starving it", flush=True)
    rec0, checks = [], {"identity": [], "readout": []}

    for b in range(0, len(pairs), args.bs):
        chunk = pairs[b:b + args.bs]
        pc = [prompts[c] for c, _ in chunk]
        pk = [prompts[k] for _, k in chunk]

        hc, ldc, _ = forward_cache(model, layers, tok, pc, yes_id, no_id, dev, K=1)
        hk_, ldk, enck = forward_cache(model, layers, tok, pk, yes_id, no_id, dev, K=1)

        # CHECK 1 -- identity. corrupted's own values back in. must be 0.0
        mid = L // 2
        ldp = ld_under_hook(model, layers, mid, hk_[mid], [-1], enck, yes_id, no_id)
        checks["identity"].append(recovery(ldp, ldc, ldk))

        col = np.zeros((L, len(chunk)))
        for ell in range(L):
            ldp = ld_under_hook(model, layers, ell, hc[ell], [-1],
                                enck, yes_id, no_id)
            col[ell] = recovery(ldp, ldc, ldk)
        # CHECK 2 -- readout. last layer at final pos must be 1.0
        checks["readout"].append(col[L - 1])
        rec0.append(col)
        print(f"  {min(b + args.bs, len(pairs))}/{len(pairs)}", flush=True)

    rec0 = np.concatenate(rec0, axis=1)                      # [L, n_pairs]
    idc = float(np.abs(np.concatenate(checks["identity"])).max())
    rdo = float(np.abs(np.concatenate(checks["readout"]) - 1.0).max())
    print(f"\n  CHECK identity  max|rec|      = {idc:.2e}   (want ~0)")
    print(f"  CHECK readout   max|rec - 1|  = {rdo:.2e}   (want ~0)")
    ok = idc < 1e-2 and rdo < 1e-2
    print(f"  -> {'PASS' if ok else '*** FAIL -- indexing is wrong, stop here ***'}\n")

    curve = rec0.mean(1)
    res["l0_curve"] = curve.tolist()
    res["l0_sem"] = (rec0.std(1) / np.sqrt(rec0.shape[1])).tolist()
    res["l0_peak_layer"] = int(np.argmax(curve))
    res["l0_peak_frac"] = float(np.argmax(curve) / (L - 1))
    res["l0_peak_recovery"] = float(curve.max())
    res["localization"] = localization(curve)
    res["localization_cumulative"] = localization_cumulative(curve)
    res["rise_layer"] = int(np.argmax(np.diff(np.clip(curve, 0, None))))
    res["check_identity"] = idc
    res["check_readout"] = rdo
    res["check_pass"] = ok
    res["n_pairs_l0"] = rec0.shape[1]

    print("  layer  recovery")
    for i, v in enumerate(curve):
        bar = "#" * max(0, int(round(v * 40)))
        print(f"  {i:5d}  {v:8.3f}  {bar}")

    # ---------------------------------------------------------------- L1 ----
    if not args.no_l1:
        K = args.k_pos
        p1 = build_pairs_matched(rows, ld_all, sd, args.n_pairs_l1, args.seed,
                                 min_len=K, tok_lens=tok_lens)
        print(f"\nL1: {len(p1)}/{args.n_pairs_l1} pairs, K={K}, {L * K} sites -> "
              f"{L * K * int(np.ceil(len(p1) / args.bs))} forwards", flush=True)
        print(f"  prompt tokens: min={tok_lens.min()} med={int(np.median(tok_lens))} "
              f"max={tok_lens.max()}", flush=True)
        if len(p1) < args.n_pairs_l1:
            print(f"  WARNING: short -- the min_len>={K} token filter is dropping "
                  f"pairs; lower --k-pos", flush=True)
        heat = []
        for b in range(0, len(p1), args.bs):
            chunk = p1[b:b + args.bs]
            pc = [prompts[c] for c, _ in chunk]
            pk = [prompts[k] for _, k in chunk]
            hc, ldc, _ = forward_cache(model, layers, tok, pc, yes_id, no_id, dev, K)
            _, ldk, enck = forward_cache(model, layers, tok, pk, yes_id, no_id, dev, K)
            blk = np.zeros((L, K, len(chunk)))
            for ell in range(L):
                for j in range(K):
                    p = -(K - j)                      # j=0 -> -K ... j=K-1 -> -1
                    ldp = ld_under_hook(model, layers, ell,
                                        hc[ell][:, j:j + 1, :], [p],
                                        enck, yes_id, no_id)
                    blk[ell, j] = recovery(ldp, ldc, ldk)
            heat.append(blk)
            print(f"  {min(b + args.bs, len(p1))}/{len(p1)}", flush=True)
        heat = np.concatenate(heat, axis=2).mean(2)          # [L, K]
        res["l1_heatmap"] = heat.tolist()
        res["l1_k"] = K
        res["n_pairs_l1"] = len(p1)
        pk_ = np.unravel_index(np.argmax(heat), heat.shape)
        res["l1_peak"] = {"layer": int(pk_[0]), "pos_from_end": int(K - pk_[1]),
                          "recovery": float(heat.max())}
        np.save(os.path.join(args.out, f"patch_l1_{key}.npy"), heat)

    np.savez(os.path.join(args.out, f"patch_l0_{key}.npz"),
             rec=rec0, curve=curve, pairs=np.array(pairs))
    with open(os.path.join(args.out, f"patch_{key}.json"), "w") as f:
        json.dump(res, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return res


def plot(table, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping figures. pip install matplotlib)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, r in table.items():
        c = np.array(r["l0_curve"])
        s = np.array(r["l0_sem"])
        x = np.arange(len(c)) / (len(c) - 1)
        ax.plot(x, c, marker="o", ms=3, label=f"{k} (L={r['n_layers']})")
        ax.fill_between(x, c - s, c + s, alpha=.2)
    ax.axhline(0, color="k", lw=.5)
    ax.axhline(1, color="k", lw=.5, ls=":")
    ax.set_xlabel("layer depth  (l / L)")
    ax.set_ylabel("recovery")
    ax.set_title("Activation patching L0 -- final position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "patch_l0_curves.png"), dpi=150)
    print(f"\nwrote {os.path.join(out, 'patch_l0_curves.png')}")

    for k, r in table.items():
        if "l1_heatmap" not in r:
            continue
        h = np.array(r["l1_heatmap"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(h, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-abs(h).max(), vmax=abs(h).max())
        ax.set_xlabel("position from end  (rightmost = final token)")
        ax.set_ylabel("layer")
        ax.set_title(f"Activation patching L1 -- {k}")
        ax.set_xticks(np.linspace(0, h.shape[1] - 1, 5))
        ax.set_xticklabels([f"-{int(v)}" for v in
                            np.linspace(h.shape[1], 1, 5)])
        fig.colorbar(im, label="recovery")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"patch_l1_{k}.png"), dpi=150)
        print(f"wrote {os.path.join(out, f'patch_l1_{k}.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["llama1b", "llama3b", "qwen1.5b", "phi35"])
    ap.add_argument("--n-pairs", type=int, default=256, help="L0 pairs")
    ap.add_argument("--n-pairs-l1", type=int, default=96, help="L1 pairs")
    # 80, not Dreaddit's 48: the MCQ item span runs ~67 tokens with only ~15
    # tokens of instruction after it, so K=80 covers the entire item.
    ap.add_argument("--k-pos", type=int, default=80, help="L1 positions from end")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-l1", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"| {torch.cuda.get_device_name(0)}")
    rows = gate.load_eval(None)

    table = {k: run_model(k, rows, args) for k in args.models}

    print("\n" + "=" * 88)
    print(f"{'model':10} {'L':>3} {'peak l':>7} {'l/L':>6} {'peak rec':>9} "
          f"{'A3 local':>9} {'ident':>9} {'readout':>9} {'ok':>4}")
    print("-" * 88)
    for k, r in table.items():
        print(f"{k:10} {r['n_layers']:3d} {r['l0_peak_layer']:7d} "
              f"{r['l0_peak_frac']:6.2f} {r['l0_peak_recovery']:9.3f} "
              f"{r['localization']:9.3f} {r['check_identity']:9.1e} "
              f"{r['check_readout']:9.1e} {'PASS' if r['check_pass'] else 'FAIL':>4}")
    print("=" * 88)
    plot(table, args.out)


if __name__ == "__main__":
    main()