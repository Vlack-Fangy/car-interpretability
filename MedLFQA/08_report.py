#!/usr/bin/env python3
"""
08_report.py

Assembles every artifact into the final tables + figures. Read-only, no GPU.

  Table 1  gate         task sanity -- the precondition for everything else
  Table 2  localisation four independent estimates of WHERE the decision lives
  Table 3  scores       A1 readability / A2 causal / A3 localisation, per method
  Table 4  agreement    do the three methods RANK the models the same way?

Table 4 is the actual POC result. If the three rankings agree, "interpretability"
behaves like one property and the full study is worth building. If they split --
say the probe ranks 8B first but patching ranks 3B first -- that is the more
interesting outcome: a model can be easy to READ and hard to INTERVENE on.

usage:
  python 08_report.py
"""

import argparse
import json
import os

import numpy as np

MODELS = ["llama1b", "qwen1.5b", "phi35"]


def load(out, pat, keys):
    d = {}
    for k in keys:
        p = os.path.join(out, pat.format(k))
        if os.path.exists(p):
            with open(p) as f:
                d[k] = json.load(f)
    return d


def md_table(headers, rows):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
         for i, h in enumerate(headers)]
    out = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def rise_share(pa):
    """
    Fraction of total marginal recovery contributed by the single largest layer
    transition, plus the layer whose output first carries it.

    Scale-free: depth cancels because it is a ratio. The entropy-based A3 divides
    by log(L-1), which lets a deep model's diffuseness hide -- llama8b scored
    A3 = 0.390 against llama1b's 0.388 while concentrating 38% vs 47% of the work
    in its biggest transition. Rank on this instead.
    """
    c = np.clip(np.array(pa["l0_curve"], dtype=float), 0, None)
    d = np.clip(np.diff(c), 0, None)
    if d.sum() <= 0:
        return 0.0, 0
    return float(d.max() / d.sum()), int(np.argmax(d)) + 1


def a2_specific(st, floor_frac=0.10):
    """
    Re-select the steering peak by SPECIFICITY, not raw effect size.

    |dLD| grows monotonically toward the output for ANY direction -- at the last
    layer you inject directly into norm -> lm_head, so a random vector moves the
    logits almost as much as the probe direction does. argmax(dLD_probe) therefore
    lands on the layer where that artifact is biggest, not where the probe
    direction is most specifically causal.

    Restrict to layers carrying real signal (dLD_probe >= floor_frac * max), then
    take the layer maximising 1 - random/probe.
    """
    p = np.array(st["dLD_probe_by_layer"], dtype=float)
    r = np.array(st["dLD_random_by_layer"], dtype=float)
    a2 = np.clip(1.0 - r / np.maximum(p, 1e-9), 0, 1)
    a2[p <= 0] = 0.0
    masked = np.where(p >= floor_frac * p.max(), a2, -1.0)
    l = int(np.argmax(masked))
    return float(a2[l]), l


def rank(d, higher_better=True):
    """1 = best."""
    items = [(k, v) for k, v in d.items() if v == v]
    items.sort(key=lambda kv: kv[1], reverse=higher_better)
    return {k: i + 1 for i, (k, _) in enumerate(items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    M = args.models

    gate = load(args.out, "gate_{}.json", M)
    patch = load(args.out, "patch_{}.json", M)
    probe = load(args.out, "probe_{}.json", M)
    steer = load(args.out, "steer_{}.json", M)

    doc = ["# Interpretability comparison — Long-form Medical QA (MedLFQA), must-have vs nice-to-have\n"]

    # ---- Table 1 ----------------------------------------------------------
    doc.append("\n## Table 1 — Task gate (precondition, not a score)\n")
    rows = []
    for k in M:
        if k not in gate:
            continue
        s = gate[k]["stats"]
        rows.append([k, s["n_layers"], s["d_model"], f"{s['accuracy']:.3f}",
                     f"{s['accuracy_tuned_threshold']:.3f}",
                     f"{s['pct_pred_yes']:.3f}", f"{s['auroc_ld']:.3f}",
                     f"{s['ld_std']:.2f}", f"{s.get('paired_2afc_acc', float('nan')):.3f}"])
    doc.append(md_table(["model", "L", "d", "acc", "tuned", "%yes", "AUROC", "|LD|sd",
                         "2AFC"], rows))
    doc.append("\n*AUROC is threshold-free and is what matters here; raw accuracy is "
               "depressed by miscalibration, not by absent signal. `%A` near 0 or 1 "
               "means the model has a slot preference, which AUROC absorbs as a "
               "constant offset. `2AFC` is the honest competence number: the "
               "fraction of questions where LD moves the right way when that "
               "question's own options are swapped.*\n")

    # ---- Table 2 ----------------------------------------------------------
    doc.append("\n## Table 2 — Where does the decision live? (depth as l/L)\n")
    rows = []
    for k in M:
        pr, pa, st = probe.get(k), patch.get(k), steer.get(k)
        rows.append([
            k,
            f"{pr['probe_peak_frac']:.2f}" if pr else "-",
            f"{pr['lens_peak_frac']:.2f}" if pr else "-",
            f"{pa['l0_peak_frac']:.2f}" if pa else "-",
            f"{rise_share(pa)[1] / (pa['n_layers'] - 1):.2f}" if pa else "-",
            f"{a2_specific(st)[1] / (st['n_layers'] - 1):.2f}" if st else "-",
        ])
    doc.append(md_table(["model", "probe peak", "lens peak", "patch peak",
                         "patch rise", "steer peak"], rows))
    doc.append("\n*Four independent estimates. If they disagree that is a finding, "
               "not an error — probes find where information is linearly READABLE, "
               "patching finds where it is causally USED.*\n")

    # ---- Table 3 ----------------------------------------------------------
    doc.append("\n## Table 3 — Scores (per method, deliberately not blended)\n")
    rows, cols = [], {"A1": {}, "A2": {}, "A3": {}}
    for k in M:
        pr, pa, st = probe.get(k), patch.get(k), steer.get(k)
        if pr:
            fl = pr["floors"]
            # absolute: vs the strongest floor incl. TF-IDF. A validity check --
            # does the representation beat bag-of-words at all.
            a1_abs = pr["probe_peak_acc"] - max(fl.values())
            # relative: vs THIS model's own untrained/uncontextualised baseline.
            # TF-IDF is identical across models, so subtracting it cannot change a
            # ranking; only the model-specific floor controls for width/capacity.
            mf = max(v for kk, v in fl.items() if kk != "tfidf")
            a1_rel = pr["probe_peak_acc"] - mf
        else:
            a1_abs = a1_rel = float("nan")
            mf = float("nan")
        if st:
            a2, a2_layer = a2_specific(st)
            abl = st["dLD_ablate_at_peak"] / (st["dLD_probe_at_peak"] + 1e-9)
        else:
            a2, a2_layer, abl = float("nan"), "-", float("nan")
        if pa:
            rs, rl = rise_share(pa)
            a3 = pa["localization"]
        else:
            rs, rl, a3 = float("nan"), "-", float("nan")
        cols["A1"][k], cols["A2"][k], cols["A3"][k] = a1_rel, a2, rs
        rows.append([
            k,
            f"{pr['probe_peak_acc']:.3f}" if pr else "-",
            f"{mf:.3f}" if pr else "-",
            f"{a1_rel:+.3f}", f"{a1_abs:+.3f}", f"{a2:.3f}", f"{a2_layer}",
            f"{abl:.3f}", f"{rs:.3f}", f"{rl}", f"{a3:.3f}",
            f"{pr['subreddit_control_acc']:.3f}" if pr else "-",
            f"{pa['check_readout']:.1e}" if pa else "-",
        ])
    doc.append(md_table(["model", "probe peak", "model floor", "A1 rel", "A1 abs",
                         "A2 causal", "A2 layer", "abl/steer", "A3 rise", "rise l",
                         "A3 entropy", "topic", "readout chk"], rows))
    doc.append("\n*A1 rel = probe peak minus max(layer-0, random-init) — the "
               "model's own baseline, which controls for representation width. "
               "A1 abs additionally clears the TF-IDF lexical floor; since TF-IDF is "
               "model-independent it is a validity check, not a comparison. "
               "Ranking below uses A1 rel. A2 = 1 − random/probe steering effect. "
               "A3 = 1 − normalised entropy of per-layer marginal recovery. "
               "`topic` is a POSITIVE control here, not a leakage bound: both "
               "rows of an item share a question and carry opposite labels, so "
               "MI(topic; label) is exactly 0 and topic cannot help. Unlike the "
               "option-swap tasks the two labels carry DIFFERENT text, so the "
               "layer-0 bag-of-words floor is genuinely above chance and A1 is "
               "the margin the representation clears it by. `readout chk` must be "
               "~0 or the patching numbers are void.*\n")

    # ---- Table 4 ----------------------------------------------------------
    doc.append("\n## Table 4 — Do the three methods agree on the ranking?\n")
    r1, r2, r3 = rank(cols["A1"]), rank(cols["A2"]), rank(cols["A3"])
    rows = [[k, r1.get(k, "-"), r2.get(k, "-"), r3.get(k, "-")] for k in M]
    doc.append(md_table(["model", "probe rank", "steering rank", "patching rank"],
                        rows))

    common = [k for k in M if k in r1 and k in r2 and k in r3]
    if len(common) >= 3:
        a = np.array([[r1[k], r2[k], r3[k]] for k in common])
        from itertools import combinations
        rhos = []
        for i, j in combinations(range(3), 2):
            x, y = a[:, i], a[:, j]
            rhos.append(np.corrcoef(x, y)[0, 1])
        agree = bool(np.all(a[:, 0] == a[:, 1]) and np.all(a[:, 1] == a[:, 2]))
        doc.append(f"\nPairwise rank correlation: "
                   f"probe~steer {rhos[0]:+.2f}, probe~patch {rhos[1]:+.2f}, "
                   f"steer~patch {rhos[2]:+.2f}")
        doc.append(f"\n**Rankings {'AGREE' if agree else 'SPLIT'}.** "
                   + ("Interpretability behaves like a single property on these "
                      "three models — the full study is worth building."
                      if agree else
                      "The methods disagree, which means 'interpretability' is not "
                      "one dimension. This is the stronger POC result: it is the "
                      "justification for the larger study."))

    text = "\n".join(doc)
    path = os.path.join(args.out, "REPORT.md")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n\nwrote {path}")

    # ---- figures ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -- pip install matplotlib for figures)")
        return

    fig, axes = plt.subplots(1, len(M), figsize=(5 * len(M), 4), squeeze=False)
    for ax, k in zip(axes[0], M):
        pr, pa, st = probe.get(k), patch.get(k), steer.get(k)
        if pr:
            a = np.array(pr["probe_acc_by_layer"])
            ax.plot(np.linspace(0, 1, len(a)), (a - a.min()) / (np.ptp(a) + 1e-9),
                    label="probe acc")
            ln = np.array(pr["lens_auroc_by_layer"])
            ax.plot(np.linspace(0, 1, len(ln)), (ln - ln.min()) / (np.ptp(ln) + 1e-9),
                    label="logit lens")
        if pa:
            c = np.array(pa["l0_curve"])
            ax.plot(np.linspace(0, 1, len(c)), np.clip(c, 0, 1), label="patch recovery")
        if st:
            d = np.array(st["dLD_probe_by_layer"])
            ax.plot(np.linspace(0, 1, len(d)), d / (d.max() + 1e-9), label="steering")
        ax.set_title(k)
        ax.set_xlabel("depth (l / L)")
        ax.set_ylabel("normalised")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "localisation_overlay.png"), dpi=150)
    print(f"wrote {os.path.join(args.out, 'localisation_overlay.png')}")

    labs = ["A1 read", "A2 causal", "A3 local"]
    ang = np.linspace(0, 2 * np.pi, len(labs), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(polar=True)
    for k in M:
        v = [cols["A1"].get(k, 0), cols["A2"].get(k, 0), cols["A3"].get(k, 0)]
        v = [0 if x != x else float(np.clip(x, 0, 1)) for x in v]
        ax.plot(ang, v + v[:1], marker="o", label=k)
        ax.fill(ang, v + v[:1], alpha=.1)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "scores_radar.png"), dpi=150)
    print(f"wrote {os.path.join(args.out, 'scores_radar.png')}")


if __name__ == "__main__":
    main()