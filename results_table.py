#!/usr/bin/env python3
"""
results_table.py

Emits the one table this study is for: four interpretability metrics x
(task, dataset, model).

  Faithfulness - Probe Readability   A1 = probe peak acc - max(floors)
                                     06_probe.py:"A1_readability"
  Sensitivity  - Steering Causality  A2 = 1 - dLD_random / dLD_probe, taken at the
                                     most SPECIFIC layer rather than the layer of
                                     largest raw effect (see a2_specific below)
  Sparseness   - Patch Localization  A3 = 1 - normalised entropy of the per-layer
                                     MARGINAL recovery   05_patch.py:"localization"
  Probe AUROC                        max over layers of the probe's test AUROC

Read-only, CPU, seconds.

usage:
  python results_table.py
  python results_table.py --csv results.csv
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

NAMES = {
    "llama1b":  "Llama-3.2-1B-Instruct",
    "llama3b":  "Llama-3.2-3B-Instruct",
    "qwen1.5b": "Qwen2.5-1.5B-Instruct",
    "phi35":    "Phi-3.5-mini-instruct",
    "qwen3b":   "Qwen2.5-3B-Instruct",
    "llama8b":  "Llama-3.1-8B-Instruct",
    # CNN arm -- DummyCNNBreast/v1
    "resnet18":        "ResNet18",
    "densenet121":     "DenseNet121",
    "efficientnet_b0": "EfficientNet_B0",
}

# (directory, task label, dataset label, model keys) -- the model list differs per
# task, so it is carried here rather than assumed global.
TASKS = [
    ("Suicide",  "Suicide Detection",     "Suicidal Ideation - Reddit Dataset",
     ["llama1b", "qwen3b", "llama8b"]),
    ("PsychMCQ", "Psychoeducation MCQ",   "psychology-question-answer",
     ["llama1b", "qwen3b", "llama8b"]),
    ("MedQA",    "Medical-MCQ",           "MedQA-USMLE-4-MultiTurnRobust (200 ex.)",
     ["llama1b", "llama3b", "qwen1.5b", "phi35"]),
    ("MedLFQA",  "Long-form-medical-QA",  "MedLFQA (200 ex.)",
     ["llama1b", "qwen1.5b", "phi35"]),
    ("MedHallu", "Answer-generation",     "MedHallu-labeled (200 ex.)",
     ["llama1b", "qwen1.5b", "phi35"]),
    ("DummyCNNBreast/v1", "Cancer Detection", "DDSM Breast Cancer",
     ["resnet18", "densenet121", "efficientnet_b0"]),
]


def load(task, name):
    p = os.path.join(HERE, task, "artifacts", name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def a2_specific(st, floor_frac=0.10):
    """
    Steering causality at the most SPECIFIC layer, not the biggest-effect layer.

    |dLD| grows toward the output for ANY direction -- at the last layer you inject
    straight into norm -> lm_head, so a random vector moves the logits nearly as
    much as the probe direction does. argmax(dLD_probe) therefore lands where that
    artifact is largest, not where the probe direction is most specifically causal.
    Restrict to layers carrying real signal, then maximise 1 - random/probe.
    """
    p = np.array(st["dLD_probe_by_layer"], dtype=float)
    r = np.array(st["dLD_random_by_layer"], dtype=float)
    a2 = np.clip(1.0 - r / np.maximum(p, 1e-9), 0, 1)
    a2[p <= 0] = 0.0
    masked = np.where(p >= floor_frac * p.max(), a2, -1.0)
    return float(a2[int(np.argmax(masked))])


def md_table(headers, rows):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
         for i, h in enumerate(headers)]
    out = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "RESULTS.csv"))
    ap.add_argument("--md", default=os.path.join(HERE, "RESULTS.md"))
    args = ap.parse_args()

    fmt = lambda v, s="{:.3f}": s.format(v) if v == v else "—"
    rows, csv = [], ["task,dataset,model,faithfulness_probe_readability,"
                     "sensitivity_steering_causality,sparseness_patch_localization,"
                     "probe_auroc,gate_auroc,pool"]
    for tdir, tlabel, dlabel, mkeys in TASKS:
        for i, mkey in enumerate(mkeys):
            mlabel = NAMES[mkey]
            pr = load(tdir, f"probe_{mkey}.json")
            pool = pr.get("pool", "mean") if pr else "?"
            st = load(tdir, f"steer_{mkey}.json")
            pa = load(tdir, f"patch_{mkey}.json")
            a1 = pr["A1_readability"] if pr else float("nan")
            auc = max(pr["probe_auroc_by_layer"]) if pr else float("nan")
            a2 = a2_specific(st) if st else float("nan")
            a3 = pa["localization"] if pa else float("nan")
            g = load(tdir, f"gate_{mkey}.json")
            gauc = g["stats"]["auroc_ld"] if g else float("nan")
            flag = "  \u26a0" if gauc == gauc and gauc < 0.60 else ""
            rows.append([tlabel if i == 0 else "", dlabel if i == 0 else "", mlabel,
                         fmt(a1, "{:+.3f}"), fmt(a2), fmt(a3), fmt(auc),
                         fmt(gauc) + flag, pool])
            csv.append(f'"{tlabel}","{dlabel}","{mlabel}",{a1:.4f},{a2:.4f},'
                       f'{a3:.4f},{auc:.4f},{gauc:.4f},{pool}')

    hdr = ["Task", "Dataset", "Model",
           "Faithfulness — Probe Readability",
           "Sensitivity — Steering Causality",
           "Sparseness — Patch Localization",
           "Probe AUROC", "Gate AUROC", "Pool"]
    table = md_table(hdr, rows)

    doc = ["# Interpretability metrics\n", "", table, "",
           "*Faithfulness (A1) = probe peak accuracy minus the strongest trivial "
           "floor (layer-0 bag-of-words, random-init, TF-IDF, shuffled-label); it "
           "is what contextualisation buys over the best non-representational "
           "explanation. Sensitivity (A2) = 1 − (random-direction steering effect / "
           "probe-direction steering effect) at the most specific layer; 0 means "
           "the probe direction is readable but causally inert. Sparseness (A3) = "
           "1 − normalised entropy of per-layer marginal patching recovery; 1.0 "
           "means one layer transition does all the work. Probe AUROC is the best "
           "layer's threshold-free separation on the held-out test split. Gate AUROC is "
           "the MODEL's own yes/no readout, shown because it bounds how much the "
           "causal columns can mean: \u26a0 marks a cell where the model is at or "
           "near chance on the task, so its Sensitivity and Sparseness are measured "
           "against a readout that carries no signal. `Pool` is where the probe reads the "
           "residual stream. Medical-MCQ uses the FINAL position rather than a mean "
           "over the item: its clinical vignettes run ~159 tokens against ~11 tokens "
           "of options, so the options are only 6.7% of the pooled span (vs 80.4% on "
           "Psychoeducation MCQ) and, because a question's two renders share the "
           "identical vignette, mean-pooling averaged the option-order signal away — "
           "it put Probe AUROC at 0.53-0.60 with negative A1 while the models' own "
           "readout reached 0.88. Reading at the final position recovers it.*",
           "",
           "**Cancer Detection is a CNN arm and its three rows carry a health "
           "warning.** `pool=gap` means the probe reads each conv stage's feature "
           "map global-average-pooled over space (9 taps per backbone); the LD "
           "readout is logit(malignant) - logit(benign); patching replaces a fixed "
           "25% AREA of the feature map rather than the whole map, because "
           "replacing the whole map in a feed-forward CNN is a complete state swap "
           "and returns recovery 1.000 at every tap, which leaves A3 undefined. "
           "More importantly, `09_pixel_floor.py` shows the DDSM crops are "
           "separable from a 4-pixel border ring alone at AUROC 0.998, against "
           "0.750 for the lesion centre, and all three models sit at balanced "
           "accuracy 1.000 with 96-99% saturation. Patching confirms this "
           "causally: replacing the border region recovers 1.4-3.0x more of the "
           "decision than replacing the lesion centre. So Gate AUROC 1.000 and "
           "Probe AUROC 1.000 describe a border artefact, not lesion reasoning, "
           "and A1 = +0.020 is the whole margin the learned representation holds "
           "over a raw-pixel baseline that already scores 0.980.*"]
    txt = "\n".join(doc)
    with open(args.md, "w") as f:
        f.write(txt + "\n")
    with open(args.csv, "w") as f:
        f.write("\n".join(csv) + "\n")
    print(txt)
    print(f"\nwrote {args.md}\nwrote {args.csv}")


if __name__ == "__main__":
    main()
