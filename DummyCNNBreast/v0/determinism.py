"""
Make the GPU give the same gradient twice.

WHY THIS FILE EXISTS
    One of our two explanation methods IS the gradient. If the gradient
    changes between identical runs, the heatmap changes. And AvgSensitivity
    works by nudging the image and measuring how much the heatmap moved --
    so a jittery gradient makes that score measure the graphics card instead
    of the model.

    The forward pass was already stable. Only the backward pass wobbles,
    because cuDNN picks fast kernels that accumulate in nondeterministic
    order.

USAGE -- import and call enable() BEFORE anything touches CUDA:

    import determinism; determinism.enable()
    import torch  # everything else after

CUBLAS_WORKSPACE_CONFIG must be set before the CUDA context is created,
which is why this module writes it at import time rather than inside
enable().
"""
import os

# Must land in the environment before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random                                          # noqa: E402
import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402


def enable(seed: int = 1337, warn_only: bool = True) -> dict:
    """Turn on every determinism switch. Returns what was set.

    warn_only=True: ops with no deterministic implementation warn instead of
    raising. Keeps things running while telling you exactly what is unsafe.
    Set warn_only=False when you want a hard stop instead.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False        # kernel autotuning picks
                                                  # different kernels per run
    torch.use_deterministic_algorithms(True, warn_only=warn_only)

    return dict(
        seed=seed,
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        warn_only=warn_only,
        cuda_available=torch.cuda.is_available(),
    )


def seed_worker(worker_id: int) -> None:
    """Pass to DataLoader(worker_init_fn=...) so workers seed reproducibly."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------- diagnostic
def gradient_stability(model, x, target_class: int = 1, trials: int = 3) -> dict:
    """How reproducible is the gradient, and does the wobble actually matter?

    Bitwise equality is the ideal, but the question that decides whether we
    can proceed is different: do the differences change which pixels rank as
    important? Sparseness is a Gini coefficient over sorted attributions and
    FaithfulnessCorrelation sums attributions over pixel subsets -- both read
    the ORDER, not the exact float. A wobble far below the signal that leaves
    the top-k unchanged is harmless. One that reshuffles the top-k is not.

    Returns absolute error, error relative to the signal, and rank agreement.
    """
    model.eval()
    grads = []
    for _ in range(trials):
        xi = x.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(xi)[:, target_class].sum()
        grads.append(torch.autograd.grad(out, xi)[0].detach().clone())

    g0 = grads[0]
    signal = g0.abs().mean().item()
    max_abs = max((g - g0).abs().max().item() for g in grads[1:])
    bitwise = all(torch.equal(g, g0) for g in grads[1:])

    # rank agreement on the pixels that actually carry the explanation
    a0 = g0.abs().flatten(1)
    n_px = a0.shape[1]
    k = max(1, int(0.01 * n_px))                 # top 1% of pixels
    top0 = torch.topk(a0, k, dim=1).indices
    overlaps = []
    for g in grads[1:]:
        tk = torch.topk(g.abs().flatten(1), k, dim=1).indices
        for b in range(a0.shape[0]):
            inter = len(set(top0[b].tolist()) & set(tk[b].tolist()))
            overlaps.append(inter / k)
    top_k_overlap = float(np.mean(overlaps)) if overlaps else 1.0

    return dict(
        bitwise=bitwise,
        max_abs_diff=max_abs,
        mean_abs_grad=signal,
        relative=max_abs / signal if signal > 0 else float("nan"),
        top1pct_overlap=top_k_overlap,
        k=k,
        n_pixels=n_px,
        # harmless if the top 1% of pixels is fully preserved and the
        # numerical wobble is four orders below the signal
        acceptable=bool(bitwise or (top_k_overlap >= 0.999
                                    and max_abs / max(signal, 1e-30) < 1e-4)),
    )


def report(d: dict, indent: str = "    ") -> None:
    print(f"{indent}bitwise identical : {d['bitwise']}")
    print(f"{indent}max |difference|  : {d['max_abs_diff']:.3e}")
    print(f"{indent}mean |gradient|   : {d['mean_abs_grad']:.3e}")
    print(f"{indent}relative wobble   : {d['relative']:.3e}")
    print(f"{indent}top-1% pixel agree: {d['top1pct_overlap']:.4f} "
          f"(k={d['k']:,} of {d['n_pixels']:,})")
    print(f"{indent}-> {'ACCEPTABLE' if d['acceptable'] else 'NOT ACCEPTABLE'}")