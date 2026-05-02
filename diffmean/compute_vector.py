"""Analyse pre-collected activations: per-layer DiffMean vector + linear-probe AUC.

Given an activations directory laid out as <root>/L<NN>/{H_pos,H_neg,diffmean_vec}.pt,
for each layer we report:
  - ||v||             magnitude of the DiffMean direction
  - cos(H+, v)        mean cosine of pos examples with v (should be > 0)
  - cos(H-, v)        mean cosine of neg examples with v (should be < 0)
  - AUC               of `H @ v` separating pos vs neg (1.0 = linearly separable)
  - acc@thr=0         accuracy with threshold at 0 (after centering by joint mean)

Usage:
    python -m diffmean.compute_vector \\
        --acts diffmean/outputs/acts/gemma2-9b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _auc(scores_pos: torch.Tensor, scores_neg: torch.Tensor) -> float:
    """Mann-Whitney U based AUC. O(N log N)."""
    s = torch.cat([scores_pos, scores_neg])
    y = torch.cat([torch.ones_like(scores_pos), torch.zeros_like(scores_neg)])
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    n_pos = scores_pos.numel()
    n_neg = scores_neg.numel()
    sum_ranks_pos = ranks[y == 1].sum().item()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def analyze_layer(layer_dir: Path) -> dict:
    H_pos = torch.load(layer_dir / "H_pos.pt").float()  # [N, d]
    H_neg = torch.load(layer_dir / "H_neg.pt").float()
    v = (H_pos.mean(0) - H_neg.mean(0))  # recompute from stored acts
    v_unit = v / (v.norm() + 1e-9)

    scores_pos = H_pos @ v_unit
    scores_neg = H_neg @ v_unit
    cos_pos = (H_pos / (H_pos.norm(dim=-1, keepdim=True) + 1e-9)) @ v_unit
    cos_neg = (H_neg / (H_neg.norm(dim=-1, keepdim=True) + 1e-9)) @ v_unit

    # Threshold at midpoint of class means → balanced accuracy
    thr = 0.5 * (scores_pos.mean() + scores_neg.mean())
    acc = ((scores_pos > thr).float().mean().item()
           + (scores_neg <= thr).float().mean().item()) / 2

    return {
        "layer": int(layer_dir.name.lstrip("L")),
        "n_pos": H_pos.shape[0],
        "n_neg": H_neg.shape[0],
        "d_model": H_pos.shape[1],
        "v_norm": v.norm().item(),
        "cos_pos_mean": cos_pos.mean().item(),
        "cos_neg_mean": cos_neg.mean().item(),
        "score_pos_mean": scores_pos.mean().item(),
        "score_neg_mean": scores_neg.mean().item(),
        "auc": _auc(scores_pos, scores_neg),
        "balanced_acc": acc,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, required=True,
                   help="Activations root containing L<NN>/ subdirs.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional jsonl path to dump per-layer stats.")
    args = p.parse_args()

    layer_dirs = sorted([d for d in args.acts.iterdir()
                         if d.is_dir() and d.name.startswith("L")])
    if not layer_dirs:
        raise SystemExit(f"no L*/ subdirs under {args.acts}")

    print(f"{'layer':>5} {'n':>5} {'||v||':>8} {'cos+':>7} {'cos-':>7} "
          f"{'AUC':>6} {'acc':>6}")
    rows = []
    for d in layer_dirs:
        r = analyze_layer(d)
        rows.append(r)
        print(f"{r['layer']:>5d} {r['n_pos']:>5d} {r['v_norm']:>8.2f} "
              f"{r['cos_pos_mean']:>+7.3f} {r['cos_neg_mean']:>+7.3f} "
              f"{r['auc']:>6.3f} {r['balanced_acc']:>6.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
