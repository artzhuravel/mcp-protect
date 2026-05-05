"""Compute DiffMean vectors per attack paradigm and security_risk slice.

If a single global vector doesn't steer cleanly, per-attack-type vectors may.
We re-load the saved [N, d_model] activation tensors, the index sidecar that
maps row k → tags{paradigm, security_risk}, and emit per-slice vectors at the
chosen layer:

  <acts>/L<NN>/diffmean_vec.pt                 (global, already exists)
  <acts>/L<NN>/by_paradigm/<P>.pt              (per paradigm)
  <acts>/L<NN>/by_security_risk/<R>.pt         (per risk category)

Also reports per-slice AUC for diagnostic comparison.

Usage:
  python -m diffmean.per_paradigm_vectors \\
      --acts diffmean/outputs/acts/phi4 \\
      --layer 20
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unknown"


def _auc(scores_pos: torch.Tensor, scores_neg: torch.Tensor) -> float:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--layer", type=int, default=20)
    args = p.parse_args()

    layer_dir = args.acts / f"L{args.layer:02d}"
    H_pos = torch.load(layer_dir / "H_pos.pt").float()
    H_neg = torch.load(layer_dir / "H_neg.pt").float()
    idx = [json.loads(l) for l in (args.acts / "index.jsonl").open()]
    print(f"[slice] {len(idx)} index rows, H_pos {tuple(H_pos.shape)}, "
          f"H_neg {tuple(H_neg.shape)}, layer={args.layer}")

    # Build per-axis groupings. Each index row has either i_pos OR i_neg
    # (flat-mode rollouts) or both (legacy paired). Track them separately so
    # we can pull from the right tensor when computing per-group means.
    groups: dict[str, dict[str, dict[str, list[int]]]] = {
        "paradigm": {}, "security_risk": {},
    }
    for row in idx:
        tags = row.get("tags", {})
        for axis in groups:
            grp = str(tags.get(axis, "Unknown"))
            g = groups[axis].setdefault(grp, {"pos": [], "neg": []})
            if row.get("i_pos") is not None:
                g["pos"].append(row["i_pos"])
            elif row.get("i_neg") is not None:
                g["neg"].append(row["i_neg"])
            else:
                # Legacy paired schema: index k maps to row k in both
                # H_pos and H_neg (rare path).
                k = row.get("i")
                if k is not None:
                    g["pos"].append(k); g["neg"].append(k)

    by_p_dir = layer_dir / "by_paradigm"
    by_r_dir = layer_dir / "by_security_risk"
    by_p_dir.mkdir(parents=True, exist_ok=True)
    by_r_dir.mkdir(parents=True, exist_ok=True)

    for axis, mapping in groups.items():
        out_dir = by_p_dir if axis == "paradigm" else by_r_dir
        print(f"\n=== {axis} ===")
        print(f"{'group':40s} {'n_pos':>6} {'n_neg':>6} {'AUC':>6} {'||v||':>8}")
        for grp, buckets in sorted(mapping.items(), key=lambda kv: -(len(kv[1]['pos']) + len(kv[1]['neg']))):
            n_pos, n_neg = len(buckets["pos"]), len(buckets["neg"])
            if n_pos < 3 or n_neg < 3:
                print(f"{grp[:40]:40s} {n_pos:>6d} {n_neg:>6d}  (skipped, <3 in a class)")
                continue
            Hp = H_pos[torch.tensor(buckets["pos"])]
            Hn = H_neg[torch.tensor(buckets["neg"])]
            v = Hp.mean(0) - Hn.mean(0)
            v_unit = v / (v.norm() + 1e-9)
            auc = _auc(Hp @ v_unit, Hn @ v_unit)
            torch.save(v.to(torch.float16), out_dir / f"{_safe(grp)}.pt")
            print(f"{grp[:40]:40s} {n_pos:>6d} {n_neg:>6d} {auc:>6.3f} {v.norm().item():>8.2f}")

    print(f"\n→ vectors saved under {by_p_dir}/ and {by_r_dir}/")


if __name__ == "__main__":
    main()
