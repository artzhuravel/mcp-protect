"""Step 1 of the filtered SAE pipeline: rank features by activation difference.

Inputs:
  --h-pos    [N_pos, d_model] activations from compliant prompts
  --h-neg    [N_neg, d_model] activations from resistant prompts
  --sae-path Qwen-Scope layer{N}.sae.pt for the layer those activations came from
  --layer    layer index (recorded in output JSON)
  --top-k    how many candidates to keep (default 100)

Output:
  features.json: {"<layer>": [feature_idx, ...]}  ← consumed by compute_s_out.py

The activations should be the team's H_pos.pt / H_neg.pt or our own.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sae_utils import QwenScopeSAE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--h-pos", type=Path, required=True)
    p.add_argument("--h-neg", type=Path, required=True)
    p.add_argument("--sae-path", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--chunk", type=int, default=256,
        help="Encode activations in chunks to bound peak memory.",
    )
    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"[select] loading activations  pos={args.h_pos}  neg={args.h_neg}")
    H_pos = torch.load(args.h_pos, map_location=args.device, weights_only=True).float()
    H_neg = torch.load(args.h_neg, map_location=args.device, weights_only=True).float()
    if H_pos.dim() != 2 or H_neg.dim() != 2:
        raise SystemExit(f"expected 2D tensors, got {H_pos.shape} and {H_neg.shape}")
    if H_pos.shape[1] != H_neg.shape[1]:
        raise SystemExit(f"d_model mismatch: {H_pos.shape[1]} vs {H_neg.shape[1]}")
    print(f"[select]   pos={tuple(H_pos.shape)}  neg={tuple(H_neg.shape)}")

    print(f"[select] loading SAE  {args.sae_path}")
    weights = torch.load(args.sae_path, map_location="cpu", weights_only=True)
    sae = QwenScopeSAE(weights, k=50, device=args.device)
    if sae.d_model != H_pos.shape[1]:
        raise SystemExit(
            f"SAE d_model {sae.d_model} != activation d_model {H_pos.shape[1]}"
        )
    print(f"[select]   d_model={sae.d_model}  d_sae={sae.d_sae}  k={sae.k}")

    # Encode in chunks; sparse mean is the same shape (d_sae,) regardless of N.
    print("[select] encoding activations through SAE ...")
    with torch.inference_mode():
        sums_pos = torch.zeros(sae.d_sae, dtype=torch.float32, device=args.device)
        for i in range(0, H_pos.shape[0], args.chunk):
            sums_pos += sae.encode(H_pos[i : i + args.chunk]).to(torch.float32).sum(0)
        sums_neg = torch.zeros(sae.d_sae, dtype=torch.float32, device=args.device)
        for i in range(0, H_neg.shape[0], args.chunk):
            sums_neg += sae.encode(H_neg[i : i + args.chunk]).to(torch.float32).sum(0)

    mean_pos = sums_pos / H_pos.shape[0]
    mean_neg = sums_neg / H_neg.shape[0]
    diff = (mean_pos - mean_neg).cpu()

    abs_diff = diff.abs()
    topk_vals, topk_idx = abs_diff.topk(args.top_k)

    # Sort the chosen K by signed diff so compliance-ward features come first.
    feature_ids = sorted(topk_idx.tolist(), key=lambda i: -diff[i].item())

    output = {str(args.layer): feature_ids}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))

    print("[select] top-20 candidates by signed diff (compliance-ward first):")
    print(f"  {'rank':>4}  {'feature':>7}  {'diff':>9}  {'pos_mean':>9}  {'neg_mean':>9}")
    for rank, fid in enumerate(feature_ids[:20]):
        print(
            f"  {rank:>4}  {fid:>7}  "
            f"{diff[fid].item():>+9.4f}  "
            f"{mean_pos[fid].item():>9.4f}  "
            f"{mean_neg[fid].item():>9.4f}"
        )
    print(f"[select] wrote {args.out} ({len(feature_ids)} features)")


if __name__ == "__main__":
    main()
