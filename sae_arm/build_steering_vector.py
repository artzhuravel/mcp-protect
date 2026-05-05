"""Step 3: filter candidates by S_out and produce a steering vector .pt file.

Inputs:
  --features-file  features.json from select_candidates.py
  --s-out-file     s_out.json from compute_s_out.py
  --sae-path       Qwen-Scope layer{N}.sae.pt
  --layer          which layer to build the vector for
  --threshold      default 0.1 (Arad et al. paper's recommended cut)
  --top-n          optional: keep only the top-N filtered features by S_out

Output:
  --out  the steering vector as a (d_model,) tensor
         alongside <out>.meta.json with provenance

The vector is the unscaled sum of decoder columns of the surviving features:
    v = sum_{i in filtered} W_dec[i, :]
The interventions.yaml `scale` knob handles magnitude at inference time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sae_utils import QwenScopeSAE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-file", type=Path, required=True)
    p.add_argument("--s-out-file", type=Path, required=True)
    p.add_argument("--sae-path", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    raw_features = json.loads(args.features_file.read_text())
    features_by_layer = {int(k): [int(v) for v in vs] for k, vs in raw_features.items()}
    s_out_scores = json.loads(args.s_out_file.read_text())

    if args.layer not in features_by_layer:
        raise SystemExit(
            f"layer {args.layer} not in {args.features_file}; "
            f"have {list(features_by_layer)}"
        )
    candidates = features_by_layer[args.layer]

    scored: list[tuple[int, float]] = []
    missing = 0
    for fid in candidates:
        key = f"{args.layer}_{fid}"
        if key not in s_out_scores:
            missing += 1
            continue
        s = s_out_scores[key]
        if s >= args.threshold:
            scored.append((fid, s))
    if missing:
        print(f"[build] WARNING: {missing} candidates missing S_out scores; ignored")
    scored.sort(key=lambda t: -t[1])
    if args.top_n is not None:
        scored = scored[: args.top_n]

    if not scored:
        raise SystemExit(f"no features survive S_out >= {args.threshold}")

    print(
        f"[build] {len(scored)}/{len(candidates)} features survive "
        f"S_out >= {args.threshold}:"
    )
    for rank, (fid, s) in enumerate(scored[:10]):
        print(f"  #{rank:>2}  feature={fid:>6}  S_out={s:.4f}")
    if len(scored) > 10:
        print(f"  ... + {len(scored) - 10} more")

    print(f"[build] loading SAE  {args.sae_path}")
    sae = QwenScopeSAE.from_qwen_scope_file(args.sae_path)

    steering = torch.zeros(sae.d_model, dtype=torch.float32)
    for fid, _ in scored:
        steering = steering + sae.W_dec[fid, :].float()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(steering, args.out)

    metadata = {
        "layer": args.layer,
        "threshold": args.threshold,
        "top_n": args.top_n,
        "n_features_kept": len(scored),
        "n_candidates": len(candidates),
        "features": [{"id": fid, "s_out": s} for fid, s in scored],
        "norm": float(steering.norm()),
        "d_model": sae.d_model,
        "sae_path": str(args.sae_path),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))
    print(f"[build] wrote {args.out}  (||v||={float(steering.norm()):.3f})")
    print(f"[build] wrote {args.out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
