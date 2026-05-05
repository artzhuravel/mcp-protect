"""Step 2: compute Arad et al.'s output score (S_out) per candidate feature.

Algorithm, mirroring src/output_score.py + src/sae_utils.py from
github.com/technion-cs-nlp/saes-are-good-for-steering with Qwen-Scope as the
SAE backend instead of Gemma-Scope:

  1. Precompute, per feature i, its top-K logit-lens tokens by projecting
     SAE.W_dec[i] through the model's final LayerNorm + lm_head.
  2. For each candidate feature i:
        - Hook layer L's residual-stream output. The hook re-encodes the
          activation through the SAE, boosts feature i at the last token,
          decodes, and adds back the SAE reconstruction error so non-SAE-
          captured information passes through unchanged.
        - Run forward on the neutral prompt ("In my experience,").
        - Read the final-token logits, find where the top-K logit-lens tokens
          rank in the model's actual output distribution, and compute
              S_out = (1 − min_rank/|V|) · max_prob_among_ll_tokens
  3. Cache scores incrementally to JSON so a crash mid-run doesn't lose work.

Inputs:
  --features-file features.json from select_candidates.py
  --sae-path      Qwen-Scope layer{N}.sae.pt for the same layer as the features
  --model-name    HF id of the post-trained model (default Qwen/Qwen3-8B)
  --neutral-prompt  default "In my experience," (Arad et al.'s choice)
  --amp-factor    default 10 (paper uses 10 for S_out scoring)

Output:
  s_out.json: {"<layer>_<feature>": <score>, ...}  ← consumed by build_steering_vector.py
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_utils import QwenScopeSAE


def cache_logit_lens(
    sae: QwenScopeSAE,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    top_k: int = 20,
) -> torch.Tensor:
    """Project every SAE decoder column through the unembedding to get top-k tokens
    per feature. Returns indices: [d_sae, top_k]."""
    fn = final_norm.cpu()
    lh = lm_head.cpu()
    W_dec = sae.W_dec.detach().cpu().float()  # [d_sae, d_model]
    with torch.no_grad():
        normed = fn(W_dec)
        logits = lh(normed)  # [d_sae, vocab]
        probs = torch.softmax(logits.float(), dim=-1)
        topk = torch.topk(probs, k=top_k, dim=-1)
    return topk.indices  # [d_sae, top_k]


def make_amplify_hook(sae: QwenScopeSAE, feature_idx: int, amp_factor: float):
    """Per Arad et al. AmlifySAEHook: encode → boost feature i at last token →
    decode + add back the reconstruction error.

    The error term keeps non-SAE-captured signal from leaking into the boost.
    """
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        # Reference encode/decode to extract the SAE reconstruction error.
        with torch.no_grad():
            a_ref = sae.encode(h)
            recon_ref = sae.decode(a_ref)
        sae_error = h.to(torch.float32) - recon_ref.to(torch.float32)

        # Modified path: boost feature i at the last token by amp_factor × the
        # max activation across all features at that token.
        a_mod = sae.encode(h)
        max_act = a_mod[:, -1, :].max().item()
        a_mod[:, -1, feature_idx] = a_mod[:, -1, feature_idx] + max_act * amp_factor
        recon_mod = sae.decode(a_mod)

        out = recon_mod.to(torch.float32) + sae_error
        out = out.to(h.dtype)
        return (out,) + rest if rest is not None else out

    return hook


def compute_s_out_one(
    model,
    tokenizer,
    sae: QwenScopeSAE,
    layer: int,
    feature_idx: int,
    ll_token_indices: list[int],
    neutral_prompt: str,
    amp_factor: float,
    device: str,
) -> float:
    layer_module = model.model.layers[layer]
    hook = make_amplify_hook(sae, feature_idx, amp_factor)
    handle = layer_module.register_forward_hook(hook)

    inputs = tokenizer(neutral_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    try:
        with torch.inference_mode():
            outputs = model(**inputs)
    finally:
        handle.remove()

    logits = outputs.logits[:, -1].squeeze().float()
    probs = torch.softmax(logits, dim=0).cpu()
    vocab_size = probs.shape[0]

    # Rank of each logit-lens token in the actual output distribution.
    sorted_idx = torch.argsort(probs, descending=True)
    pos_in_sorted = torch.empty_like(sorted_idx)
    pos_in_sorted[sorted_idx] = torch.arange(vocab_size, dtype=sorted_idx.dtype)
    ll_ranks = [pos_in_sorted[t].item() for t in ll_token_indices]

    rank_score = 1.0 - (min(ll_ranks) / vocab_size)
    top_prob = max(probs[t].item() for t in ll_token_indices)
    return rank_score * top_prob


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--sae-path", type=Path, required=True)
    p.add_argument("--features-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--neutral-prompt", default="In my experience,")
    p.add_argument("--amp-factor", type=float, default=10.0)
    p.add_argument("--logit-lens-top-k", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--save-every", type=int, default=10,
        help="Persist the score JSON every N features in case of crash.",
    )
    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32

    print(f"[s_out] loading {args.model_name} on {args.device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, device_map=args.device,
    ).eval()

    print(f"[s_out] loading SAE from {args.sae_path}")
    sae = QwenScopeSAE.from_qwen_scope_file(args.sae_path).to(args.device, dtype=dtype)

    print(f"[s_out] loading features from {args.features_file}")
    raw_features = json.loads(args.features_file.read_text())
    features_by_layer = {int(k): [int(v) for v in vs] for k, vs in raw_features.items()}

    print(
        f"[s_out] caching logit lens (top-{args.logit_lens_top_k}) for all "
        f"{sae.d_sae} features ..."
    )
    ll_indices = cache_logit_lens(
        sae, model.model.norm, model.lm_head, top_k=args.logit_lens_top_k
    )
    print(f"[s_out]   logit-lens cache shape: {tuple(ll_indices.shape)}")

    scores: dict[str, float] = {}
    if args.out.exists():
        scores = json.loads(args.out.read_text())
        print(f"[s_out] resuming with {len(scores)} cached scores from {args.out}")

    total = sum(len(v) for v in features_by_layer.values())
    done = 0
    t0 = time.time()
    for layer, feature_ids in features_by_layer.items():
        for feature_idx in feature_ids:
            key = f"{layer}_{feature_idx}"
            done += 1
            if key in scores:
                continue
            ll_for_feature = ll_indices[feature_idx].tolist()
            s = compute_s_out_one(
                model, tokenizer, sae, layer, feature_idx, ll_for_feature,
                args.neutral_prompt, args.amp_factor, args.device,
            )
            scores[key] = s
            if done % args.save_every == 0:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(scores, indent=2))
            if done % 10 == 0:
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            print(
                f"  [{done:>4}/{total}] L{layer} f{feature_idx:>6}  "
                f"S_out = {s:.4f}    ({time.time()-t0:.0f}s elapsed)"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2))
    print(f"[s_out] wrote {len(scores)} scores to {args.out}")


if __name__ == "__main__":
    main()
