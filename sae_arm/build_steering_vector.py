from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_utils import QwenScopeSAE, out_dir_for, stratify_activations

ROOT = Path(__file__).parent
# Activations live at the team's canonical location (committed in git via the
# diffmean/ subtree). sae_arm/acts/ is just a local convenience mirror for
# visual browsing on the laptop and is gitignored.
ACTS_DIR = ROOT.parent / "sae_arm" / "acts"


def _check_cache_provenance(path: Path, cached: dict, expected: dict) -> None:
    """Fail fast if any expected provenance field disagrees with the cached
    value. Forces explicit cache invalidation on the user's part rather than
    silently producing a vector built from stale candidates / scores."""
    for k, v in expected.items():
        if cached.get(k) != v:
            raise SystemExit(
                f"{path.name} cache mismatch on {k!r}:\n"
                f"  cached:  {cached.get(k)!r}\n"
                f"  current: {v!r}\n"
                f"To rebuild: rm {path}\n"
            )


def select_candidates(
    H_pos: torch.Tensor,
    H_neg: torch.Tensor,
    sae: QwenScopeSAE,
    top_k: int,
    chunk: int = 256,
) -> tuple[list[int], list[int]]:
    """Return (feature_ids, signs) for the top-K features by |mean_pos - mean_neg|.

    Sign tells you which class the feature fires more on:
      sign = +1: feature fires more on H_pos (compliance-correlated)
      sign = -1: feature fires more on H_neg (resistance-correlated)

    The sign matters at assembly time: a resistance-correlated feature's
    decoder column points toward "resistance," so it must be subtracted (not
    added) when building a "compliance direction." Tracking sign here and
    applying it in assemble_steering_vector keeps v cleanly oriented.
    """
    device = sae.W_enc.device
    sums_pos = torch.zeros(sae.d_sae, dtype=torch.float32, device=device)
    for i in range(0, H_pos.shape[0], chunk):
        sums_pos += sae.encode(H_pos[i:i + chunk].to(device)).float().sum(0)
    sums_neg = torch.zeros(sae.d_sae, dtype=torch.float32, device=device)
    for i in range(0, H_neg.shape[0], chunk):
        sums_neg += sae.encode(H_neg[i:i + chunk].to(device)).float().sum(0)
    mean_pos = sums_pos / max(H_pos.shape[0], 1)
    mean_neg = sums_neg / max(H_neg.shape[0], 1)
    diff = (mean_pos - mean_neg).cpu()
    topk_idx = diff.abs().topk(top_k).indices
    sign_vals = diff[topk_idx].sign().to(torch.int64)
    # Drop sign==0 candidates: features that fire identically (usually zero)
    # on both classes contribute nothing to v and waste an S_out call. Common
    # in stratified slices where many TopK-SAE features are silent in both.
    keep = sign_vals != 0
    feature_ids = topk_idx[keep].tolist()
    signs = sign_vals[keep].tolist()
    return feature_ids, signs


def compute_s_out_one(
    model, tokenizer,
    sae: QwenScopeSAE, layer: int,
    features: list[int], logit_lens_indices: list[int],
    sentence: str, amp_factor: float,
) -> float:
    def sae_hook(module, args, output):
        # transformers v5 Qwen3 layers return the hidden-states tensor
        # directly; older versions returned a (hidden_states, ...) tuple.
        # Handle both so this works across pinned-and-unpinned envs.
        is_tuple = isinstance(output, (tuple, list))
        output_tensor = output[0] if is_tuple else output
        feature_acts = sae.encode(output_tensor)

        with torch.no_grad():
            x_reconstruct_clean = sae.decode(sae.encode(output_tensor))
        sae_error = output_tensor.to(torch.float32) - x_reconstruct_clean.to(torch.float32)

        max_act_value = torch.max(feature_acts[:, -1, :]).item()
        for feature in features:
            feature_acts[:, -1, feature] += max_act_value * amp_factor

        sae_out = sae.decode(feature_acts).to(torch.float32) + sae_error
        sae_out_typed = sae_out.to(output_tensor.dtype)
        if is_tuple:
            return tuple([sae_out_typed] + list(output[1:]))
        return sae_out_typed

    layer_module = model.model.layers[layer]
    handle = layer_module.register_forward_hook(sae_hook, always_call=True)

    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    handle.remove()

    intervention_probs = torch.softmax(outputs.logits[:, -1].squeeze().float(), dim=0).cpu()
    vocab_size = intervention_probs.shape[0]
    tokens_argsort = torch.argsort(intervention_probs, dim=0, descending=True)
    ll_tokens_ranks = [
        (tokens_argsort == ll_token).nonzero(as_tuple=True)[0].item()
        for ll_token in logit_lens_indices
    ]
    top_token_score = torch.max(intervention_probs[logit_lens_indices]).item()
    rank_output_score = 1 - (min(ll_tokens_ranks) / vocab_size)
    return rank_output_score * top_token_score


def compute_s_out_table(
    model, tokenizer, sae: QwenScopeSAE, layer: int,
    candidate_ids: list[int], ll_indices: torch.Tensor,
    neutral_prompt: str, amp_factor: float,
) -> dict[int, float]:
    """Apply compute_s_out_one to every candidate, returning a dict {feature_id: score}."""
    scores: dict[int, float] = {}
    for feature_idx in tqdm(candidate_ids, desc="S_out"):
        scores[feature_idx] = compute_s_out_one(
            model, tokenizer, sae, layer,
            [feature_idx], ll_indices[feature_idx].tolist(),
            neutral_prompt, amp_factor,
        )
        torch.cuda.empty_cache()
        gc.collect()
    return scores


def assemble_steering_vector(
    sae: QwenScopeSAE,
    candidate_ids: list[int],
    candidate_signs: list[int],
    s_out: dict[int, float],
    threshold: float,
    top_n: int | None,
) -> tuple[torch.Tensor, list[tuple[int, int, float]]]:
    """Arad et al.'s filter recipe + sign-aware decoder sum.

    For each surviving feature i:
      - sign = +1 (compliance-correlated) → add  +W_dec[i, :]
      - sign = -1 (resistance-correlated) → add  -W_dec[i, :]
    Net result: v points "compliance-ward" regardless of which class the
    feature fired more on. Steering convention: +α·v pushes toward
    compliance, -α·v defends.
    """
    sign_map = dict(zip(candidate_ids, candidate_signs))
    scored = [
        (fid, sign_map[fid], s_out[fid])
        for fid in candidate_ids
        if s_out.get(fid, 0.0) >= threshold
    ]
    scored.sort(key=lambda t: -t[2])
    if top_n is not None:
        scored = scored[:top_n]
    if not scored:
        raise SystemExit(f"no features survive S_out >= {threshold}")
    v = torch.zeros(sae.d_model, dtype=torch.float32)
    for fid, sign, _ in scored:
        v = v + sign * sae.W_dec[fid, :].cpu().float()
    return v, scored


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True,
                   choices=["qwen3-thinking-decision", "qwen3-v2-contrast"])
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--paradigm", default=None,
                   help="One of Template-1, Template-2, Template-3 (or omit).")
    p.add_argument("--security-risk", default=None,
                   help="Use the form from index.jsonl (with spaces, e.g. 'Credential Leakage').")
    p.add_argument("--sae-path", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100,
                   help="Step-1: how many candidates to rank by activation diff.")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Step-3: drop features with S_out below this.")
    p.add_argument("--top-n", type=int, default=None,
                   help="Step-3: optionally keep only the top-N filtered features.")
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--neutral-prompt", default="In my experience,")
    p.add_argument("--amp-factor", type=float, default=10.0)
    p.add_argument("--logit-lens-top-k", type=int, default=20)
    args = p.parse_args()
    print(args)

    out_dir = out_dir_for(args.set, args.layer, args.paradigm, args.security_risk)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.json"
    s_out_path = out_dir / "s_out.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    weights = torch.load(args.sae_path, map_location="cpu", weights_only=True)
    sae = QwenScopeSAE(weights, k=50, device=device, dtype=dtype)

    # Step 1: candidate identification. Cached. Cache validity depends on
    # which SAE was used and how many candidates we asked for — both are
    # checked on read; mismatch fails fast.
    step1_provenance = {
        "sae_path": str(args.sae_path),
        "top_k": args.top_k,
    }
    if features_path.exists():
        cached = json.loads(features_path.read_text())
        _check_cache_provenance(features_path, cached, step1_provenance)
        candidate_ids = cached["feature_ids"]
        candidate_signs = cached["signs"]
        print(f"[step 1] cached: {len(candidate_ids)} candidates  ({features_path})")
    else:
        print("[step 1] selecting candidates by activation diff (Path B) ...")
        acts_dir = ACTS_DIR / args.set / f"L{args.layer}"
        index_path = ACTS_DIR / args.set / "index.jsonl"
        H_pos = torch.load(acts_dir / "H_pos.pt", map_location=device, weights_only=True).float()
        H_neg = torch.load(acts_dir / "H_neg.pt", map_location=device, weights_only=True).float()
        if args.paradigm or args.security_risk:
            H_pos, H_neg = stratify_activations(
                H_pos, H_neg, index_path,
                paradigm=args.paradigm, security_risk=args.security_risk,
            )
            print(f"[step 1] stratified to H_pos={tuple(H_pos.shape)}  H_neg={tuple(H_neg.shape)}")
        if H_pos.shape[0] == 0 or H_neg.shape[0] == 0:
            raise SystemExit("stratified H_pos or H_neg is empty; widen the filter")
        candidate_ids, candidate_signs = select_candidates(H_pos, H_neg, sae, args.top_k)
        features_path.write_text(json.dumps({
            **step1_provenance,
            "feature_ids": candidate_ids,
            "signs": candidate_signs,
        }))
        print(f"[step 1] wrote {features_path} "
              f"({len(candidate_ids)} candidates: "
              f"{sum(1 for s in candidate_signs if s > 0)} comply-correlated, "
              f"{sum(1 for s in candidate_signs if s < 0)} resist-correlated)")

    # Step 2: S_out scoring. Cached. Cache validity depends on the SAE,
    # the neutral prompt, the amplification factor, and the logit-lens k.
    step2_provenance = {
        "sae_path": str(args.sae_path),
        "model_name": args.model_name,        # affects lm_head logit-lens AND S_out scoring
        "neutral_prompt": args.neutral_prompt,
        "amp_factor": args.amp_factor,
        "logit_lens_top_k": args.logit_lens_top_k,
    }
    ll_path = out_dir / "feature_logit_lens.json"
    if s_out_path.exists():
        cached = json.loads(s_out_path.read_text())
        _check_cache_provenance(s_out_path, cached, step2_provenance)
        s_out = {int(k): v for k, v in cached["scores"].items()}
        print(f"[step 2] cached: {len(s_out)} scores  ({s_out_path})")
        if not ll_path.exists():
            print(f"[step 2] WARNING: {ll_path.name} missing — to backfill, "
                  f"`rm {s_out_path}` and re-run step 2.")
    else:
        print(f"[step 2] computing S_out for {len(candidate_ids)} candidates ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=dtype, device_map=device,
        ).eval()
        with torch.no_grad():
            ll_logits = model.lm_head(model.model.norm(sae.W_dec))
            ll_indices = torch.topk(ll_logits, k=args.logit_lens_top_k, dim=-1).indices.cpu()

        # Decode the candidate features' logit-lens token IDs to strings — the
        # cheapest first-pass interpretation of "what does each feature
        # represent?" Tokens like " apple", " Apple" suggest a fruit-y
        # feature; tokens like " refuse", " I", " cannot" suggest refusal.
        # Self-contained file (provenance-tagged), pullable to the laptop
        # without the SAE/model weights.
        ll_provenance = {
            "sae_path": str(args.sae_path),
            "model_name": args.model_name,
            "logit_lens_top_k": args.logit_lens_top_k,
        }
        ll_decoded = {
            str(fid): [tokenizer.decode([tid]) for tid in ll_indices[fid].tolist()]
            for fid in candidate_ids
        }
        ll_path.write_text(json.dumps({**ll_provenance, "tokens": ll_decoded},
                                      indent=2, ensure_ascii=False))
        print(f"[step 2] wrote {ll_path} ({len(ll_decoded)} features × top-{args.logit_lens_top_k})")

        s_out = compute_s_out_table(
            model, tokenizer, sae, args.layer, candidate_ids, ll_indices,
            args.neutral_prompt, args.amp_factor,
        )
        s_out_path.write_text(json.dumps({
            **step2_provenance,
            "scores": {str(k): v for k, v in s_out.items()},
        }))
        print(f"[step 2] wrote {s_out_path}")

    # Step 3: filter and assemble.
    v, kept = assemble_steering_vector(
        sae, candidate_ids, candidate_signs, s_out, args.threshold, args.top_n,
    )

    suffix = f"_top{args.top_n}" if args.top_n else ""
    out_pt = out_dir / f"sae_thr{args.threshold:g}{suffix}.pt"
    torch.save(v, out_pt)

    # In-sample AUC of the steering vector against the same H_pos/H_neg (stratified the same way). Mann-Whitney U based.
    acts_dir = ACTS_DIR / args.set / f"L{args.layer}"
    index_path = ACTS_DIR / args.set / "index.jsonl"
    H_pos = torch.load(acts_dir / "H_pos.pt", map_location="cpu", weights_only=True).float()
    H_neg = torch.load(acts_dir / "H_neg.pt", map_location="cpu", weights_only=True).float()
    if args.paradigm or args.security_risk:
        H_pos, H_neg = stratify_activations(
            H_pos, H_neg, index_path,
            paradigm=args.paradigm, security_risk=args.security_risk,
        )
    v_unit = v / (v.norm() + 1e-9)
    scores_pos = H_pos @ v_unit
    scores_neg = H_neg @ v_unit
    # Cosine similarities.
    H_pos_unit = H_pos / (H_pos.norm(dim=-1, keepdim=True) + 1e-9)
    H_neg_unit = H_neg / (H_neg.norm(dim=-1, keepdim=True) + 1e-9)
    cos_pos_mean = float((H_pos_unit @ v_unit).mean())
    cos_neg_mean = float((H_neg_unit @ v_unit).mean())
    s = torch.cat([scores_pos, scores_neg])
    y = torch.cat([torch.ones_like(scores_pos), torch.zeros_like(scores_neg)])
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    n_pos, n_neg = scores_pos.numel(), scores_neg.numel()
    sum_ranks_pos = ranks[y == 1].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if n_pos and n_neg else float("nan")

    # Flag synthetic / smoke-test artifacts so they're not mistaken for
    # publishable results. Anything under .test_artifacts/ is non-production.
    is_synthetic = ".test_artifacts" in str(args.sae_path)
    metadata = {
        "WARNING_synthetic_sae": "smoke-test only, NOT a real result" if is_synthetic else None,
        "set": args.set,
        "layer": args.layer,
        "paradigm": args.paradigm,
        "security_risk": args.security_risk,
        "threshold": args.threshold,
        "top_n": args.top_n,
        "n_features_kept": len(kept),
        "n_candidates": len(candidate_ids),
        "features": [{"id": fid, "sign": sign, "s_out": s} for fid, sign, s in kept],
        "norm": float(v.norm()),
        "d_model": sae.d_model,
        "sae_path": str(args.sae_path),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "score_pos_mean": float(scores_pos.mean()),
        "score_neg_mean": float(scores_neg.mean()),
        "cos_pos_mean": cos_pos_mean,
        "cos_neg_mean": cos_neg_mean,
        "auc": float(auc),
    }
    out_pt.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))
    print(f"[step 3] {len(kept)}/{len(candidate_ids)} survive S_out >= {args.threshold}")
    print(f"[step 3] wrote {out_pt}  (||v||={float(v.norm()):.3f}, AUC={auc:.4f})")


if __name__ == "__main__":
    main()
