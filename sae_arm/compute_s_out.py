from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_utils import QwenScopeSAE


def compute_s_out_one(
    model, tokenizer,
    sae: QwenScopeSAE, layer: int,
    features: list[int], logit_lens_indices: list[int],
    sentence: str, amp_factor: float,
) -> float:
    """Port of Arad et al.'s get_output_score (src/output_score.py + AmlifySAEHook).
    Clamp the given feature(s) at this layer, run the model on `sentence`,
    score how high the feature's logit-lens tokens rank in the resulting
    output distribution: rank_output_score × top_token_score."""

    def sae_hook(module, args, output):
        output_tensor = output[0]
        feature_acts = sae.encode(output_tensor)

        with torch.no_grad():
            x_reconstruct_clean = sae.decode(sae.encode(output_tensor))
        sae_error = output_tensor.to(torch.float32) - x_reconstruct_clean.to(torch.float32)

        max_act_value = torch.max(feature_acts[:, -1, :]).item()
        for feature in features:
            feature_acts[:, -1, feature] += max_act_value * amp_factor

        sae_out = sae.decode(feature_acts).to(torch.float32) + sae_error
        return tuple([sae_out.to(output_tensor.dtype)] + list(output[1:]))

    model_block_to_hook = model.model.layers[layer]
    handle = model_block_to_hook.register_forward_hook(sae_hook, always_call=True)

    inputs = tokenizer(sentence, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)
    handle.remove()

    logits_after = outputs.logits[:, -1]
    intervention_logits = logits_after.squeeze()
    intervention_probs = torch.softmax(intervention_logits, dim=0).detach().cpu()

    vocab_size = intervention_probs.shape[0]
    tokens_argsort = torch.argsort(intervention_probs, dim=0, descending=True)
    ll_tokens_ranks = [
        (tokens_argsort == ll_token).nonzero(as_tuple=True)[0].item()
        for ll_token in logit_lens_indices
    ]
    top_token_score = torch.max(intervention_probs[logit_lens_indices]).item()
    rank_output_score = 1 - (min(ll_tokens_ranks) / vocab_size)

    return rank_output_score * top_token_score


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--sae-path", type=Path, required=True)
    p.add_argument("--features-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--neutral-prompt", default="In my experience,")
    p.add_argument("--amp-factor", type=float, default=10.0)
    p.add_argument("--logit-lens-top-k", type=int, default=20)
    args = p.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, device_map=device,
    ).eval()
    weights = torch.load(args.sae_path, map_location="cpu", weights_only=True)
    sae = QwenScopeSAE(weights, k=50, device=device, dtype=dtype)
    features_by_layer = {
        int(k): [int(v) for v in vs]
        for k, vs in json.loads(args.features_file.read_text()).items()
    }

    # Top-K logit-lens tokens per feature, precomputed in one big matmul.
    with torch.no_grad():
        ll_logits = model.lm_head(model.model.norm(sae.W_dec))
        ll_indices = torch.topk(ll_logits, k=args.logit_lens_top_k, dim=-1).indices.cpu()

    output_scores = json.loads(args.out.read_text()) if args.out.exists() else {}

    for layer, features in features_by_layer.items():
        for feature in tqdm(features):
            key = f"{layer}_{feature}"
            if key in output_scores:
                continue
            ll = ll_indices[feature].tolist()
            output_scores[key] = compute_s_out_one(
                model, tokenizer, sae, layer, [feature], ll,
                args.neutral_prompt, args.amp_factor,
            )
            torch.cuda.empty_cache()
            gc.collect()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output_scores))


if __name__ == "__main__":
    main()
