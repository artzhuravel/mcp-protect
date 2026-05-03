"""Collect last-token residual-stream activations for (y_pos, y_neg) pairs.

For each row in the input jsonl:
  ctx        = chat_template(system_prompt + user_query)
  full_pos   = ctx + y_pos
  full_neg   = ctx + y_neg
  H_pos[i]   = residual stream at layer L, last *content* token of y_pos
  H_neg[i]   = residual stream at layer L, last *content* token of y_neg

"Last content token" = last token before any chat-template terminator
(<|im_end|>, <end_of_turn>, <|eot_id|>). Different source models in MCPTox
emit different terminators, so we strip them by string match before tokenizing.

Saves (one subdir per layer):
  <out>/L<NN>/H_pos.pt          [N, d_model] float16
  <out>/L<NN>/H_neg.pt          [N, d_model] float16
  <out>/L<NN>/diffmean_vec.pt   [d_model]    float16
  <out>/index.jsonl             one row per index with {id, source, tags}

Usage:
    python -m diffmean.collect_activations \\
        --in    diffmean/outputs/mcptox_pairs.clean.jsonl \\
        --out   diffmean/outputs/acts/gemma2-9b \\
        --model google/gemma-2-9b-it \\
        --layers 12,16,20,24,28
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_PIPELINE_ROOT = _THIS.parent
if str(_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT.parent))

from diffmean.schema import read_jsonl  # noqa: E402


# Chat-template terminators from the various source models in MCPTox + common ones.
# Stripped from y_pos / y_neg before tokenization so the "last token" is real content.
TERMINATORS = [
    "<|im_end|>",        # Phi-4, Qwen
    "<end_of_turn>",     # Gemma
    "<|eot_id|>",        # Llama-3
    "<|endoftext|>",
    "</s>",
]


def _strip_terminators(text: str) -> str:
    text = text.rstrip()
    changed = True
    while changed:
        changed = False
        for term in TERMINATORS:
            if text.endswith(term):
                text = text[: -len(term)].rstrip()
                changed = True
    return text


def _build_chat_prompt(tokenizer, system_prompt: str, user_query: str) -> str:
    """Render system+user as a chat-templated string up to the assistant turn.
    Gemma's template doesn't support a system role, so we prepend it to user."""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Gemma path: fold system into user
        merged = f"{system_prompt}\n\n{user_query}" if system_prompt else user_query
        messages = [{"role": "user", "content": merged}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _decision_token_pos(tokenizer, ctx: str, full_text: str) -> int:
    """Position of the *decision token* in the tokenized full_text.

    For thinking-trace pairs, that's the last token of the response that occurs
    at-or-before the closing </think> tag — i.e. the moment the model commits
    to a decision but hasn't yet emitted the tool call. We locate it by string
    search on the full text and convert offset → token index.

    Returns -1 (= true last token, the legacy behaviour) when no </think> is
    found, so this is a no-op for non-thinking inputs.
    """
    lower = full_text.lower()
    end = lower.rfind("</think>")
    if end < 0:
        return -1
    decision_char = end + len("</think>")
    # Tokenize once with offsets so we can map char-pos → token-pos.
    enc = tokenizer(full_text, return_offsets_mapping=True,
                    add_special_tokens=False)
    offsets = enc["offset_mapping"]
    for i, (a, b) in enumerate(offsets):
        if b >= decision_char:
            return i
    return -1


@torch.no_grad()
def _residual_at(model, tokenizer, full_text: str, layers: list[int],
                 device: str, max_len: int, ctx: str) -> dict[int, torch.Tensor]:
    """Forward `full_text` once, return residual at each layer in `layers` at
    either the last token (default) or at the closing-</think> position when
    the response contains a thinking trace."""
    enc = tokenizer(full_text, return_tensors="pt", truncation=True,
                    max_length=max_len, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn,
                output_hidden_states=True, use_cache=False)
    # Pick the position to read activations from.
    pos = _decision_token_pos(tokenizer, ctx, full_text)
    if pos < 0 or pos >= input_ids.shape[1]:
        pos = input_ids.shape[1] - 1
    # hidden_states: tuple of (n_layers+1) tensors [1, T, d_model]
    # index 0 = embedding output; layer L's output = hidden_states[L+1]
    return {
        L: out.hidden_states[L + 1][0, pos, :].detach().to(torch.float16).cpu()
        for L in layers
    }


# Backwards-compat alias used by older callers.
_last_token_residual = _residual_at


def run(in_path: Path, out_dir: Path, model_name: str, layers: list[int],
        device: str, max_len: int, limit: int | None, dtype: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = list(read_jsonl(str(in_path)))
    if limit:
        rows = rows[:limit]
    print(f"[collect] {len(rows)} rows, model={model_name}, layers={layers}, device={device}",
          file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device,
    )
    model.eval()

    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is not None:
        for L in layers:
            if not (0 <= L < n_layers):
                raise SystemExit(f"layer {L} out of range [0,{n_layers})")

    out_dir.mkdir(parents=True, exist_ok=True)
    # One growing list per layer: each row will append a [d_model] vector to both pos and neg.
    H_pos: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    H_neg: dict[int, list[torch.Tensor]] = {L: [] for L in layers}

    with (out_dir / "index.jsonl").open("w", encoding="utf-8") as idx_f:
        for i, row in enumerate(rows):
            # Render system+user as the chat-templated prefix (up to assistant turn).
            ctx = _build_chat_prompt(tokenizer, row.get("system_prompt", ""),
                                     row.get("user_query", ""))
            # Drop chat-template terminators so the last token is real content,
            # not a Phi/Gemma EOS marker that would dominate the DiffMean direction.
            y_pos = _strip_terminators(row.get("y_pos", ""))
            y_neg = _strip_terminators(row.get("y_neg", ""))
            if not y_pos or not y_neg:
                continue

            try:
                # One forward per response → dict {layer: [d_model]} for all requested layers.
                h_pos = _residual_at(model, tokenizer, ctx + y_pos,
                                     layers, device, max_len, ctx)
                h_neg = _residual_at(model, tokenizer, ctx + y_neg,
                                     layers, device, max_len, ctx)
            except Exception as e:
                # OOM / tokenization edge cases: skip the row, keep the run going.
                print(f"  [{i}] skip ({type(e).__name__}: {str(e)[:120]})",
                      file=sys.stderr)
                continue

            # Slot each layer's vector into its respective list. Order is preserved
            # across layers, so H_pos[L][k] and H_neg[L][k] always refer to the same row.
            for L in layers:
                H_pos[L].append(h_pos[L])
                H_neg[L].append(h_neg[L])
            kept = len(H_pos[layers[0]])
            # Sidecar: row k in H_pos.pt corresponds to this id/tags. Used downstream
            # for per-paradigm slicing (e.g. AUC on Template-1 only).
            idx_f.write(json.dumps({
                "i": kept - 1,
                "id": row.get("id"),
                "source": row.get("source"),
                "tags": row.get("tags", {}),
            }, ensure_ascii=False) + "\n")

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(rows)}] kept={kept}", file=sys.stderr)

    # Stack lists → tensors and persist. One subdir per layer keeps the analysis
    # script (compute_vector.py) trivial: just iterate L*/ dirs.
    for L in layers:
        H_pos_t = torch.stack(H_pos[L])  # [N, d_model]
        H_neg_t = torch.stack(H_neg[L])
        sub = out_dir / f"L{L:02d}"
        sub.mkdir(parents=True, exist_ok=True)
        torch.save(H_pos_t, sub / "H_pos.pt")
        torch.save(H_neg_t, sub / "H_neg.pt")
        # The DiffMean steering vector itself: pos centroid − neg centroid.
        # Cast to float for the mean (avoid fp16 underflow), then back to fp16 for storage.
        v = (H_pos_t.float().mean(0) - H_neg_t.float().mean(0)).to(torch.float16)
        torch.save(v, sub / "diffmean_vec.pt")
        print(f"[collect] L{L}: H_pos {tuple(H_pos_t.shape)}, vec {tuple(v.shape)} → {sub}",
              file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_dir", type=Path, required=True)
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--layers", default="20",
                   help="Comma-separated layer indices, e.g. 12,16,20,24,28")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   ("mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None, help="Smoke-test limit.")
    args = p.parse_args()

    layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    run(args.in_path, args.out_dir, args.model, layers,
        args.device, args.max_len, args.limit, args.dtype)


if __name__ == "__main__":
    main()
