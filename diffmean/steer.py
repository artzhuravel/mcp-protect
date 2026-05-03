"""Apply the DiffMean steering vector at generation time and dump completions.

We register a forward-hook on `model.layers[L]` that adds `alpha * v` to the
residual stream output (last token only by default — that's what AxBench/RePS
do for steering eval; full-sequence is available with --all-tokens).

Negative alpha → suppress the attack-compliance direction (defense).
Positive alpha → amplify it (sanity check that the vector actually steers).

Output is jsonl with the original row + a `steered` field per (alpha, layer):
  {... original fields ..., "steered": [{"alpha": -4.0, "layer": 20,
                                         "completion": "..."}]}

Usage:
    python -m diffmean.steer \\
        --in     diffmean/outputs/mcptox_pairs.clean.jsonl \\
        --vec    diffmean/outputs/acts/gemma2-9b/L20/diffmean_vec.pt \\
        --layer  20 \\
        --alphas -8,-4,-2,0,2,4 \\
        --out    diffmean/outputs/steered/gemma2-9b-L20.jsonl \\
        --limit  100
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_PIPELINE_ROOT = _THIS.parent
if str(_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT.parent))

from diffmean.schema import read_jsonl  # noqa: E402


def _get_layer_module(model, layer: int):
    """Return the nn.Module whose forward output is the residual stream after
    layer `layer`. Works for Gemma-2, Llama, Qwen-2/3, Phi-3/4 — they all
    expose `model.model.layers[L]`."""
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise SystemExit("could not locate model.model.layers — adjust _get_layer_module")
    return layers[layer]


@contextmanager
def steering_hook(model, layer: int, v: torch.Tensor, alpha: float,
                  all_tokens: bool):
    """Add `alpha * v` to the residual stream coming out of `model.layers[L]`.
    Decoder layers return a tuple `(hidden_states, ...)`; we modify [0]."""
    if alpha == 0.0:
        yield
        return

    module = _get_layer_module(model, layer)

    def hook(_mod, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        delta = (alpha * v).to(h.dtype).to(h.device)
        if all_tokens:
            h = h + delta
        else:
            # Last token only — matches generation-time intervention
            h[..., -1, :] = h[..., -1, :] + delta
        return (h, *rest) if rest is not None else h

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _build_chat_prompt(tokenizer, system_prompt: str, user_query: str) -> str:
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        merged = f"{system_prompt}\n\n{user_query}" if system_prompt else user_query
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}],
            tokenize=False, add_generation_prompt=True,
        )


@torch.no_grad()
def _generate(model, tokenizer, prompt: str, alpha: float, layer: int,
              v: torch.Tensor, max_new: int, all_tokens: bool, device: str) -> str:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with steering_hook(model, layer, v, alpha, all_tokens):
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def run(in_path: Path, out_path: Path, model_name: str, vec_path: Path,
        layer: int, alphas: list[float], device: str, dtype: str,
        max_new: int, limit: int | None, all_tokens: bool) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = list(read_jsonl(str(in_path)))
    if limit:
        rows = rows[:limit]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device,
    )
    model.eval()

    v = torch.load(vec_path).float()
    v = v / (v.norm() + 1e-9)  # unit vector — alphas express magnitude in σ-ish units
    print(f"[steer] {len(rows)} rows × {len(alphas)} alphas={alphas} at L{layer}, "
          f"||v|| normalized to 1", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            prompt = _build_chat_prompt(tokenizer,
                                        row.get("system_prompt", ""),
                                        row.get("user_query", ""))
            steered = []
            for a in alphas:
                try:
                    text = _generate(model, tokenizer, prompt, a, layer, v,
                                     max_new, all_tokens, device)
                except Exception as e:
                    text = f"[steer_error: {type(e).__name__}: {str(e)[:120]}]"
                steered.append({"alpha": a, "layer": layer, "completion": text})

            out_row = dict(row)
            out_row["steered"] = steered
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(rows)}]", file=sys.stderr)

    print(f"[steer] → {out_path}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    p.add_argument("--vec", type=Path, required=True,
                   help="Path to diffmean_vec.pt")
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--alphas", default="-8,-4,-2,0,2,4",
                   help="Comma-separated steering coefficients.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   ("mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--all-tokens", action="store_true",
                   help="Steer every token in the sequence (default: last token only).")
    args = p.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    run(args.in_path, args.out_path, args.model, args.vec, args.layer,
        alphas, args.device, args.dtype, args.max_new, args.limit,
        args.all_tokens)


if __name__ == "__main__":
    main()
