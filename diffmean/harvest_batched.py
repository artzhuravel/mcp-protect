"""Batched local-GPU harvest of (prompt → CoT rollout) for MCPTox.

Bypasses vf-eval + serve.py to avoid the 1-request-at-a-time HTTP serialization
we saw with serve.py. Loads the target model once, runs `model.generate` on
batches of `batch_size` prompts at a time. ~3-4x faster than serial on a 46GB
L40S with Qwen3-8B at bf16.

Pipeline:
  1. Read MCPTox cases from `mcptox_pairs.clean.jsonl` (already has system_prompt,
     user_query, poisoned_tool, tags).
  2. Render each as a chat-templated prompt.
  3. Batch-tokenize with left-padding (so generated tokens line up at the right
     edge per row).
  4. `model.generate(batch, max_new_tokens, do_sample=True, temperature)`.
  5. Decode each row's continuation only (skip the prompt tokens).
  6. Save raw rollouts (prompt, completion, ground-truth tags). Judging is done
     afterwards by `judge_rollouts.py` (async OpenRouter, fast).

Usage:
    python -m diffmean.harvest_batched \\
        --in   diffmean/outputs/mcptox_pairs.clean.jsonl \\
        --out  diffmean/outputs/qwen3_rollouts.raw.jsonl \\
        --model Qwen/Qwen3-8B \\
        --batch-size 4 \\
        --max-new-tokens 4000 \\
        --temperature 0.7 \\
        --num-examples 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_PIPELINE_ROOT = _THIS.parent
if str(_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT.parent))

from diffmean.schema import read_jsonl  # noqa: E402


def _build_chat_prompt(tokenizer, system_prompt: str, user_query: str) -> str:
    try:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        merged = (system_prompt + "\n\n" + user_query) if system_prompt else user_query
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}],
            tokenize=False, add_generation_prompt=True,
        )


@torch.no_grad()
def _generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int,
                    temperature: float, device: str, max_input_len: int) -> list[str]:
    """Tokenize with left-padding, generate, return decoded continuations."""
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_input_len, add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    in_len = input_ids.shape[1]

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    out = model.generate(**gen_kwargs)
    # out shape: [B, in_len + new]. Slice to continuation.
    cont = out[:, in_len:]
    return tokenizer.batch_decode(cont, skip_special_tokens=True)


def run(in_path: Path, out_path: Path, model_name: str, batch_size: int,
        max_new_tokens: int, temperature: float, num_examples: int | None,
        device: str, dtype: str, max_input_len: int) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = list(read_jsonl(str(in_path)))
    if num_examples:
        rows = rows[:num_examples]
    print(f"[harvest-batch] {len(rows)} rows, model={model_name}, "
          f"batch_size={batch_size}, device={device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device,
    )
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    t_start = time.time()

    with out_path.open("w", encoding="utf-8") as fout:
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start: batch_start + batch_size]
            prompts = [
                _build_chat_prompt(tokenizer,
                                   r.get("system_prompt", ""),
                                   r.get("user_query", ""))
                for r in batch_rows
            ]
            try:
                completions = _generate_batch(
                    model, tokenizer, prompts, max_new_tokens, temperature,
                    device, max_input_len,
                )
            except Exception as e:
                print(f"  batch [{batch_start}] err {type(e).__name__}: {str(e)[:160]}",
                      file=sys.stderr)
                continue

            for r, comp in zip(batch_rows, completions):
                fout.write(json.dumps({
                    "id": r.get("id"),
                    "source": (r.get("source") or "mcptox") + "_rollout_qwen3-8b",
                    "system_prompt": r.get("system_prompt", ""),
                    "user_query": r.get("user_query", ""),
                    "completion": comp,
                    "tags": r.get("tags", {}),
                    "extra": {
                        "poisoned_tool": (r.get("extra") or {}).get("poisoned_tool", ""),
                        "rollout_model": model_name,
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                    },
                }, ensure_ascii=False) + "\n")
            fout.flush()

            n_done += len(batch_rows)
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 0.1)
            eta = (len(rows) - n_done) / max(rate, 1e-6)
            print(f"  [{n_done}/{len(rows)}] elapsed={elapsed:.0f}s "
                  f"rate={rate:.2f}/s eta={eta/60:.1f}min", file=sys.stderr)

    print(f"[harvest-batch] done → {out_path}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=4000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--num-examples", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-input-len", type=int, default=4096)
    args = p.parse_args()

    run(args.in_path, args.out_path, args.model, args.batch_size,
        args.max_new_tokens, args.temperature, args.num_examples,
        args.device, args.dtype, args.max_input_len)


if __name__ == "__main__":
    main()
