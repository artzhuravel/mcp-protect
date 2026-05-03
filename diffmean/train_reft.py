"""Train a LoReFT intervention for tool-poisoning resistance.

ReFT (Representation Fine-Tuning, Wu et al. 2024) learns a low-rank edit to
the residual stream:  h → h + R^T(Rh - b)  at chosen layers/positions.
Unlike DiffMean (a fixed computed vector), LoReFT is *trained* on examples of
the desired behaviour — resistant completions — so it can capture a richer
subspace than the mean difference.

Training objective
------------------
Supervised next-token prediction on y_neg (attack-resistant) completions, with
the LoReFT intervention applied at the last `--positions` tokens of the prompt
prefix.  The model weights are frozen; only R and b are updated.

Typical run
-----------
    python -m diffmean.train_reft \\
        --in   diffmean/outputs/mcptox_pairs.clean.jsonl \\
        --out  diffmean/outputs/reft/qwen3-8b-L20-r4 \\
        --model Qwen/Qwen3-8B \\
        --layers 20 \\
        --rank 4 \\
        --positions l4 \\
        --epochs 3 \\
        --batch 4 \\
        --grad-accum 4

Then serve the trained intervention:
    python -m diffmean.serve \\
        --model Qwen/Qwen3-8B \\
        --reft-dir diffmean/outputs/reft/qwen3-8b-L20-r4
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

from diffmean.collect_activations import _build_chat_prompt, _strip_terminators  # noqa: E402
from diffmean.schema import read_jsonl  # noqa: E402


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_examples(rows: list[dict], tokenizer, max_len: int
                    ) -> list[dict]:
    """Convert PairedRows into (input_ids, labels) dicts for LoReFT training.

    Each row produces one training example:
      input  = chat_prefix  +  y_neg   (resist completion)
      labels = -100 for prefix tokens, token ids for y_neg tokens

    y_pos (comply) is not used during training — the loss signal comes only
    from correct prediction of the resistant completion.
    """
    examples = []
    for row in rows:
        ctx = _build_chat_prompt(tokenizer,
                                 row.get("system_prompt", ""),
                                 row.get("user_query", ""))
        y_neg = _strip_terminators(row.get("y_neg", ""))
        if not y_neg:
            continue

        ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
        neg_ids = tokenizer(y_neg, add_special_tokens=False)["input_ids"]

        full_ids = ctx_ids + neg_ids
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
            # Keep at least one completion token; skip if completion got truncated away.
            if len(full_ids) <= len(ctx_ids):
                continue
            neg_ids = full_ids[len(ctx_ids):]

        labels = [-100] * len(ctx_ids) + neg_ids
        examples.append({
            "input_ids": full_ids,
            "labels": labels,
            "ctx_len": len(ctx_ids),
        })
    return examples


def _parse_positions(spec: str, ctx_len_min: int) -> list[int]:
    """Convert a position spec to a list of *relative-to-start* token indices.

    Specs:
      "l1"  → last 1 token of the prefix  (index ctx_len-1)
      "l4"  → last 4 tokens of the prefix
      "f1"  → first token (index 0)
      "f1+l4" → first token and last 4 tokens of prefix

    Returns sorted unique non-negative indices relative to the full sequence.
    We return a closure fn(ctx_len) → list[int] because ctx_len varies per row.
    """
    # We return a *callable* that resolves given the actual ctx_len.
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    def resolve(ctx_len: int) -> list[int]:
        idxs: list[int] = []
        for part in parts:
            if part.startswith("l"):
                n = int(part[1:])
                idxs += list(range(max(0, ctx_len - n), ctx_len))
            elif part.startswith("f"):
                n = int(part[1:])
                idxs += list(range(min(n, ctx_len)))
        return sorted(set(idxs))
    return resolve


# ---------------------------------------------------------------------------
# LoReFT collator (minimal, no pyreft dependency for now)
# ---------------------------------------------------------------------------

class _ReFTCollator:
    """Pad a batch of variable-length examples to the same length."""
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, ctx_lens = [], [], []
        for ex in batch:
            pad = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.pad_id] * pad)
            labels.append(ex["labels"] + [-100] * pad)
            ctx_lens.append(ex["ctx_len"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "ctx_lens": ctx_lens,
        }


# ---------------------------------------------------------------------------
# LoReFT module
# ---------------------------------------------------------------------------

class LoReftIntervention(torch.nn.Module):
    """Low-Rank Linear Subspace ReFT (Wu et al. 2024).

    Applies:  h' = h + R^T (R h - b)
    where R is [rank × d] (orthonormal rows via Cayley param is optional here —
    we use plain gradient descent on R for simplicity) and b is [rank].

    This is identical to the LoReFT formulation but without the Cayley
    parameterisation, which is fine for our use-case (small rank, short
    training).
    """
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.R = torch.nn.Linear(d_model, rank, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(rank))
        # Initialise R as a random orthonormal projection.
        torch.nn.init.orthogonal_(self.R.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Cast R and b to h's dtype so the intervention works with bf16/fp16 models.
        w = self.R.weight.to(h.dtype)   # [rank, d_model]
        b = self.b.to(h.dtype)          # [rank]
        proj = h @ w.T - b              # [..., rank]
        return h + proj @ w             # [..., d_model]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _register_hooks(model, interventions: dict[int, LoReftIntervention],
                    positions_fn, ctx_lens: list[int]
                    ) -> list[torch.utils.hooks.RemovableHook]:
    """Register forward hooks on the specified layer outputs.

    Each hook fires after transformer block `L` and edits only the token
    positions returned by positions_fn(ctx_len) for each sequence in the batch.
    """
    hooks = []
    for layer_idx, intervention in interventions.items():
        layer = model.model.layers[layer_idx]

        def make_hook(interv, layer_i):
            def hook(module, args, output):
                # output is typically (hidden_states, ...) or just hidden_states.
                hs = output[0] if isinstance(output, tuple) else output
                for b_i, ctx_len in enumerate(ctx_lens[0]):  # closure over mutable list
                    for pos in positions_fn(ctx_len):
                        if pos < hs.shape[1]:
                            hs[b_i, pos, :] = interv(hs[b_i, pos, :])
                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs
            return hook

        hooks.append(layer.register_forward_hook(make_hook(intervention, layer_idx)))
    return hooks


def train(in_path: Path, out_dir: Path, model_name: str,
          layers: list[int], rank: int, positions_spec: str,
          epochs: int, batch_size: int, grad_accum: int,
          lr: float, max_len: int, limit: int | None,
          device: str, dtype: str) -> None:

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = list(read_jsonl(str(in_path)))
    if limit:
        rows = rows[:limit]
    print(f"[reft] {len(rows)} rows, model={model_name}, layers={layers}, rank={rank}",
          file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device,
    )
    model.eval()
    # Gradient checkpointing recomputes activations on the backward pass instead
    # of storing them — cuts ~40% of activation memory at ~20% speed cost.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    for L in layers:
        if not (0 <= L < n_layers):
            raise SystemExit(f"layer {L} out of range [0,{n_layers})")

    # Build interventions (one per layer), move to device.
    interventions: dict[int, LoReftIntervention] = {}
    for L in layers:
        iv = LoReftIntervention(d_model, rank).to(device)
        interventions[L] = iv

    positions_fn = _parse_positions(positions_spec, 0)

    examples = _build_examples(rows, tokenizer, max_len)
    print(f"[reft] built {len(examples)} training examples", file=sys.stderr)

    collator = _ReFTCollator(tokenizer.pad_token_id)
    loader = torch.utils.data.DataLoader(
        examples, batch_size=batch_size, shuffle=True, collate_fn=collator,
    )

    params = [p for iv in interventions.values() for p in iv.parameters()]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = (len(loader) // grad_accum) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Mutable container so hooks can read the current batch's ctx_lens.
    _ctx_lens_ref: list[list[int]] = [[]]

    step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _ctx_lens_ref[0] = batch["ctx_lens"]

            # Register hooks for this forward pass.
            hooks = _register_hooks(model, interventions, positions_fn, _ctx_lens_ref)
            try:
                out = model(input_ids=input_ids,
                            attention_mask=(input_ids != tokenizer.pad_token_id).long(),
                            labels=labels)
                loss = out.loss / grad_accum
            finally:
                for h in hooks:
                    h.remove()

            loss.backward()
            total_loss += loss.item() * grad_accum
            n_batches += 1

            if (batch_i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 20 == 0:
                    print(f"  epoch={epoch+1} step={step} "
                          f"loss={total_loss/n_batches:.4f}", file=sys.stderr)

        print(f"[reft] epoch {epoch+1}/{epochs} mean_loss={total_loss/n_batches:.4f}",
              file=sys.stderr)

    # Save intervention weights + config.
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model_name,
        "layers": layers,
        "rank": rank,
        "positions": positions_spec,
        "d_model": d_model,
    }
    with (out_dir / "reft_config.json").open("w") as f:
        json.dump(config, f, indent=2)
    for L, iv in interventions.items():
        torch.save(iv.state_dict(), out_dir / f"L{L:02d}_intervention.pt")
    print(f"[reft] saved to {out_dir}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layers", default="20",
                   help="Comma-separated layer indices, e.g. 16,20,24")
    p.add_argument("--rank", type=int, default=4,
                   help="LoReFT low-rank dimension. Start with 4; try 8 or 16 if loss plateaus.")
    p.add_argument("--positions", default="l4",
                   help="Which prefix tokens to intervene on: l4=last-4, f1+l4=first+last-4.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   ("mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    train(args.in_path, args.out_dir, args.model, layers,
          args.rank, args.positions, args.epochs, args.batch,
          args.grad_accum, args.lr, args.max_len, args.limit,
          args.device, args.dtype)


if __name__ == "__main__":
    main()
