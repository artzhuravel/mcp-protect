"""
Phase 2: build DiffMean directions for the tool-skepticism concept.

For each MCPTox malicious instance where label[--label-key] matches
--label-pattern (default substring "Failure-"):

    +side = (datas.system,             datas.query)   # poisoned prompt the model fell for
    -side = (servers.clean_system_promot, datas.query)  # same query, no injected tool

Run a single forward pass on the model for each side, capture the residual
stream at the probe token (the last input token, which is the '\\n' after
'<|im_start|>assistant') for every layer in --layers. The DiffMean direction
at layer L is `mean(act_+) - mean(act_-)`, saved to
`<out-dir>/diffmean_l{L}.pt` along with a sidecar JSON describing the split.

Splits:
  - --eval-paradigm (default Template-3) is fully held out from direction
    estimation, used later as the cross-paradigm generalization slice.
  - From the remaining pool, the first --max-val-pairs go to validation,
    the next --max-train-pairs go to direction estimation.

Usage (full run):
    python sae_arm/build_directions.py --model-name Qwen/Qwen3-8B \\
        --layers 8 16 24 32 --max-train-pairs 600 --out-dir sae_arm/directions

Smoke test (small model, tiny slice):
    python sae_arm/build_directions.py --model-name Qwen/Qwen3-0.6B \\
        --layers 4 12 20 --max-train-pairs 10 --max-val-pairs 5 \\
        --out-dir sae_arm/directions/_smoke
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MCPTOX_REPO = "https://github.com/zhiqiangwang4/MCPTox-Benchmark.git"


def _resolve_data_path(arg: Path | None) -> Path:
    if arg is not None:
        if not arg.is_file():
            raise SystemExit(f"--data-path {arg} not found")
        return arg
    # Reuse the copy from earlier exploration if present, else the prime-envs
    # cache, else shallow-clone.
    for cached in (
        Path("/tmp/mcptox_inspect/response_all.json"),
        Path("prime-envs/tmp/mcptox/response_all.json"),
    ):
        if cached.is_file():
            print(f"[data] using cached {cached}")
            return cached
    tmp = Path(tempfile.mkdtemp(prefix="mcptox_"))
    print(f"[data] shallow-cloning {MCPTOX_REPO} into {tmp}")
    subprocess.run(
        ["git", "clone", "--depth", "1", MCPTOX_REPO, str(tmp)],
        check=True, capture_output=True,
    )
    return tmp / "response_all.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--device", default=None)
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24, 32])
    p.add_argument("--label-key", default="Qwen3-8b-Think")
    p.add_argument(
        "--label-pattern", default="Failure-",
        help="Substring match against label value. 'Failure-' includes "
             "Failure-Direct Execution, Failure-Ignored, etc.",
    )
    p.add_argument("--eval-paradigm", default="Template-3")
    p.add_argument("--max-train-pairs", type=int, default=600)
    p.add_argument("--max-val-pairs", type=int, default=100)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("sae_arm/directions"))
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = _resolve_data_path(args.data_path)
    raw = json.loads(data_path.read_text())

    # Build candidate pair list, filtered by Qwen3-8b-Think outcome.
    pairs: list[dict] = []
    for server_name, srv in raw["servers"].items():
        # Field name is intentionally typo'd in the dataset.
        clean_sys = srv.get("clean_system_promot") or srv.get("clean_system_prompt")
        if not clean_sys:
            continue
        for inst in srv.get("malicious_instance", []):
            paradigm = inst.get("metadata", {}).get("paradigm", "Unknown")
            for entry in inst.get("datas", []):
                label = entry.get("label", {}).get(args.label_key, "") or ""
                if args.label_pattern not in label:
                    continue
                pairs.append({
                    "server": server_name,
                    "paradigm": paradigm,
                    "id": entry.get("id"),
                    "query": entry["query"],
                    "sys_pos": entry["system"],
                    "sys_neg": clean_sys,
                    "label": label,
                })

    # Split: held-out paradigm reserved for generalization; the rest split into
    # validation (first --max-val-pairs) then direction-estimation (next
    # --max-train-pairs).
    held = [p for p in pairs if p["paradigm"] == args.eval_paradigm]
    pool = [p for p in pairs if p["paradigm"] != args.eval_paradigm]
    random.shuffle(pool)
    val = pool[: args.max_val_pairs]
    train = pool[args.max_val_pairs : args.max_val_pairs + args.max_train_pairs]

    print(f"[data] total {args.label_pattern}* pairs: {len(pairs)}")
    print(f"[data] held-out '{args.eval_paradigm}': {len(held)}")
    print(f"[data] val: {len(val)}  train: {len(train)}")
    if not train:
        raise SystemExit("no training pairs after filter+split")

    # Pick device + dtype.
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    print(f"[model] loading {args.model_name} on {args.device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, device_map=args.device,
    ).eval()
    d_model = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    for layer in args.layers:
        if not (0 <= layer < num_layers):
            raise SystemExit(f"layer {layer} out of range [0,{num_layers})")

    # Register hooks: each writes the last-token residual to a scratch dict at
    # forward time. We accumulate float32 sums on CPU to avoid bf16 reduction
    # drift on long runs.
    captured: dict[int, torch.Tensor | None] = {layer: None for layer in args.layers}
    handles = []
    for layer in args.layers:
        def make_hook(layer_idx: int):
            def hook(_mod, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h[:, -1, :].detach().to(torch.float32).cpu()
            return hook
        handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))

    sums_pos = {layer: torch.zeros(d_model, dtype=torch.float32) for layer in args.layers}
    sums_neg = {layer: torch.zeros(d_model, dtype=torch.float32) for layer in args.layers}
    n_pos = 0
    n_neg = 0
    n_skipped = 0

    def forward_capture(system: str, user: str) -> bool:
        out = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, enable_thinking=True,
            return_tensors="pt",
        )
        ids = (out["input_ids"] if hasattr(out, "input_ids") else out)
        if ids.shape[1] > args.max_input_tokens:
            return False
        ids = ids.to(args.device)
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
        return True

    t0 = time.time()
    for i, pair in enumerate(train):
        ok_pos = forward_capture(pair["sys_pos"], pair["query"])
        if ok_pos:
            for layer in args.layers:
                sums_pos[layer] += captured[layer][0]
            n_pos += 1
        ok_neg = forward_capture(pair["sys_neg"], pair["query"])
        if ok_neg:
            for layer in args.layers:
                sums_neg[layer] += captured[layer][0]
            n_neg += 1
        if not (ok_pos and ok_neg):
            n_skipped += 1
        if (i + 1) % 20 == 0 or (i + 1) == len(train):
            print(f"  [{i+1}/{len(train)}] {time.time()-t0:.1f}s elapsed, skipped={n_skipped}")

    for h in handles:
        h.remove()

    if n_pos == 0 or n_neg == 0:
        raise SystemExit("no usable pairs collected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "model_name": args.model_name,
        "label_key": args.label_key,
        "label_pattern": args.label_pattern,
        "eval_paradigm_held_out": args.eval_paradigm,
        "n_train_attempted": len(train),
        "n_pos_used": n_pos,
        "n_neg_used": n_neg,
        "n_skipped_oversize": n_skipped,
        "max_input_tokens": args.max_input_tokens,
        "seed": args.seed,
        "layers": args.layers,
        "d_model": d_model,
        "directions": {},
        "wall_seconds": round(time.time() - t0, 1),
    }
    for layer in args.layers:
        mean_pos = sums_pos[layer] / n_pos
        mean_neg = sums_neg[layer] / n_neg
        v = (mean_pos - mean_neg).contiguous()
        out_path = args.out_dir / f"diffmean_l{layer}.pt"
        torch.save(v, out_path)
        norm = float(v.norm())
        summary["directions"][str(layer)] = {
            "path": str(out_path),
            "norm": norm,
            "mean_pos_norm": float(mean_pos.norm()),
            "mean_neg_norm": float(mean_neg.norm()),
            "cosine_to_mean_pos": float((v / v.norm().clamp(min=1e-8)) @ (mean_pos / mean_pos.norm().clamp(min=1e-8))),
        }
        print(
            f"[layer {layer:>2}] {out_path}  "
            f"||v||={norm:.3f}  ||mean+||={float(mean_pos.norm()):.3f}  "
            f"||mean-||={float(mean_neg.norm()):.3f}"
        )

    (args.out_dir / "diffmean_summary.json").write_text(json.dumps(summary, indent=2))
    keep_keys = {"server", "paradigm", "id", "label"}
    (args.out_dir / "diffmean_train_pairs.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in p.items() if k in keep_keys}) for p in train)
    )
    (args.out_dir / "diffmean_val_pairs.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in p.items() if k in keep_keys}) for p in val)
    )
    print(f"[done] wall-clock {time.time()-t0:.1f}s; summary at {args.out_dir / 'diffmean_summary.json'}")


if __name__ == "__main__":
    main()
