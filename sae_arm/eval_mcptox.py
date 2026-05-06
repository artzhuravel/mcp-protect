"""Run MCPTox eval on a steering vector built by build_steering_vector.py.

Auto-resolves paths from (set, layer, paradigm, security_risk, threshold,
top_n) — the same identifying tuple build_steering_vector.py uses — so you
can think in terms of the experiment cell, not file paths.

After the eval finishes:
  - summary.jsonl lands in <vector_dir>/eval_thr<X>[_top<N>]/
  - the existing sae_thr<X>[_top<N>].meta.json next to the vector is updated
    with `defense_curve`, `eval_n`, `judge_model`, `pairs_file` fields, so
    each meta.json becomes the single record of (build + eval) for that cell.

If the build was stratified by paradigm (e.g. Template-2), the eval also
filters MCPTox cases to that same paradigm by default — matches the team's
per-template evaluation pattern. Override with --no-paradigm-filter / pass
explicit --paradigm-eval / --security-risk-eval.

Example:
    python sae_arm/eval_mcptox.py \\
        --set qwen3-v2-contrast --layer 20 --paradigm Template-2 \\
        --threshold 0.1 \\
        --alphas -10,-5,-2,-1,0,1,2,5,10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from batched_steered_eval import run
from sae_utils import out_dir_for

ROOT = Path(__file__).parent
# Pairs file lives under sae_arm/ on the pod (fetched from origin/main by
# fetch_team_data.sh, since this branch dropped diffmean/ to keep clones slim).
DEFAULT_PAIRS = ROOT / "mcptox_pairs.clean.jsonl"


def main() -> None:
    p = argparse.ArgumentParser()
    # Identifies which steering vector to evaluate (must match the build call)
    p.add_argument("--set", required=True,
                   choices=["qwen3-thinking-decision", "qwen3-v2-contrast"])
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--paradigm", default=None,
                   help="Build-time paradigm stratum. Also used as default "
                        "eval-time MCPTox filter unless --no-paradigm-filter.")
    p.add_argument("--security-risk", default=None,
                   help="Build-time security_risk stratum. Also used as default "
                        "eval-time filter unless --no-security-risk-filter.")
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--weighting", choices=["sign", "diff"], default="sign",
                   help="Must match the build's --weighting. 'sign' resolves to "
                        "sae_thr<X>.pt; 'diff' resolves to sae_thr<X>_diffw.pt.")
    # Eval knobs
    p.add_argument("--alphas", default="-15,-10,-5,0,5,10,15",
                   help="Default matches the team's published Phase-2 sweep "
                        "(L20 × Template-2 flagship: 7 cells, all-tok mode).")
    p.add_argument("--modes", default="all",
                   help="Comma-separated. last (last-token only), all "
                        "(every position), or both. Default: all (matches team).")
    p.add_argument("--num-examples", type=int, default=50,
                   help="Default 50 matches the team's published per-cell N. "
                        "Bump to 100+ for tighter CIs at 2× cost.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-input-len", type=int, default=4096)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--judge-model", default="openai/gpt-5.4-nano",
                   help="OpenRouter model id for the LLM judge. Default matches "
                        "the team's batched_steered_eval (gpt-5.4-nano) — note "
                        "this is DIFFERENT from prime-envs/environments/mcp_tox/"
                        "mcp_tox.py's default (gpt-5.4-mini); pass explicitly if "
                        "comparing across paths. On the pod, run with "
                        "--num-examples 5 first to verify the id resolves.")
    p.add_argument("--judge-concurrency", type=int, default=16)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                   help=f"MCPTox cases JSONL. Default: {DEFAULT_PAIRS}")
    # Decouple build-stratum from eval-stratum if needed
    p.add_argument("--no-paradigm-filter", action="store_true",
                   help="Even if the build was stratified by paradigm, "
                        "evaluate against ALL MCPTox cases, not just that paradigm.")
    p.add_argument("--no-security-risk-filter", action="store_true")
    p.add_argument("--paradigm-eval", default=None,
                   help="Override eval-time paradigm filter (default = --paradigm).")
    p.add_argument("--security-risk-eval", default=None)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for deterministic shuffle of MCPTox rows before "
                        "--num-examples slicing. Default 0 gives reproducible runs.")
    p.add_argument("--force-synthetic", action="store_true",
                   help="Override the safety check that refuses to evaluate "
                        "synthetic-SAE vectors (those tagged "
                        "WARNING_synthetic_sae in their meta.json). Only set "
                        "when intentionally pipeline-testing.")
    args = p.parse_args()

    base_dir = out_dir_for(args.set, args.layer, args.paradigm, args.security_risk)
    suffix = f"_top{args.top_n}" if args.top_n else ""
    weight_tag = "_diffw" if args.weighting == "diff" else ""
    vec_path = base_dir / f"sae_thr{args.threshold:g}{suffix}{weight_tag}.pt"
    meta_path = base_dir / f"sae_thr{args.threshold:g}{suffix}{weight_tag}.meta.json"
    eval_out = base_dir / f"eval_thr{args.threshold:g}{suffix}{weight_tag}"

    if not vec_path.exists():
        raise SystemExit(
            f"steering vector not found: {vec_path}\n"
            "Did you run build_steering_vector.py first with the same args?"
        )

    # Refuse to evaluate synthetic-SAE vectors by default — every cell call
    # ~$0.20 of OpenRouter judge fees, and folding fake numbers into a
    # synthetic meta.json silently pollutes the results tree.
    if meta_path.exists():
        meta_pre = json.loads(meta_path.read_text())
        if meta_pre.get("WARNING_synthetic_sae") and not args.force_synthetic:
            raise SystemExit(
                f"refusing to evaluate synthetic-SAE vector at {vec_path}\n"
                f"  meta says: {meta_pre['WARNING_synthetic_sae']!r}\n"
                "Pass --force-synthetic to override (e.g., for pipeline plumbing tests)."
            )

    # Resolve eval-time filters (default = build-time stratum, unless overridden)
    if args.paradigm_eval is not None:
        paradigm_eval = args.paradigm_eval
    elif args.no_paradigm_filter:
        paradigm_eval = None
    else:
        paradigm_eval = args.paradigm
    if args.security_risk_eval is not None:
        risk_eval = args.security_risk_eval
    elif args.no_security_risk_filter:
        risk_eval = None
    else:
        risk_eval = args.security_risk

    if not args.pairs.exists():
        raise SystemExit(
            f"MCPTox pairs file not found: {args.pairs}\n"
            "Try --pairs <path> or run the team's prep pipeline."
        )

    print(f"[eval] set={args.set}  layer={args.layer}  "
          f"paradigm={args.paradigm}  security_risk={args.security_risk}")
    print(f"[eval] vector:    {vec_path}  (||v|| → unit-normalized)")
    print(f"[eval] eval-out:  {eval_out}")
    print(f"[eval] eval filters: paradigm={paradigm_eval}  security_risk={risk_eval}")

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run(
        pairs_path=args.pairs,
        vec_path=vec_path,
        layer=args.layer,
        alphas=alphas,
        modes=modes,
        out_dir=eval_out,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_input_len=args.max_input_len,
        model_name=args.model,
        dtype=args.dtype,
        device=device,
        judge_model=args.judge_model,
        judge_concurrency=args.judge_concurrency,
        paradigm_filter=paradigm_eval,
        security_risk_filter=risk_eval,
        seed=args.seed,
    )

    # Fold the per-cell defense numbers + full eval provenance back into the
    # vector's meta.json — every knob that affects the defense_curve gets
    # recorded so the meta file is a complete reproducibility record.
    summary_path = eval_out / "summary.jsonl"
    if summary_path.exists() and meta_path.exists():
        defense_curve = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
        meta = json.loads(meta_path.read_text())
        meta["defense_curve"] = defense_curve
        meta["eval_provenance"] = {
            "alphas": alphas,
            "modes": modes,
            "num_examples": args.num_examples,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "max_input_len": args.max_input_len,
            "model": args.model,
            "dtype": args.dtype,
            "judge_model": args.judge_model,
            "judge_concurrency": args.judge_concurrency,
            "paradigm_filter": paradigm_eval,
            "security_risk_filter": risk_eval,
            "pairs_file": str(args.pairs),
            "seed": args.seed,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[eval] folded {len(defense_curve)}-cell defense_curve + provenance into {meta_path.name}")


if __name__ == "__main__":
    main()
