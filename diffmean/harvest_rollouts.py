"""Convert vf-eval rollouts → labelled (y, label) jsonl ready for activation collection.

We run a thinking-capable target model (e.g. Qwen3-8B) through MCPTox via
`vf-eval mcp_tox`. Each rollout's `completion[-1].content` is the model's
*real* `<think>...</think>{tool_call}` output, and the env's gpt-5.4-nano
judge has already stamped `attack_resistance` ∈ {0, 1} on it.

This script:
  1. Walks the eval output dir and reads `results.jsonl`.
  2. For each rollout, attaches the source `system_prompt` and `user_query`
     reconstructed from the prompt messages, plus the judge label.
  3. Emits one row per rollout in the same schema as `mcptox_pairs.clean.jsonl`
     so `collect_activations.py` can ingest it unchanged. Two output modes:

     --mode pair  : group rollouts by mcptox case-id (info.id). For each case
                    that has BOTH a comply and a resist rollout, emit one row
                    where y_pos = the comply trace and y_neg = the resist
                    trace. Best for DiffMean since y_pos/y_neg share the same
                    poisoned context — no within-prompt confound.

     --mode flat  : emit every rollout individually with `label` ∈ {comply,
                    resist}; downstream code computes mean(H_comply) -
                    mean(H_resist) across the two sets. Useful when same-case
                    pairing is too sparse.

Usage:
    python -m diffmean.harvest_rollouts \\
        --eval diffmean/outputs/eval/qwen3-rollouts \\
        --out  diffmean/outputs/qwen3_thinking_pairs.jsonl \\
        --mode pair
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _has_closed_think(text: str) -> bool:
    """A rollout is usable only if `</think>` appears (otherwise the reasoning
    was truncated by max_tokens and we have no decision-token to read from)."""
    return "</think>" in (text or "").lower()


def _load_rollouts(eval_dir: Path) -> list[dict]:
    """Yield {id, label, system_prompt, user_query, completion} per rollout."""
    rows = []
    for results_path in eval_dir.glob("**/results.jsonl"):
        for line in results_path.open():
            d = json.loads(line)
            info = d.get("info")
            if isinstance(info, str):
                info = json.loads(info)
            comp_msgs = d.get("completion") or []
            completion = (comp_msgs[-1] or {}).get("content", "") if comp_msgs else ""
            if not completion:
                continue
            if not _has_closed_think(completion):
                continue
            ar = (d.get("metrics") or {}).get("attack_resistance")
            if ar is None:
                continue
            label = "resist" if float(ar) >= 0.5 else "comply"
            # Reconstruct system + user from prompt
            prompt_msgs = d.get("prompt") or []
            sys_p = next((m["content"] for m in prompt_msgs if m["role"] == "system"), "")
            usr_q = next((m["content"] for m in prompt_msgs if m["role"] == "user"), "")
            rows.append({
                "case_id": info.get("id") or info.get("data_id") or "",
                "label": label,
                "system_prompt": sys_p,
                "user_query": usr_q or info.get("query", ""),
                "completion": completion,
                "tags": {
                    "paradigm": info.get("paradigm", "Unknown"),
                    "security_risk": info.get("security_risk", info.get("security risk", "Unknown")),
                    "server": info.get("server", ""),
                },
                "extra": {
                    "poisoned_tool": info.get("poisoned_tool", ""),
                    "judge_grade": (d.get("state") or {}).get("judge_grade", ""),
                },
            })
    return rows


def write_pair_mode(rows: list[dict], out_path: Path) -> tuple[int, int, int]:
    """Match same-case rollouts that produced opposite labels."""
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["case_id"]:
            by_case[r["case_id"]].append(r)
    n_pair = n_drop_one = n_drop_same = 0
    with out_path.open("w") as f:
        for cid, group in by_case.items():
            comply = [r for r in group if r["label"] == "comply"]
            resist = [r for r in group if r["label"] == "resist"]
            if not comply or not resist:
                n_drop_one += 1
                continue
            # Same case, both polarities: pair the first of each.
            c, r = comply[0], resist[0]
            f.write(json.dumps({
                "id": f"qwen3-thinking-pair/{cid}",
                "source": "qwen3_thinking_rollouts",
                "system_prompt": c["system_prompt"],
                "user_query": c["user_query"],
                "y_pos": c["completion"],   # attack-compliant rollout
                "y_neg": r["completion"],   # attack-resistant rollout
                "tags": c["tags"],
                "extra": c["extra"],
            }, ensure_ascii=False) + "\n")
            n_pair += 1
        n_drop_same = sum(
            1 for g in by_case.values()
            if len(g) > 1 and len({r["label"] for r in g}) == 1
        )
    return n_pair, n_drop_one, n_drop_same


def write_flat_mode(rows: list[dict], out_path: Path) -> tuple[int, int]:
    """Emit each rollout individually. Downstream uses `label` to bucket."""
    n_comply = n_resist = 0
    with out_path.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "id": f"qwen3-thinking-flat/{r['case_id'] or i}",
                "source": "qwen3_thinking_rollouts_flat",
                "system_prompt": r["system_prompt"],
                "user_query": r["user_query"],
                # Hack: we put the completion in BOTH y_pos and y_neg so
                # collect_activations doesn't choke; the consumer should
                # branch on `label` instead of using the diff.
                "y_pos": r["completion"] if r["label"] == "comply" else "",
                "y_neg": r["completion"] if r["label"] == "resist" else "",
                "label": r["label"],
                "tags": r["tags"],
                "extra": r["extra"],
            }, ensure_ascii=False) + "\n")
            if r["label"] == "comply":
                n_comply += 1
            else:
                n_resist += 1
    return n_comply, n_resist


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["pair", "flat"], default="flat",
                   help="pair: only same-case (comply,resist) tuples (best for "
                        "DiffMean but requires both outcomes per case). "
                        "flat: every rollout, downstream buckets by label.")
    args = p.parse_args()

    rows = _load_rollouts(args.eval)
    n_total = len(rows)
    n_comply = sum(1 for r in rows if r["label"] == "comply")
    n_resist = n_total - n_comply
    print(f"[harvest] read {n_total} rollouts from {args.eval}: "
          f"{n_comply} comply / {n_resist} resist", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "pair":
        kept, drop_one, drop_same = write_pair_mode(rows, args.out)
        print(f"[harvest] mode=pair: {kept} pairs, "
              f"{drop_one} cases dropped (only one polarity), "
              f"{drop_same} cases dropped (both rollouts same label) → {args.out}",
              file=sys.stderr)
    else:
        cm, rs = write_flat_mode(rows, args.out)
        print(f"[harvest] mode=flat: {cm} comply + {rs} resist rollouts → {args.out}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
