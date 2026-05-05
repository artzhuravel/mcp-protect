"""Extract paired (y_malicious, y_benign) examples from MCPTox.

MCPTox bundles a `label` dict per query with verdicts from ~11 evaluated models:
  - "Success"               → attack worked (agent followed poison) → y_pos
  - "Failure-Ignored"       → agent ignored poison → y_neg
  - "Failure-Direct Execution" → ambiguous, dropped
  - "None"                  → parse failure, dropped

For each query, we pick one Success response (preferring vulnerable models)
and one Failure-Ignored response (preferring resistant models). 736 of 1348
queries have at least one of each. No synthesis needed at this stage.

Usage:
    python -m diffmean.extractors.mcptox \\
        --src prime-envs/tmp/mcptox/response_all.json \\
        --out diffmean/outputs/mcptox_pairs.raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python -m diffmean.extractors.mcptox` from repo root,
# or directly as a script.
_THIS = Path(__file__).resolve()
_PIPELINE_ROOT = _THIS.parent.parent
if str(_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT.parent))

from diffmean.schema import PairedRow, write_jsonl  # noqa: E402


# Model preference order. Earlier = higher priority when picking a Success
# (more vulnerable → cleaner attack-compliant trace) or a Failure-Ignored
# (more resistant → cleaner refusal trace). Ranking from MCPTox paper Table 2.
POS_MODEL_PRIORITY = [
    "Phi-4-Think",
    "Qwen3-14b-Think",
    "Qwen3-8b-Think",
    "Qwen3-30b-A3b-No-Think",
    "Deepseek-qwen3-8b-Think",
    "mistral",
    "gemma-2-9b",
    "Qwen3-14b-No-Think",
    "Qwen3-8b-No-Think",
    "LLama-3-8b",
    "LLama-13b",
]

NEG_MODEL_PRIORITY = [
    "gemma-2-9b",
    "Qwen3-14b-No-Think",
    "Qwen3-8b-No-Think",
    "mistral",
    "LLama-3-8b",
    "Qwen3-14b-Think",
    "Qwen3-8b-Think",
    "Phi-4-Think",
    "Qwen3-30b-A3b-No-Think",
    "Deepseek-qwen3-8b-Think",
    "LLama-13b",
]


def _pick_response(label_dict: dict, response_dict: dict, want_label: str,
                   priority: list[str]) -> tuple[str, str] | None:
    """Return (model_name, response_text) for the highest-priority model whose
    verdict on this query matches `want_label`. None if no match."""
    for model in priority:
        if label_dict.get(model) == want_label and model in response_dict:
            text = response_dict[model]
            if isinstance(text, str) and text.strip():
                return model, text
    return None


def extract_pairs(mcptox_json_path: Path) -> list[PairedRow]:
    with mcptox_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows: list[PairedRow] = []
    skipped_no_pair = 0
    skipped_empty = 0

    for server_name, server_data in data["servers"].items():
        for inst_idx, instance in enumerate(server_data["malicious_instance"]):
            poisoned_tool = instance.get("poisoned_tool", "")
            meta = instance.get("metadata", {}) or {}
            paradigm = meta.get("paradigm", "Unknown")
            security_risk = meta.get("security risk", "Unknown")

            for q_idx, entry in enumerate(instance.get("datas", [])):
                query = entry.get("query", "")
                system = entry.get("system", "")
                label_dict = entry.get("label") or {}
                response_dict = entry.get("response") or {}

                if not query or not system:
                    skipped_empty += 1
                    continue

                pos = _pick_response(label_dict, response_dict, "Success", POS_MODEL_PRIORITY)
                neg = _pick_response(label_dict, response_dict, "Failure-Ignored", NEG_MODEL_PRIORITY)

                if pos is None or neg is None:
                    skipped_no_pair += 1
                    continue

                pos_model, y_pos = pos
                neg_model, y_neg = neg

                row = PairedRow(
                    id=f"mcptox/{server_name}/inst{inst_idx}/q{q_idx}",
                    source="mcptox",
                    system_prompt=system,
                    user_query=query,
                    y_pos=y_pos.strip(),
                    y_neg=y_neg.strip(),
                    tags={
                        "paradigm": paradigm,
                        "security_risk": security_risk,
                        "server": server_name,
                        "y_pos_source_model": pos_model,
                        "y_neg_source_model": neg_model,
                    },
                    extra={
                        "poisoned_tool": poisoned_tool,
                        "data_id": entry.get("id"),
                    },
                )
                rows.append(row)

    print(f"[extract_mcptox] kept {len(rows)} pairs, "
          f"skipped {skipped_no_pair} (no Success+Ignored pair available), "
          f"skipped {skipped_empty} (missing query/system).", file=sys.stderr)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=Path,
        default=Path("prime-envs/tmp/mcptox/response_all.json"),
        help="Path to MCPTox response_all.json.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("diffmean/outputs/mcptox_pairs.raw.jsonl"),
        help="Output JSONL path.",
    )
    p.add_argument("--limit", type=int, default=None, help="Smoke-test limit.")
    args = p.parse_args()

    rows = extract_pairs(args.src.resolve())
    if args.limit:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(rows, str(args.out))
    print(f"[extract_mcptox] wrote {n} rows → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
