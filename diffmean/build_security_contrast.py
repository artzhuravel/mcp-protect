"""Build the security-contrast training set.

After harvesting under the security prompt + judge labelling, we want a
length-matched, deliberation-engaged contrast set:
  y_pos (comply)   = take from existing default-prompt comply rollouts that
                     mention the poisoned tool (already long, deliberative).
  y_neg (resist)   = take from new security-prompted resist rollouts that
                     mention the poisoned tool (long, deliberative under the
                     security note).

Both classes have the original default MCPTox system_prompt saved (the
security note was used to elicit traces but stripped from the saved row),
so any DiffMean direction trained from this set captures the *decision*
axis with minimal prompt confound.

Usage:
    python -m diffmean.build_security_contrast \\
        --comply-source diffmean/outputs/qwen3_rollouts.labelled.jsonl \\
        --resist-source diffmean/outputs/qwen3_rollouts_secprompt.labelled.jsonl \\
        --out diffmean/outputs/qwen3_security_contrast.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PIPELINE_ROOT = _THIS.parent
if str(_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT.parent))

from diffmean.schema import read_jsonl  # noqa: E402

_TNRE = re.compile(r"^Tool:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_THRE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _mentions_poison(row: dict) -> bool:
    completion = (row.get("y_pos") or row.get("y_neg") or row.get("completion") or "")
    if "</think>" not in completion:
        return False
    poisoned = ((row.get("extra") or {}).get("poisoned_tool") or "")
    m = _TNRE.search(poisoned)
    tname = m.group(1) if m else None
    if not tname:
        return False
    tm = _THRE.search(completion)
    tb = tm.group(1) if tm else ""
    return tname.lower() in tb.lower()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--comply-source", type=Path, required=True,
                   help="Existing default-prompt labelled jsonl (has y_pos for comply).")
    p.add_argument("--resist-source", type=Path, required=True,
                   help="New security-prompted labelled jsonl (will use rows where label=resist).")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    n_comply = n_resist = n_total_in = 0
    seen_ids = set()
    with args.out.open("w", encoding="utf-8") as fout:
        # Comply class: from existing default-prompt rollouts
        for r in read_jsonl(str(args.comply_source)):
            n_total_in += 1
            if r.get("label") != "comply":
                continue
            if not _mentions_poison(r):
                continue
            cid = r.get("id")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            # Normalize: y_pos = the comply trace, y_neg = ""
            out = dict(r)
            out["source"] = "security_contrast/comply_default_prompt"
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_comply += 1

        # Resist class: from new security-prompted rollouts (filter by judge label)
        for r in read_jsonl(str(args.resist_source)):
            n_total_in += 1
            label = r.get("label")
            if label != "resist":
                continue
            if not _mentions_poison(r):
                continue
            cid = r.get("id")
            if cid in seen_ids:
                continue  # Don't double-count if the same case appears
            seen_ids.add(cid)
            # The labelled jsonl from judge_rollouts.py has y_pos/y_neg set
            # by label. For resist, y_neg = completion. We'll re-emit consistently.
            completion = r.get("y_neg") or r.get("completion") or ""
            out = {
                "id": cid,
                "source": "security_contrast/resist_secprompt_then_swapped",
                "system_prompt": r.get("system_prompt", ""),
                "user_query": r.get("user_query", ""),
                "y_pos": "",
                "y_neg": completion,
                "label": "resist",
                "tags": r.get("tags", {}),
                "extra": r.get("extra", {}),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_resist += 1

    print(f"[contrast] read {n_total_in} input rows", file=sys.stderr)
    print(f"[contrast] kept {n_comply} comply (default prompt) + {n_resist} resist "
          f"(security-prompted, prompt swapped) = {n_comply+n_resist} → {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
