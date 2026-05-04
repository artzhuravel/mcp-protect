"""Build the v2 audit-prompted contrast set.

Both classes (comply and resist) sourced from the v2 audit-prompted harvest:
  - Same prompt context (security note appended at gen time, swapped back in storage)
  - Both deliberation-mode (model audited the tool)
  - Filtered to mentions_poison=True (model engaged with poisoned tool by name)

Compared to the original baseline contrast (comply much longer than resist),
this set should be length-matched and engagement-matched, leaving "decision
conclusion" as the dominant axis for DiffMean.

Output schema mirrors collect_activations.py input expectations:
  y_pos = comply trace (when label=comply)
  y_neg = resist trace (when label=resist)
  label = "comply" or "resist"
  tags + extra preserved
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
    completion = (row.get("completion") or row.get("y_pos") or row.get("y_neg") or "")
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
    p.add_argument("--source", type=Path, required=True,
                   help="v2 audit-prompted labelled jsonl.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    n_in = n_comply = n_resist = 0
    seen = set()
    with args.out.open("w", encoding="utf-8") as fout:
        for r in read_jsonl(str(args.source)):
            n_in += 1
            label = r.get("label")
            if label not in ("comply", "resist"):
                continue
            if not _mentions_poison(r):
                continue
            cid = r.get("id")
            if cid in seen:
                continue
            seen.add(cid)

            completion = r.get("completion") or r.get("y_pos") or r.get("y_neg") or ""
            row = {
                "id": cid,
                "source": (r.get("source") or "v2_audit") + "_contrast",
                "system_prompt": r.get("system_prompt", ""),
                "user_query": r.get("user_query", ""),
                "y_pos": completion if label == "comply" else "",
                "y_neg": completion if label == "resist" else "",
                "label": label,
                "tags": r.get("tags", {}),
                "extra": r.get("extra", {}),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if label == "comply": n_comply += 1
            else: n_resist += 1
    print(f"[v2-contrast] read {n_in} rows from {args.source}", file=sys.stderr)
    print(f"[v2-contrast] kept {n_comply} comply + {n_resist} resist (mentions_poison) "
          f"= {n_comply+n_resist} → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
