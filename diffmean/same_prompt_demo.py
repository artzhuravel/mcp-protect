"""Same-prompt steering demo.

Pick a handful of MCPTox cases, sweep alpha for each, save the resulting
deliberation traces side-by-side. Shows steering effects without the
aggregate-metric noise of the multi-cell vf-eval sweeps.

Uses per-request steering override via serve.py's `extra_body={"steering": {"alpha": ...}}`,
so no server restart between alphas — one warm model serves the whole demo.

Usage:
    python -m diffmean.same_prompt_demo \\
        --pairs    diffmean/outputs/mcptox_pairs.clean.jsonl \\
        --out      diffmean/outputs/eval/same_prompt_demo.jsonl \\
        --base-url http://localhost:8000/v1 \\
        --alphas   -10,-5,-3,-2,-1,0,1,2,3,5,10 \\
        --n-cases  5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
_PIPELINE_ROOT = _THIS.parent
if str(_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT.parent))

from diffmean.schema import read_jsonl  # noqa: E402


def _pick_cases(rows: list[dict], n: int) -> list[dict]:
    """One case per attack paradigm, biased toward distinct security risks
    so the demo covers the dataset's diversity."""
    seen_para = set()
    seen_risk = set()
    picked = []
    for r in rows:
        tags = r.get("tags", {})
        p = tags.get("paradigm", "")
        risk = tags.get("security_risk", tags.get("security risk", ""))
        if p in seen_para and risk in seen_risk:
            continue
        picked.append(r)
        seen_para.add(p)
        seen_risk.add(risk)
        if len(picked) >= n:
            break
    return picked


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out",   type=Path, required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--alphas", default="-10,-5,-3,-2,-1,0,1,2,3,5,10")
    p.add_argument("--n-cases", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=2500)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    from openai import OpenAI

    rows = list(read_jsonl(str(args.pairs)))
    cases = _pick_cases(rows, args.n_cases)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    print(f"[demo] {len(cases)} cases × {len(alphas)} alphas = {len(cases)*len(alphas)} requests",
          file=sys.stderr)

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as fout:
        for c_i, case in enumerate(cases):
            for a_i, alpha in enumerate(alphas):
                t0 = time.time()
                try:
                    resp = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": case.get("system_prompt", "")},
                            {"role": "user",   "content": case.get("user_query", "")},
                        ],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        extra_body={"steering": {"alpha": alpha}},
                    )
                    completion = resp.choices[0].message.content or ""
                    err = None
                except Exception as e:
                    completion = ""
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                elapsed = time.time() - t0
                row = {
                    "case_id": case.get("id"),
                    "case_idx": c_i,
                    "alpha": alpha,
                    "elapsed_s": round(elapsed, 1),
                    "user_query": case.get("user_query", "")[:200],
                    "poisoned_tool": (case.get("extra") or {}).get("poisoned_tool", "")[:400],
                    "tags": case.get("tags", {}),
                    "completion": completion,
                    "error": err,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                msg = err or f"{len(completion)} chars"
                print(f"  [{c_i+1}/{len(cases)}] α={alpha:+.1f} t={elapsed:.1f}s {msg}",
                      file=sys.stderr)
    print(f"[demo] → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
