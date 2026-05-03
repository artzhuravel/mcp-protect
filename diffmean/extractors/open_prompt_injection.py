"""Extract paired (y_pos, y_neg) examples from Open Prompt Injection.

OpenPromptInjection's attack is: inject an instruction into the *input data* of
a benign NLP task. Success = model outputs the injected task's answer instead
of the original task's answer.

Both responses are derivable from the dataset itself — no synthesis needed:
  y_neg = label-name of the clean input under the original task   (correct behavior)
  y_pos = label-name of the injected target under the original task (attack succeeded)

We use the LABEL_NAMES / FAKE_COMPLETIONS tables already defined in the prime-envs
environment to materialise the response strings.

Usage:
    python -m diffmean.extractors.open_prompt_injection \\
        --out diffmean/outputs/opi_pairs.raw.jsonl \\
        --num-per-task 200
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PIPELINE_ROOT = _THIS.parent.parent
if str(_PIPELINE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT.parent))

from diffmean.schema import PairedRow, write_jsonl  # noqa: E402


def _load_opi_module():
    """Dynamically load prime-envs/.../open_prompt_injection.py so we can reuse
    its TASK_INSTRUCTIONS / LABEL_NAMES / FAKE_COMPLETIONS / _apply_attack /
    _load_task_data without copying them."""
    repo_root = _PIPELINE_ROOT.parent
    src = repo_root / "prime-envs" / "environments" / "open_prompt_injection" / "open_prompt_injection.py"
    if not src.is_file():
        raise SystemExit(f"OPI source not found at {src}")
    spec = importlib.util.spec_from_file_location("opi_env", str(src))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Attack strategies handled by OPI's _apply_attack. We sample multiple per (target, inject)
# pairing to diversify the contrastive set.
ATTACK_STRATEGIES = ["naive", "escape", "ignore", "fake_comp", "combine"]


# We only build pairs over CLASSIFICATION tasks — they have closed-form labels we
# can use as y_pos/y_neg without further synthesis. Summarization/grammar are
# free-form and would need the LLM to generate target & inject responses.
CLASSIFICATION_TASKS = [
    "sentiment_analysis",
    "spam_detection",
    "hate_detection",
    "duplicate_sentence_detection",
    "natural_language_inference",
]


def _label_text(opi, task: str, label: int) -> str | None:
    table = opi.LABEL_NAMES.get(task)
    if not table:
        return None
    return table.get(label)


def _build_pairs(opi, num_per_task: int, split: str) -> list[PairedRow]:
    rows: list[PairedRow] = []
    for target_task in CLASSIFICATION_TASKS:
        target_instr, inject_instr_template = opi.TASK_INSTRUCTIONS[target_task]
        try:
            clean_pairs = opi._load_task_data(target_task, split, num_per_task)
        except Exception as e:
            print(f"[opi] skip {target_task} ({type(e).__name__}: {str(e)[:120]})", file=sys.stderr)
            continue

        for inject_task in CLASSIFICATION_TASKS:
            if inject_task == target_task:
                continue  # injection should differ from target
            try:
                inject_pairs = opi._load_task_data(inject_task, split, num_per_task)
            except Exception:
                continue

            inject_instr = opi.TASK_INSTRUCTIONS[inject_task][1]

            for i, (clean_data, true_label) in enumerate(clean_pairs):
                if i >= len(inject_pairs):
                    break
                inject_data, inject_target_label = inject_pairs[i]

                y_neg = _label_text(opi, target_task, true_label)
                # y_pos: model is supposed to answer the INJECTED task on the INJECTED data,
                # but the original task's instruction is still active — so the answer string
                # is the injected task's label name.
                y_pos = _label_text(opi, inject_task, inject_target_label)
                if y_neg is None or y_pos is None or y_pos == y_neg:
                    continue

                for strategy in ATTACK_STRATEGIES:
                    poisoned_input = opi._apply_attack(
                        clean_data=str(clean_data),
                        inject_instruction=inject_instr,
                        inject_data=str(inject_data),
                        strategy=strategy,
                        target_task=target_task,
                    )
                    rows.append(PairedRow(
                        id=f"opi/{target_task}/{inject_task}/{strategy}/{i}",
                        source="open_prompt_injection",
                        system_prompt=target_instr,
                        user_query=poisoned_input,
                        y_pos=y_pos,
                        y_neg=y_neg,
                        tags={
                            "target_task": target_task,
                            "inject_task": inject_task,
                            "attack_strategy": strategy,
                        },
                        extra={
                            "true_label_idx": true_label,
                            "inject_target_label_idx": inject_target_label,
                            "inject_instruction": inject_instr,
                            "clean_data": str(clean_data)[:1000],
                            "inject_data": str(inject_data)[:1000],
                        },
                    ))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path("diffmean/outputs/opi_pairs.raw.jsonl"))
    p.add_argument("--num-per-task", type=int, default=200,
                   help="Cap on examples loaded per task before pairing.")
    p.add_argument("--split", default="validation",
                   help="HuggingFace dataset split (passed through to _load_task_data).")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    opi = _load_opi_module()
    rows = _build_pairs(opi, args.num_per_task, args.split)
    if args.limit:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(rows, str(args.out))
    print(f"[extract_opi] wrote {n} rows → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
