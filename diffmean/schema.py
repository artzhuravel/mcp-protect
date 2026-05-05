"""Unified row schema emitted by every extractor.

One row = one (context, y_malicious, y_benign) triple, ready for activation
collection / DiffMean / RePS training. All three datasets (MCPTox,
open-prompt-injection, agent-dojo) emit the same shape so downstream code
treats them uniformly.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PairedRow:
    id: str
    source: str  # "mcptox" | "open_prompt_injection" | "agent_dojo"

    system_prompt: str
    user_query: str

    y_pos: str  # attack-compliant response (the bad one)
    y_neg: str  # attack-resistant response (the good one)

    tags: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_jsonl(rows, path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.to_jsonl() + "\n")
            n += 1
    return n


def append_jsonl(row: PairedRow, path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(row.to_jsonl() + "\n")


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
