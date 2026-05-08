# DiffMean re-evaluation under matched conditions

Re-evaluation of the team's per-paradigm DiffMean steering vectors using:

1. **Deterministic shuffle (`seed=0`)** of the paradigm-filtered MCPTox pool, *not* the team's original "first 50 in file order" selection. The MCPTox pairs file is server-clustered: a `[:50]` slice after paradigm filter contains rows from only ~5 unique servers (FileSystem, GitHub, Puppeteer, Slack, AdFin), missing the largest server categories (Email, Commander, Prisma) entirely. The seed=0 shuffle samples 19+ unique servers, giving an unbiased estimator of the population defense rate.
2. **REG0 judge prompt** (the team's "grade the FINAL tool call, not the reasoning trace" fix from `EXPERIMENT_LOG.md` §2.3), baked directly into `sae_arm/batched_steered_eval.py`'s `_JUDGE_PROMPT`. Prevents the OLD-judge over-counting of attack-success when the model engages with the injection in `<think>` but emits a safe tool call.
3. **Same model + judge** as the SAE arm (Qwen3-8B in thinking-mode + `openai/gpt-5.4-nano`), so cross-arm comparisons against `sae_arm/directions/...` (post-regrade) are apples-to-apples on every dimension except prompt-shuffle source.

## Cells re-run

| name | set | layer | paradigm | source vector | team's old-slice/old-judge headline |
|------|-----|-------|----------|---------------|--------------------------------------|
| `v1_L20_T2` | qwen3-thinking-decision | 20 | Template-2 | `diffmean/outputs/acts/qwen3-thinking-decision/L20/by_paradigm/Template-2.pt` | 0.880 (PHASE2 flagship) |
| `v2_L20_T2` | qwen3-v2-contrast | 20 | Template-2 | `diffmean/outputs/acts/qwen3-v2-contrast/L20/by_paradigm/Template-2.pt` | 0.900 (length-balanced peak) |
| `v2_L16_T2` | qwen3-v2-contrast | 16 | Template-2 | `diffmean/outputs/acts/qwen3-v2-contrast/L16/by_paradigm/Template-2.pt` | 0.880 |
| `v1_L16_T2` | qwen3-thinking-decision | 16 | Template-2 | `diffmean/outputs/acts/qwen3-thinking-decision/L16/by_paradigm/Template-2.pt` | 0.780 (apples-to-apples with our SAE L16 v1 T2) |
| `v1_L20_T3` | qwen3-thinking-decision | 20 | Template-3 | `diffmean/outputs/acts/qwen3-thinking-decision/L20/by_paradigm/Template-3.pt` | 0.480 (paradigm-3 baseline) |

## How to run on the pod

```bash
cd ~/mcp-protect
export OPENROUTER_API_KEY=sk-or-...

nohup bash diffmean_rerun_reg0_seed0/run_diffmean_reg0_seed0.sh \
    > diffmean_rerun_reg0_seed0/logs/sweep.log 2>&1 & disown

tail -f diffmean_rerun_reg0_seed0/logs/sweep.log
```

Each cell is ~55–60 min at `batch_size=24` on an A100 80GB. Total: ~4.5–5 hours for all 5 cells.

## Output layout

```
diffmean_rerun_reg0_seed0/
├── README.md                  (this file)
├── run_diffmean_reg0_seed0.sh (sweep script)
├── vectors_used/              (snapshot of each input vector, for provenance)
├── evals/
│   ├── v1_L20_T2/
│   │   ├── alpha_n15_all/results.jsonl     (50 rows, REG0-graded)
│   │   ├── alpha_n10_all/results.jsonl
│   │   ├── alpha_n5_all/results.jsonl
│   │   ├── alpha_0_all/results.jsonl
│   │   ├── alpha_5_all/results.jsonl
│   │   ├── alpha_10_all/results.jsonl
│   │   ├── alpha_15_all/results.jsonl
│   │   └── summary.jsonl                   (per-α defense numbers)
│   ├── v2_L20_T2/...
│   └── ...
└── logs/
    ├── sweep.log
    ├── v1_L20_T2.log
    ├── v2_L20_T2.log
    └── ...
```

The `summary.jsonl` in each cell directory contains the new defense_curve directly. Each `results.jsonl` row already has the REG0 grade in its `judge` field — no separate regrade pass needed.

## Pipeline verification

The eval pipeline used is `sae_arm/batched_steered_eval.py` (patched 2026-05-08 to use the REG0 judge prompt). Verified:

- Steering hook: identical to `diffmean/batched_steered_eval.py` (same `h ← h + α·v` at all positions, unit-normalized v).
- Vector loading: accepts any `(d_model,)` tensor via `--vec`. The team's per-paradigm DiffMean vectors at `diffmean/outputs/acts/*/L*/by_paradigm/*.pt` load directly.
- Shuffle: `random.Random(seed).shuffle(rows)` with `seed=0` before `[:num_examples]`.
- Judge: REG0 prompt (from `diffmean/regrade_mcp_tox_diffmean.py:21-62`), `openai/gpt-5.4-nano`, `concurrency=32`.

## What to compare against

- **Our SAE arm** (post-regrade): `sae_arm/directions/*/L*/by_paradigm/Template-*/eval_thr*/results_regraded.jsonl`. These were graded with the REG0 prompt on `gpt-5.4-nano` over the seed=0 slice. The `v1_L16_T2`, `v1_L20_T2`, and `v1_L20_T3` cells are direct apples-to-apples comparisons.
- **Team's old DiffMean**: `diffmean/outputs/eval/per_template{,_v2}/L*_Template-*/results.jsonl` — graded with OLD prompt on `gpt-5.4-nano` over a non-shuffled (file-order) slice. Cross-method gap with this re-eval will quantify how much of the team's reported absolute numbers were artifacts of the prompt-slice and judge-prompt issues.
