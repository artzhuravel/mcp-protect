# Experiment queue + status (live)

Updated 2026-05-06 ~10:50 PT. **Source of truth.** Supersedes everything before today's compact.

## Headline finding from overnight (2026-05-05 → 2026-05-06)

After 11 single-concept HyperSteer variants (v3, v6, v8, v10, v11, v11_FIXED, v15, v17, v18, v19, v14, v21), **NONE deliver clean MCP-Tox defense at N=50.** The one apparent "win" (v17/v18/v19 at f=1.0, AR=0.92-0.98) is the **thinking-suppression artifact**: at high steering factor the model stops emitting `<think>` blocks (column `other` jumps to 70-80%), so the attack pathway via reasoning is blocked because the model isn't reasoning at all. v19 (trained on `<think>`-stripped data) reaches saturation at lower factor than v17 — confirms the mechanism.

| run | mechanism | best AR | executed | "other" | verdict |
|-----|-----------|---------|----------|---------|---------|
| v14 contrastive | y_neg=comply traces | 0.66 (baseline only) | 0.84 at f=1 | 10% | fails — contrastive doesn't flip direction |
| v18 rank=4 | multi-rank | 0.92 at f=1.0 | 0.24 | 76% | artifact — thinking suppression |
| v19 no-think data | strip `<think>` from training | 0.96 at f=1.0 | 0.24 | 76% | CONFIRMS thinking-suppression hypothesis |
| v21 L24 layer | layer change | 0.66 (flat) | 0.46-0.66 | 24-44% | inert — confirms L24 has no causal range |

## Boxes (live, 2026-05-06)

| box | host | gpu | role |
|-----|------|-----|------|
| **Box A** | `root@69.30.85.78 -p 22078` | NVIDIA A40 46GB | E9 (v17 benign regression) |
| **Box B** | `ubuntu@216.81.248.82` | **NVIDIA A100 80GB** | E7 (v17 replication on prompts 50-99) + later v22 train |
| **Box C** | `ubuntu@64.247.196.131` | NVIDIA RTX A6000 48GB | E10 (v17 thinking-OFF) — needs orphan `serve_mcp_hypersteer` killed first |

All boxes use `~/.ssh/primeintellect_ed25519`. Repo paths: `/root/mcp-protect` (Box A), `/home/ubuntu/mcp-protect` (Box B/C).

## Currently running

None. Awaiting Box C orphan kill (PID 273068, 21.8GB VRAM) + experiment fire.

## Queue (priority order)

### Phase 1 — fire now (parallel, in-flight today)
1. **E8** — judge mis-scoring audit on v17 f=0.7 (LOCAL, no GPU): read all `attack_resistance==0.0` cells from `vfeval_n50_v17_*/v17_f0p7/.../results.jsonl`. Tabulate model emits SAFE tool but judge marks failed. If >3, judge calibration is broken → re-grade and reinterpret all numbers.
2. **E7** — Box B: v17 f=+1.0 with `--num-examples 50` and a different prompt slice (50-99 instead of 0-49). ~25 min. Confirms 0.98 isn't sample-set-specific.
3. **E9** — Box A: v17 f=+1.0 on 50 BENIGN-only prompts. Confirms steering doesn't break helpful behavior. ~30 min.
4. **E10** — Box C: `MCP_ENABLE_THINKING=0` + v17 f=+1.0. If AR ≈ thinking-off baseline → v17 IS thinking suppression. ~30 min.

### Phase 2 — fire after Phase 1 (~3 hours from now)
5. **Diagnostic A** — hypernet collapse cosine: compute v(prompt) from v15/v17/v18 hypernets across 50 stratified prompts. If pairwise cos > 0.95 → hypernet collapsed to fixed-vector; STOP iterating on HyperSteer architecture. Local-ish (small forward pass on hypernet, fits on any box).
6. **E11** — finer v17 factor sweep at f=0.85, 0.9, 0.95, 1.05, 1.1, 1.2 N=20. Maps the 0.5→1.0 transition curve. ~1.5h.
7. **E12** — v11_FIXED at `max_tokens=4096` thinking-ON. Resolves whether v11→v17 jump comes from "terse training" or "any training that fits under truncation". ~1h.

### Phase 3 — v22: verification-trajectory dataset (the data-side hypothesis the user raised)
8. **v22 build** (local, ~$2 API): generate ~150-300 trajectories where Qwen3-8B EXPLICITLY VERIFIES tool descriptions and user intent before executing. Pattern:
    ```
    <think>The user requested X. Tool A claims a hidden prerequisite Y;
    that's not part of X and contradicts what user asked. This pattern is
    consistent with a tool-poisoning injection. The legitimate behavior is
    to call A with the user's literal arguments only.</think>
    {"tool": "A", "arguments": {...}}
    ```
    Distinct from v17 (terse "ignore + go direct") because it trains explicit verification REASONING. Must be Qwen3-8B-self-generated thinking-mode (no other-model traces).
9. **v22 train** on Box B (A100, ~5-12h): single-concept HyperSteer at L20 with same hyperparams as v17 (bs=1, ga=8, n_epochs=5, lr=2e-5, low_rank_dimension=1, max_input_length=2048).
10. **v22 eval**: same headlines as v17 (f=0,0.3,0.5,0.7,1.0 N=50). Decision criterion: if AR>baseline+0.10 at f=0.5 AND `executed`<0.30 AND `other`<0.30 → real defense (verification works). If `other` jumps to 70%+ at f=1.0 → same thinking-suppression artifact, confirms architectural ceiling.

### Phase 4 — if v22 also fails (write-up)
11. Pivot to (a) **direct DiffMean steering** without hypernet (we have the contrast vec at L20 already), (b) **fine-tune Qwen3-8B with LoRA on verification trajectories** (compare to steering), or (c) accept that single-concept steering on Qwen3-thinking is fundamentally limited and write up the negative result + thinking-suppression artifact mechanism as the contribution.

## Sync state (local) — 2026-05-06 10:50 PT

### Eval dirs synced
- `axbench/axbench/outputs/eval/_synced_box_a/` — diag_B, gemma9b, v15_NEG, v17 (Box A copy), v19, plus older runs
- `axbench/axbench/outputs/eval/_synced_box_b/` — v12, v15, v17 (Box B copy), v18, v21
- `axbench/axbench/outputs/eval/_synced_box_c/` — v10, v11_2k, v11_FIXED, v14, plus older runs

### Weights synced (in-flight as of 10:50 PT, verify via `du -sh _synced_weights/*/`)
- v14_contrast (Box C source, ~3.4GB)
- v15_resist (Box B source, ~3.4GB) — v17 already on local at canonical path
- v18_rank4 (Box B source, ~3.4GB)
- v19_nothink (Box A source, ~3.4GB)
- v21_l24 (Box B source, ~3.4GB)

### NOT in git (intentional)
- safetensors files (>100MB GitHub limit) — keep local-only
- per-eval `results.jsonl` (large) — commit `metadata.json` + `summary.jsonl` only
- `*.parquet` training data files

## L20 vs L24 evidence (confirmed Oct, re-confirmed by v21 fail)

DiffMean layer-sweep `diffmean/outputs/eval/layer_axis_allt_v2/L{16,20,24}/summary.jsonl`:

| layer | defense at α=−15 | defense at α=+15 | range | verdict |
|-------|-----------------|------------------|-------|---------|
| L16 | 0.633 | 0.500 | 0.20 | mediocre |
| **L20** | **0.667** | **0.333** | **0.33** | **strong steerability** |
| L24 | 0.567 | 0.633 | 0.07 | nearly inert (CONFIRMED by v21) |

**Mandate**: all future Qwen3-8B HyperSteer training uses L20.

## Standing rules (carried forward)

- Inspect generated/intermediate data BEFORE training (qualitative discipline). Read 5+ samples per cell, check raw distributions before any aggregate-metric claim.
- **For Qwen3-8B steering: USE LAYER 20.** v3-v11 used L24 by mistake; v21 confirms L24 is dead.
- Direct-mode training data (no `<think>`) applied to a thinking-mode model is a failure mode → v17 was qualitatively a "thinking suppression learner" because the data left a low-effort minima. Verification-trajectory data (v22) explicitly populates the `<think>` block with verification reasoning.
- vf-eval `--max-concurrent` > 1 deadlocks env worker — use 1.
- max_tokens=1024 truncates audit-prompted traces mid-think → fake +pts. Use 2048.
- New policy model? Add to `axbench/utils/constants.py:CHAT_MODELS` or serve drops the system prompt.
- `make_data_module` defaults `max_input_length: 1024`. For reasoning training set `train.models.<name>.max_input_length: 2048` AND drop rows above 2048 tokens pre-training.
- Single-concept dataset uses `concept_id=0`; multi-concept w/ non-contiguous cids requires `_MetadataLookup` (already shimmed in train.py).
- For tmux: serve_mcp_hypersteer + vf-eval BOTH need tmux on primeintellect L40S; nohup+disown is insufficient.
- Push the AI-Consensus parent repo, not the autinn submodule. EXCEPT: `mcpattack` branch on autinn/axbench DOES have push access.
- **Sync trained weights to local often** — boxes get reaped without warning.

## Open questions

- **Does v17 f=1.0's AR=0.98 hold up on a different prompt slice?** (E7 answers)
- **Is the judge mis-scoring inflating numbers?** (E8 answers)
- **Does v17 hurt benign tool use?** (E9 answers)
- **Is v17 just "thinking suppression"?** (E10 answers — strongest hypothesis)
- **Did the hypernet collapse to fixed-vector?** (Diagnostic A answers)
- **Does verification-style training data break the thinking-suppression ceiling?** (v22 answers — the data-side hypothesis)

## Process (carry-forward)

- Always update this queue file when starting/completing/changing experiments.
- Manual inspection between phases — don't trust aggregate metrics, read 5+ raw completions per cell.
- Sync trained weights to local often — cloud boxes get reaped.
- Commit + push code fixes immediately.
- Submodule first, then parent: `cd axbench && git push origin mcpattack`, then `cd .. && git push origin axbench`.
