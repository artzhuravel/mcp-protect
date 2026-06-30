# Experiment queue + status (live)

Updated each wakeup. **Source of truth**. New plan from `EXPERIMENT_HANDOFF.md` (2026-05-05 19:07) supersedes v12-only path: v11_FIXED + v13 + v14 are the primary tracks.

## Boxes

| box | host | gpu | role |
|-----|------|-----|------|
| **box1** | `ubuntu@64.247.196.131` (`~/.ssh/primeintellect_ed25519`) | RTX A6000 48GB | v11_2k chain (ETA ~19:00 UTC) → v11_FIXED retrain at L20 → v14 (ReFT+DPO) |
| **box2** | `root@69.30.85.78 -p 22078` (same key) | NVIDIA A40 46GB | now: **Diagnostic B (mode-mismatch)**; then v13 phase 1.2 (concept expansion via API) |
| **box3** | `ubuntu@216.81.248.82` (same key) | **NVIDIA A100 80GB** | now: v12 phase 2 training (legacy, let it finish); then **v13 phase 1.3 vLLM gen → phase 2/3 train** |

## Currently running

| since | box | what | ETA |
|-------|-----|------|-----|
| 14:37 UTC | box1 | tmux `v11_2k`: trained v11 → 6 evals at 2048; now mid `v11_f0p5_2k` (5/6 cells done) | done ~19:00 UTC |
| 17:05 UTC | box3 | nohup v12 phase 2: bs=16 chosen by smoke (real batches); resuming at cid 16143 (cids 0-16142 already trained from prior run) | done ~22:00-02:00 UTC |
| **next** | box2 | **Diagnostic B**: v3_action thinking-on/off × f={0,0.5} N=20 | ~30 min |

## Queue (in priority order)

### box1 — after v11_2k finishes (~19:00 UTC)
1. **v11_FIXED** — single-concept retrain at **L20** with truncation fix:
   - Filter v11 96 rows to drop those >2048 Qwen3 tokens (expect ~5 dropped → 91 rows)
   - YAML: `train.layer: 20`, `train.models.HyperSteer.max_input_length: 2048`, `max_concept_length: 1024`
   - Re-run train.py; eval at f=[0.0, 0.3, 0.5, 0.7, 1.0] N=50 max_tokens=2048 temp=0.3
   - **GPU**: bs=8/seq=2048 ~34 GB (A6000 48GB has headroom)
   - **ETA**: ~12-18h
2. **v14 (ReFT+DPO control)** — after v11_FIXED:
   - PreferenceLoReFT or LsReFT, `use_dpo_loss=True`, beta=0.1, single concept, L20
   - Memory: ~40-45 GB (frozen ref policy doubles base load) — fits A6000
   - **ETA**: ~2-4h train + ~1-2h eval

### box2 — after Diagnostic B (~30 min from now)
1. **v13 Phase 1.2** — expand 280 → 2000 MCP-defense concepts via gpt-5.4-nano (API only, ~$0.50, ~3h)
2. **v13 Phase 1.4** — judge-filter generated rows via gpt-5.4-nano (~$5-10) [after box3 phase 1.3]
3. **diffmean direct-vec control** — apply `diffmean/outputs/acts/qwen3-thinking-decision/L20/diffmean_vec.pt` at L20 of un-steered Qwen3-8B, eval mcp_tox N=50 max_tokens=2048 at α=−2,−1,+1,+2

### box3 — after v12 phase 2 finishes (~22:00-02:00 UTC)
1. **v13 Phase 1.3** — Qwen3-8B vLLM thinking-mode gen, ~24K rollouts, max_new_tokens=1536, gpu-mem-util=0.9 → ~72 GB peak. ~4-6h.
2. **v13 Phase 2** smoke — REAL merged-parquet batches at bs=8/16/32, seq=2048; pick largest ≤70 GB (expect bs=16, ~50 GB)
3. **v13 Phase 3** train — Qwen3-8B HyperSteer at L20, bs=16/seq=2048, n_epochs=5, lr=2e-5. ~5-12h.
4. **v13 Phase 4** eval — headlines (10 cid >16001 at f=0/0.5/1.0) + neg controls (5 cid <16001) + generalization probe (5 cid matching refuse|tool|warning|caution)
5. **Diagnostic A** (post v11_FIXED) — cosine sim of hypernet v(prompt) on FRESH L20 hypernets; defer until at least one L20 model trained

## Recent done (today)

| time | box | result | AR (baseline 1k=0.66, 2k=0.64) |
|------|-----|--------|------------------------------|
| ~07:00 | box1 | headlines @ 1024, 6 configs | baseline=0.66, v3=0.58, v4_multi=0.64, v5=0.58, v6=0.66, v7=0.66 |
| 12:30 | box1 | v8/v6 extended | v8_f0p3=0.60, v8_f0p5=0.62, v8_f1p0=0.74 (incoherent), v6_f1p5=0.60, v6_f2p0=0.70 (lobotomy) |
| ~13:00 | box1 | factor sweep | v3_f0p1=0.54, v6_f0p1=0.64, v6_f0p3=0.60, v6_f1p0=0.56 |
| ~14:30 | box1 | v10_fixed @ 1024 | v10_f0p3=0.60, v10_f0p5=0.70 (~50% truncation artifact at 1024 cap) |
| 15:35 | box1 | baseline_v3 @ **2048** | **0.64** (1024 cap was inflating by +0.02) |
| 15:50 | box2 | v3_f0p5 @ 2048 | **0.58** |
| 16:11 | box2 | v4_multi_f0p5 @ 2048 | **0.64** (matches baseline) |
| 16:37 | box2 | v6_f0p5 @ 2048 | **0.58** |
| 16:56 | box2 | v8_f0p5 @ 2048 | **0.62** |
| 16:25 | box1 | v10_f0p5 @ 2048 N=50 | **0.81** if N=16 sample held; need to verify N=50 number |
| 17:08 | box2 | gemma9b paper-recipe queue | **CRASHED** — `HypernetConfig has no attribute 'layer_types'` (transformers 4.54 + Gemma2DecoderLayer compat). Deferred. |

**Take-away from box2 sanity reruns**: at max_tokens=2048, all four variants (v3, v4_multi, v6, v8) sit in [0.58, 0.64] band — within ±0.06 of baseline 0.64. Confirms what we already knew: existing variants do not deliver MCP defense at the published recipe. The L24 + truncation + direct-mode mismatch confounds were absorbing the signal.

## L20 vs L24 evidence (confirmed; commit pending)

DiffMean layer-sweep `diffmean/outputs/eval/layer_axis_allt_v2/L{16,20,24}/summary.jsonl`:

| layer | defense at α=−15 | defense at α=+15 | range | verdict |
|-------|-----------------|------------------|-------|---------|
| L16 | 0.633 | 0.500 | 0.20 | mediocre |
| **L20** | **0.667** | **0.333** | **0.33** | **strong steerability** |
| L24 | 0.567 | 0.633 | 0.07 | nearly inert |

**Mandate**: all future Qwen3-8B HyperSteer training uses **L20**. v3-v11 used L24 by mistake.

## v13 — multi-concept thinking-mode HyperSteer (PRIMARY) — handoff §6

**Hypothesis**: HyperSteer's value-add is amortization across many concepts. We've never tested it that way on Qwen3.

**Phase 1 — assemble training data** (~1 day):
- 1.1 Download AxBench 16K — `pyvene/axbench-concept16k_v2 gemma/9b/l20_131k/train/data.parquet`. Already downloaded on box3 at `/tmp/axbench16k`. Verify before reuse.
- 1.2 Expand 280 → 2000 MCP-defense concepts via gpt-5.4-nano paraphrasing (7× per seed). ~$0.50.
- 1.3 Generate thinking-mode y_neg via Qwen3-8B + vLLM on box3, 12 (input, output) pairs per concept, `enable_thinking=True`, max_new_tokens=1536. Filter rows >2048 tokens.
- 1.4 Judge filter for concept faithfulness via gpt-5.4-nano. Keep only `yes`. ~$5-10. Expected retention ~70%.
- 1.5 Concat with 16K, build metadata.jsonl indexed [0, max_cid]. **Verify length == max_cid+1** (train.py uses list indexing — see existing `_MetadataLookup` shim).

**Phase 2 — smoke** (~30 min, MUST do): bs=8/16/32 with REAL merged-parquet batches at seq=2048, pick largest ≤70 GB (expect bs=16 ~50 GB).

**Phase 3 — train** (~5-12h on A100): Qwen3-8B at L20, max_input_length=2048, max_concept_length=1024, n_epochs=5, lr=2e-5, low_rank_dimension=1, intervention_positions=all, intervention_type=addition, train_on_negative=false, hypernet_initialize_from_pretrained=true, num_hidden_layers=4. OOM monitor in bg. tmux + nohup + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Phase 4 — evaluate**:
- Headline (only these go in result table): 10 random cid >16001 × f=[0.0, 0.5, 1.0]
- Negative controls (should NOT shift): 5 random cid <16001 at f=0.5
- Generalization probe: search 16K labels for `refuse|tool|warning|caution|verify|suspicious`, pick 5, test at f=0.5
- vf-eval N=50 max_tokens=2048 temp=0.3 max_concurrent=1

**Decision criteria**:
| result | interpretation |
|--------|----------------|
| Our cid >16001 @ 0.5 AR > baseline+0.05; controls ≈ baseline | **WIN** — multi-concept thinking-mode HyperSteer delivers MCP defense, concept-conditional |
| Our cids shift; controls also shift | Hypernet emits "general defense" regardless of concept; useful but weaker |
| Our cids ≈ baseline; controls ≈ baseline | Multi-concept doesn't transfer; pivot to v14 |
| Generalization probe cids shift | Hypernet generalized — paper-worthy |

**Cost ceiling**: ~$15 API + 25-50h GPU + 1 day eng.

## v14 — ReFT+DPO control — handoff §7

**Hypothesis**: If v13 wins, we still need to know if hypernet machinery contributed beyond input-conditional fixed-vector steering. v14 strips the hypernet, uses ReFT (one learned rank-1 vector per concept) with DPO loss.

**Phase 1** (~1h): verify pairing in `diffmean/outputs/qwen3_v2_contrast.jsonl` (111 resist + 106 comply). If paired, build pairs file. If not, use `LsReFT` with unpaired-preference mode.

**Phase 2** (~2-4h): `PreferenceLoReFT` or `LsReFT`, `use_dpo_loss=True`, beta=0.1, reference_free=False, train_on_negative=True, n_epochs=5, bs=1, grad_accum=8, lr=2e-5, low_rank_dimension=1, L20. Memory ~40-45 GB (frozen reference policy).

**Phase 3** (~1-2h): same as v13 headline, single concept (cid=0), f=[0.0, 0.5, 1.0, 1.5].

**Decision**:
| result | interpretation |
|--------|----------------|
| v14 ≈ v11_FIXED (both ≈ baseline) | Single-concept methods fail; multi-concept (v13) load-bearing |
| v14 > v11_FIXED, both << v13 | DPO loss helps; multi-concept still wins |
| v14 ≈ v13 | Hypernet was overhead; ReFT+DPO simpler equivalent. Drop HyperSteer |
| v14 > v13 | Contrastive preference signal beats CE on y_neg with multi-concept richness. Surprising and publishable |

## v11_FIXED — single-concept retrain — handoff §5

**Why**: cleanest test of whether bugs alone account for v11's underperformance.

**Steps**:
1. Wait for v11_2k chain to finish, free box1.
2. Filter `axbench/outputs/mcp_hsteer_qwen3_8b_v11_audit/generate/train_data.parquet`: drop rows where Qwen3-tokenized `len(input + output) > 2048`. Save in-place; back up as `.bak`.
3. Edit YAML: `train.layer: 20` (was 24), `train.models.HyperSteer.max_input_length: 2048`, `max_concept_length: 1024`.
4. Re-run train.py.
5. Eval at f=[0.0, 0.3, 0.5, 0.7, 1.0] N=50 max_tokens=2048 temp=0.3.
6. Sync trained weights to local + commit.

**Decision criteria**:
| v11_FIXED best AR | interpretation |
|-------------------|----------------|
| > baseline + 0.10 (~0.74+) | Bugs alone explained underperformance. v13 may compound |
| baseline + 0.05 to 0.10 | Bugs partial; v13 worth running |
| baseline ± 0.05 | Single-concept HyperSteer fails even fully fixed. v13 is load-bearing |

## v12 — multi-concept paper recipe (LEGACY, running on box3)

Was the previous primary path. Now superseded by v13 (which uses thinking-mode generation instead of direct-mode AxBench data). Letting it finish for a baseline data point. Cids 0-16142 already trained from a prior run; current run resumes at 16143-16146. ~143 MCP-defense concepts will be trained.

After v12 finishes, eval for completeness (1-2 cells: random MCP cid + AxBench control). Then frees box3 for v13 phase 1.3.

## Diagnostic B — thinking-mode mismatch — handoff §3 (RUNNING)

Test whether direct-mode-trained variants (v3-v11) work differently with `enable_thinking=True` vs `False` at inference.

**Setup**: `MCP_ENABLE_THINKING` env var added to `serve_mcp_hypersteer.py` line 250+ (env var read in `_messages_to_prompt`); default True (preserves current behavior). vf-eval mcp_tox N=20 on v3_action at:
- f=0.0 thinking=True (baseline)
- f=0.0 thinking=False (baseline)
- f=0.5 thinking=True (current default)
- f=0.5 thinking=False

**Expected outcomes**:
| result | interpretation |
|--------|----------------|
| AR shifts in thinking-OFF but flat in thinking-ON | Mode mismatch confirmed → v13 thinking-mode generation is mandatory |
| AR flat in both | Mode mismatch isn't dominant; v13 still preferred but cheaper Option B viable |
| AR shifts in BOTH | Variant works generally; existing eval protocol may have masked this. Re-investigate |

## Diagnostic A — hypernet collapse cosine — handoff (DEFERRED)

Cosine sim of hypernet v(prompt) across 50 stratified prompts. Detects whether HyperSteer collapsed to fixed-vector steering. **Defer until L20 hypernets exist** (v11_FIXED at minimum). Running on the buggy L24 ones tells us nothing.

## Standing rules (handoff §9 + accumulated)

- `make_data_module` defaults `max_input_length: 1024`. For reasoning training set `train.models.<name>.max_input_length: 2048` in YAML AND drop rows above 2048 tokens pre-training.
- For Qwen3-8B steering: **USE LAYER 20**. L24 has classification AUC 0.818 but causal steerability range 0.07. L20 has range 0.33. Evidence: `diffmean/outputs/eval/layer_axis_allt_v2/L*/summary.jsonl`.
- Direct-mode training data (no `<think>`) applied to a thinking-mode model is a failure mode. v13's MCP-defense subset must be generated by Qwen3-8B in thinking mode itself. The 16K AxBench background is OK for training diversity but is NOT used for headline eval (cid >16001 only).
- `axbench/scripts/generate.py` crashes on non-Neuronpedia refs (URL parsing at line 404) AND has no Qwen3 entry in `model_name_map`. For our use case bypass it entirely — synthesize y_neg via custom Qwen3-self-generation script.
- AxBench's `output_length: 32` default is appropriate for direct-text concepts but useless for reasoning concepts. For audit/refusal y_neg generation use `max_new_tokens: 1536` minimum.
- vf-eval `--max-concurrent` > 1 deadlocks env worker — use 1.
- max_tokens=1024 truncates audit-prompted traces mid-think → fake +pts. Use 2048.
- New policy model? **Add to `axbench/utils/constants.py:CHAT_MODELS`** or serve drops the system prompt.
- Single-concept dataset uses `concept_id=0`; multi-concept w/ non-contiguous cids requires `_MetadataLookup` (already shimmed in train.py).
- Push the AI-Consensus parent repo, not the autinn submodule (no write access). **EXCEPT**: `mcpattack` branch on autinn/axbench DOES have push access (verified push of 7db36b4).
- **Inspect generated and intermediate data BEFORE training on it** (qualitative-analysis discipline). For v13: read 5+ samples per concept after Phase 1.3 (Qwen3 gen) AND after Phase 1.4 (judge filter). Verify thinking traces are coherent, that the output exemplifies the concept, and that the (input, output) pair looks like a real MCP-Tox interaction. Catch garbled outputs, judge errors, and concept drift before they get baked into training.

## Git discipline (executor responsibility)

### Order: submodule first, then parent (parent points at submodule SHA)

```
1. cd axbench && git add … && git commit && git push origin mcpattack
2. cd ..    && git add axbench EXPERIMENT_QUEUE.md … && git commit && git push origin axbench
```

### Commit IMMEDIATELY (block on these)
- Code fix in `axbench/`
- Trained model dump (sync + commit pointer)
- Eval result that informs a decision
- Update to EXPERIMENT_QUEUE.md
- Evidence behind a "we should do X" decision

### Never
- `--no-verify`, `--no-gpg-sign` (skip hooks/signing) unless explicitly asked
- `--amend` after the previous commit was pushed — make a new commit
- `git config` changes
- Force push to main / shared branches

### Skip
- `.claude/` (local Claude Code settings)
- `arXiv-*.tar.gz` (research papers, unrelated)
- Per-eval `results.jsonl` files (commit `metadata.json` and `summary.jsonl` only)
- Smoke run dirs

## Process habits

- **Always update this queue file** when starting/completing/changing experiments.
- **Manual inspection between phases** — don't trust aggregate metrics, read 5+ raw completions per cell.
- **Sync trained weights to local often** — cloud boxes get reaped.
- **Commit + push code fixes immediately**.
- **Decision tree** (handoff §10): see EXPERIMENT_HANDOFF.md.

## Open questions / followups
- v10_f0p5 @ 2048: is the 0.81 N=16 sample real at N=50?
- Mode-mismatch diagnostic outcome (running now)
- v11_FIXED outcome → does L20 + truncation fix unlock v11 alone?
- v13 outcome → does multi-concept thinking-mode generation deliver?
- v14 outcome → was the hypernet machinery actually contributing?
