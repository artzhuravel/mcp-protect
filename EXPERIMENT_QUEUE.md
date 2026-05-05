# Experiment queue + status (live)

Updated each wakeup (~10 min).

## Boxes

| box | host | gpu | role |
|-----|------|-----|------|
| **box1** | `ubuntu@64.247.196.131` (`~/.ssh/primeintellect_ed25519`) | RTX A6000 48GB | v11_2k chain → then v11_FIXED retrain+eval after data_utils truncation fix |
| **box2** | `root@69.30.85.78 -p 22078` (same key, set up 2026-05-05 14:55) | NVIDIA A40 46GB | sanity-rerun chain at 2048 → gemma9b paper-recipe queue |
| **box3** | `ubuntu@216.81.248.82` (same key, set up 2026-05-05 16:38) | **NVIDIA A100 80GB** | reserved for **v12 multi-concept training** (16K AxBench + ~280 our concepts) — needs the big GPU + big batch |

## Currently running

| time set | box | runner | action | ETA |
|----------|-----|--------|--------|-----|
| 14:37 UTC | box1 | tmux `v11_2k` | [waiting v10_FIXED_done.flag] → train v11 → 5 evals at 2048 | done ~21:00 UTC |
| 15:31 UTC | box2 | nohup pid 2704 | 4 sanity reruns at max_tokens=2048: v3 v4_multi v6 v8 (all @ f=0.5) | done ~17:15 UTC (~25 min/config) |

## Box-2 setup notes (2026-05-05)
- box2 needed `transformers>=4.51` for Qwen3 — installed 4.54.1 manually after `uv sync`
- box2 tmux is flaky (sessions sometimes don't survive "tmux new-session -d"); use `nohup ... &; disown`
- box2 SSH key: `~/.ssh/box2_to_box1` (added to box1's authorized_keys)
- box2 paths: `/root/mcp-protect/{axbench,prime-envs,diffmean}` and `~/.local/bin/uv`
- A40 is 3-4× faster per request than box1's A6000 (10s/example vs 30-90s/example)

## Queue

### box1 (after v11_2k completes, ~21:00)
- **none planned** — let v11_2k finish; review numbers before queueing more

### box2 (after smoke validates)
1. ✅ DONE/RUNNING: re-runs at max_tokens=2048 — v3, v4_multi, v6, v8 (all FACTOR=0.5)
2. ⏳ QUEUED behind chain: **gemma9b paper-recipe eval** — `mcp_hsteer_9b_v2_overnight` (2133 concepts: 2000 Neuronpedia + 133 mcpattack, L20, never previously evaluated). Runs 4 cells: baseline_f0 / Neuronpedia control cid=0 / mcpattack-MUST cid=2001 / mcpattack-SSH cid=2000 — all at f=0.5 max_tokens=2048. **Will need to download gemma-2-9b-it from HF** (~18GB, HF token now saved on box2).
3. **diffmean-direct-vec control** (planned, not yet queued): load `diffmean/outputs/acts/qwen3-thinking-decision/L24/diffmean_vec.pt`, apply at L24 of un-steered Qwen3-8B (no hypernet), eval mcp_tox N=50 max_tokens=2048 at α=−2,−1,+1,+2. Published-method baseline.
4. **v12 (planned)**: only if v11 looks promising — DPO contrastive with `train_on_negative=True`.

### Diagnostic A — are our hypernets being hypernets? (planned, defer until current chains free)

Hypothesis to test: if our hypernet's predicted v(prompt) is nearly constant across prompts (cos ≈ 1.0), then "HyperSteer" has collapsed to a fixed-vector steering method — explaining why all our variants ≈ baseline (a fixed vector applied at low factor is approximately a no-op; at high factor it lobotomizes). This would also explain why concept-text changes don't matter — the cross-attention isn't actually conditioning on anything useful.

**Inputs**:
- 5 dump dirs: `axbench/outputs/mcp_hsteer_qwen3_8b_{v3_action,v6_full,v8_refusal,v10_singlerefusal,v11_audit}`
- 50 prompts from `diffmean/outputs/qwen3_v2_contrast.jsonl` stratified:
  - ~17 poisoned-comply (label=comply)
  - ~17 poisoned-resist (label=resist)
  - ~16 clean (strip poisoned tool entry from system_prompt — see implementation note)

**Per (variant, prompt) — replicate `serve_mcp_hypersteer.py`'s v-generation path**:
1. `build_input_text(system_prompt, user_query)` → apply Qwen3 chat template via `_messages_to_prompt` (CHAT_MODELS gate must be on).
2. Forward Qwen3-8B base, capture hidden states at variant's training layer `L` (read from `train/config.json`; v3/v6/v8/v10 = L24, v11 = L24 also per their YAML).
3. Tokenize variant's concept_text (`generate/metadata.jsonl[0]["concept"]`).
4. Forward the loaded hypernet (`concept_embedding`) with cross-attention onto `base_hidden_state` → `v ∈ R^{d_model}` = last hidden state, rank-1 direction.

**Save**: `diffmean/outputs/hypernet_diagnostic_A.pt = {variant: {prompt_id: v_tensor}}`.

**Statistics per variant**:
- `v_mean = mean(v_per_prompt over 50 prompts)`
- For each prompt: `cos(v_per_prompt, v_mean)` → mean, std, min, max
- Pairwise cosine matrix (50×50); report mean off-diagonal cos
- Norm distribution: `||v_per_prompt||_2` mean / std (catch case where direction varies but a constant magnitude dominates)
- **Decision rule**:
  - mean off-diagonal cos > 0.95 → hypernet collapsed; HyperSteer ≈ fixed-vector steering; switch to plain DiffMean and stop iterating on data variants
  - mean cos in [0.5, 0.95] → partial collapse; concept-text matters less than we thought
  - mean cos < 0.5 → hypernet is genuinely input-conditional; the data-recipe is still the right axis

**Implementation note on the "clean" stratum**: qwen3_v2_contrast.jsonl is 100% poisoned. For the clean stratum, take 16 of the same prompts and substitute their system_prompt with one that lists ONLY the legitimate tool (strip the poisoned-named tool block). Easiest impl: regex-remove the matching `\nTool: <poisoned_name>\n.*?(?=\nTool: |$)` block from each.

**Cost**: 5 variants × 50 prompts × 1 forward (no generation) ≈ 250 hypernet forward passes. ~5 min per variant on box2, ~25 min total. Trivial. No vf-eval, no judge, no API spend.

**When to run**: after box2 main chain + gemma chain are done (so we don't compete with serve for GPU). ETA earliest: ~20:00 UTC tonight.

Script: `/tmp/diag_A_hypernet_collapse.py` ✅ written and synced to box1. Ready to fire.

To run (box1, after v11_2k done):
```
cd /home/ubuntu/mcp-protect/axbench && export PATH=$HOME/.local/bin:$PATH \
  && uv run --no-sync python /tmp/diag_A_hypernet_collapse.py 2>&1 | tee /tmp/diag_A.log
```

Output written to `/home/ubuntu/mcp-protect/diffmean/outputs/hypernet_diagnostic_A.pt`. Sync down for analysis.

## Recent done (today)

| done time | box | result | AR (@ 1k baseline 0.66, 2k baseline 0.64) |
|-----------|-----|--------|----------------------|
| ~07:00 UTC | box1 | headlines @ max_tokens=1024, 6 configs | baseline=0.66, v3=0.58, v4_multi=0.64, v5=0.58, v6=0.66, v7=0.66 |
| 12:30 UTC | box1 | v8/v6 extended sweep | v8_f0p3=0.60, v8_f0p5=0.62, v8_f1p0=0.74 (incoherence), v6_f1p5=0.60, v6_f2p0=0.70 (lobotomy) |
| ~13:00 UTC | box1 | factor sweep | v3_f0p1=0.54, v6_f0p1=0.64, v6_f0p3=0.60, v6_f1p0=0.56 |
| ~14:30 UTC | box1 | v10_fixed @ 1024 | v10_f0p3=0.60, v10_f0p5=0.70 (~50% truncation artifact at 1024 cap) |
| 15:35 UTC | box1 | baseline_v3 @ **2048** | **0.64** (1024 cap was inflating by +0.02) |
| 15:50 UTC | box2 | v3_f0p5 @ **2048** | **0.58** (same as 1024, no truncation effect on this variant) |
| 16:00 UTC | box1 | v10_f0p5 @ **2048** partial 16/50 | **0.81** — if holds, +17 vs real baseline 0.64. **WATCH** |
| 15:55 UTC | box2 | v4_multi @ **2048** partial 7/50 | 0.857 (tiny sample, will regress) |

## Open questions / followups
- Does v10_f0p5 +4 hold under max_tokens=2048? (re-run queued on box2)
- Does v11 audit-prompted produce real scope-test traces at f=0.5? (running on box1)
- What's diffmean's direct-vec baseline number on N=50? (queue on box2 after smoke)

## Critical bug fix landed (2026-05-05 16:35) — `data_utils.py` truncation

`axbench/utils/data_utils.py:206-209,218` had hardcoded `max_length=1024` for the input/output and concept tokenization. For v11 (audit-prompted long traces): **38% of training rows had input alone > 1024 → entire output truncated → row contributed ZERO loss.** v6 also damaged (56% rows had output partially cut).

**Fix landed** (waiting to ship to box1/2/3): added `max_input_length` / `max_concept_length` kwargs to `make_data_module`, plumbed through `train.py`'s kwargs dict to read from YAML's `models.<name>.max_input_length`. v4 builder template now defaults to `max_input_length: 2500, max_concept_length: 1024`.

**Damage table** (positives only, Qwen3-8B tokenizer):

| variant | n | in_p50 | in_max | combined>1024 | ZERO-loss rows | impact |
|---|---|---|---|---|---|---|
| v3 | 60 | 838 | 983 | 1 (2%) | 0 | clean |
| v6 | 45 | 787 | 928 | 25 (56%) | 0 | output partially cut |
| v8 | 142 | 754 | 983 | 7 (5%) | 0 | clean |
| v10 | 142 | 754 | 983 | 7 (5%) | 0 | clean |
| **v11** | **72** | **900** | **1497** | **59 (82%)** | **27 (38%)** | **27/72 trained on nothing!** |

**Re-run plan**: v11 must be re-trained with TWO fixes simultaneously:
1. data_utils.py truncation fix (max_input_length=2500)
2. **MOVE FROM L24 TO L20** — diffmean phase-1 found L20 was the empirical best layer for AUC; L24 was a guess. Two confounds removed in one run.

After v11_L20 trained, also re-run **Diagnostic A on the fresh L20 hypernets**. Diag A on the buggy L24 hypernets would just measure variance in vectors that aren't doing much anyway — wait for the fixed model.

v6 retrain: lower priority — schedule alongside v11 if box has capacity.

v3/v8/v10 are fine (no truncation), but if we want apples-to-apples vs v11_L20 we'd retrain at L20 too.

## v12 — multi-concept paper recipe on Qwen3-8B (RUNNING / queued, box 3 A100)

**Why v12 is a primary path, not gated on v11**: v11's 72-row result tells us nothing about whether the published HyperSteer recipe works at scale. v12 = 1.15M rows × 16K concepts = the actual recipe under test. A100 is idle so no reason to defer. Worst case (v12 also doesn't beat baseline) is itself a strong negative result for HyperSteer-as-MCP-defense; best case (v12 + concept-conditional generalization) is the headline win.

Test the actual published HyperSteer recipe on Qwen3-8B with full multi-concept richness.

### Phase 1 — assemble training data (~3-4h, box 3, mostly CPU + 1 LLM-API spend)

1.1 Download AxBench published 16K dataset
```
hf hub download pyvene/axbench-concept16k_v2 --include 'gemma/9b/l20_131k/train/data.parquet' --local-dir /tmp/axbench16k
```
325MB, 1.15M rows, 16,001 concepts. Model-agnostic — outputs are gpt-4o-mini-synthesized, no Gemma references.

1.2 Extract our new MCP-defense concepts. Sources:

| source | concepts | notes |
|---|---|---|
| `axbench/data/mcpattack.jsonl` | 133 | hand-written attack-defense concepts |
| `build_v3_dataset.py` | 1 | V3_CONCEPT_TEXT |
| `build_v3_think_dataset.py` | 1 | V3_THINK_CONCEPT_TEXT |
| `build_v4_dataset.py` | up to 133 | per-security_risk Autinn taxonomy (overlaps mcpattack) |
| `build_v6_full.py` | 1 | V6_CONCEPT_TEXT (if exists separately — likely uses the v4 set) |
| `build_v8_refusal.py` | 1 (or 8) | V8 templates |
| `build_v10_singlerefusal.py` | 1 | V10_CONCEPT_TEXT |
| `build_v11_audit.py` | 1 | V11_CONCEPT_TEXT |

After de-dup (v4_secrisk overlaps mcpattack), expect **~200-280 unique new concepts**.

Write `axbench/mcp-protect/extract_new_concepts.py` → outputs `new_concepts.jsonl` with `{concept_id (offset 16001+), concept, ref}` per row. Manual eyeball for de-dup quality after.

1.3 Generate training rows for new concepts via AxBench's `generate.py`:
```
python axbench/scripts/generate.py \
    --concept-path new_concepts.jsonl \
    --lm-model openai/gpt-5.4-nano \
    --num-of-examples 12 \
    --output-dir /tmp/new_concepts_train
```
Cost: ~$0.50-2 OpenRouter. Time: 30-60 min. Output: ~3K rows for ~280 new concepts.

1.4 Concat parquets:
```python
import pandas as pd
df16k = pd.read_parquet('/tmp/axbench16k/.../data.parquet')
dfnew = pd.read_parquet('/tmp/new_concepts_train/train_data.parquet')
# Verify concept_id ranges don't overlap; offset if needed.
merged = pd.concat([df16k, dfnew], ignore_index=True)
merged.to_parquet('/home/ubuntu/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v12_16k/generate/train_data.parquet')
```

Build matching `metadata.jsonl` (one row per concept).

### Phase 2 — Smoke-test batch size on A100 (~30 min, MUST DO before phase 3)

A100 80GB lets us push way past Autinn's batch_size=1. Test bs=4, 8, 16, 32 with 50 steps each, observe peak GPU memory. Pick the largest bs that uses ≤70GB peak (10GB headroom). Also confirm `gradient_accumulation_steps` reduction doesn't change loss trajectory.

Estimated: A6000 trained 1.15M rows at bs=1×ga=8 → 720K steps → 24-48h. A100 at bs=8×ga=1 should be ~10x faster with similar effective batch size = **2-5 hours**. If we can push bs=16, even faster.

### Phase 3 — train (~5-12h on A100 depending on batch size)

YAML: copy from `mcp_hsteer_qwen3_8b_v11_audit/mcp_hypersteer_config.yaml`. Change:
- target = Qwen/Qwen3-8B, layer = 20, hypernet_name_or_path = Qwen/Qwen3-8B
- batch_size from smoke result, grad_accum compensating
- `max_concepts: null` (use all)
- **`max_input_length: 2500, max_concept_length: 1024`** (the truncation fix!)
- defaults retained: `low_rank_dimension: 1, n_epochs: 5, lr: 2e-5, intervention_positions: all, intervention_type: addition, train_on_negative: false, hypernet_initialize_from_pretrained: true, num_hidden_layers: 4`

OOM monitor in background (per `/tmp/queue_v11_2k.sh` pattern). tmux session.

### Phase 4 — evaluate (~4-6h, either box)

Concept_ids to test (interspersed for fast comparison):
- **Our MCP-defense concepts**: cid 16001-16280 (these should produce real defense if recipe works)
- **Random Gemma Scope controls**: cid 0, 4732, 8000 (semantically irrelevant — should NOT shift AR)
- **AxBench concepts that semantically overlap MCP defense**: search 16K labels for "refuse", "tool", "warning", "caution" → cid list (these MIGHT shift AR if hypernet generalizes)

For each: vf-eval mcp_tox N=50 max_tokens=2048 temp=0.3 at factors [0.0, 0.5, 1.0].

### Decision criteria

| Result | Interpretation |
|---|---|
| Our new cids @ 0.5 AR > baseline+0.05; controls ≈ baseline | **Win** — multi-concept hypernet works AND is concept-conditional |
| Our new cids @ 0.5 AR > baseline+0.05; controls ALSO shift | Hypernet emits a "general defense" direction regardless of concept → not concept-conditional |
| Our new cids ≈ baseline; controls ≈ baseline | Multi-concept recipe doesn't transfer to MCP defense; pivot to ReFT or activation-direct |
| Random AxBench cids matching "refuse"/"tool" semantically also shift AR | Hypernet generalized → strongest possible result |

## Paper-artifact sync to local (DO REGULARLY)

The trained weights are the artifacts we'd ship with the paper. Sync after each retrain:

```bash
rsync -avz -e "ssh -i ~/.ssh/primeintellect_ed25519" --include='*/' --include='train/hyperreft/*' --include='generate/*' --include='*.yaml' --include='merged_concepts_mcp.json' --exclude='*' ubuntu@64.247.196.131:/home/ubuntu/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v11_audit/ /Users/hubertpysklo/Documents/Github/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v11_audit/
```

Same for v12 once trained, gemma9b_v2_overnight, and any other variants we want to publish.

## Diagnostic B — thinking-mode mismatch sanity check (planned, ~30 min on box2)

**Hypothesis to test**: v3-v11 hypernets were trained with the model's chat template (which on Qwen3 defaults to `enable_thinking=True`, giving `<think>…</think>` blocks). If at inference time the template is silently in non-thinking mode (or vice versa), the hypernet sees a different distribution than at training and the steering vector is mis-aligned.

**Test**: take an existing v3 hypernet, run vf-eval mcp_tox N=20 in TWO conditions:
1. `enable_thinking=True` (current default — what we've been measuring)
2. `enable_thinking=False`

If v3 produces meaningful AR shift in non-thinking mode but not in thinking mode (or vice versa), the mismatch hypothesis is confirmed and we need to either match training mode or train both modes. If v3 is flat in both modes, the mismatch isn't dominant.

Cost: ~30 min on box 2 (one extra config of an existing model). Run after current box 2 chain finishes.

## Git discipline (executor responsibility)

### Order matters: submodule first, then parent

axbench is a submodule pointing at `autinn/axbench` (we DO have push access to the `mcpattack` branch — verified 2026-05-05 16:55, push of `7db36b4` succeeded). If you bump the parent's pointer before pushing the submodule, the parent points at a SHA the remote doesn't have → broken clone for everyone else.

```
1. cd axbench && git add … && git commit && git push origin mcpattack
2. cd ..    && git add axbench EXPERIMENT_QUEUE.md … && git commit && git push origin axbench
```

### What to commit IMMEDIATELY (block on these — do not start new work first)

These are load-bearing artifacts that exist only on the laptop. Until they're on remote, every box is one laptop crash from being unrecoverable:
- Code fix in `axbench/` (e.g. data_utils.py truncation fix)
- Trained model dump (`axbench/outputs/mcp_hsteer_*`) — sync to local + commit pointer
- Eval result that informs a decision (full results.jsonl + metadata.json)
- An update to EXPERIMENT_QUEUE.md
- Evidence behind a "we should do X" decision

### Going forward — rule of thumb

Commit + push BEFORE moving on whenever you produce one of the above.
Smaller, more frequent commits are correct. Don't batch. The cost of an extra `git push` is zero. The cost of a reaped box or laptop crash with uncommitted work is hours-to-days of re-derivation.

## Process habits to remember

- **Always update this queue file** when starting/completing/changing experiments. Source of truth.
- **Manual inspection between phases** — don't trust aggregate metrics, read 5+ raw completions per cell. The v10 0.81→0.66 sample-size delusion shows what happens otherwise.
- **Max out GPU util for v12** on A100. Smoke-test batch sizes first; pick the largest that fits with 10GB headroom.
- **Sync trained weights to local often**. Each retrain produces a paper-shareable artifact; cloud boxes get reaped.
- **Commit + push code fixes immediately** (see git discipline section).

## Standing rules I keep getting bitten by
- vf-eval `--max-concurrent` > 1 deadlocks env worker — use 1.
- max_tokens=1024 truncates audit-prompted traces mid-think → fake +pts. Use 2048.
- New policy model? **Add to `axbench/utils/constants.py:CHAT_MODELS`** or serve drops the system prompt.
- Single-concept dataset must use `concept_id=0` (train.py uses Python list indexing).
- Push the AI-Consensus parent repo, not the autinn submodule (no write access).
