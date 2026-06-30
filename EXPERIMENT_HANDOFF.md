# Handoff to executor — v11_FIXED + v13 + v14, with critical context

> **Read this in full before acting.** This document was written near a context-compaction boundary in the research session that produced v13/v14. The plan and the bugs we found span ~5 hours of conversation; some are not yet in `EXPERIMENT_QUEUE.md`. If anything here conflicts with the queue, this doc is newer.

---

## TL;DR — what to do, in order

1. **Commit + push the truncation fix** (sitting uncommitted in axbench submodule). Block on this.
2. **Update `EXPERIMENT_QUEUE.md`** with the new standing rules + v13/v14 entries (specs in §6 §7 below). Commit + push.
3. **Let the running chains finish** (box1 v11_2k chain, box2 sanity reruns + Gemma 9b paper-recipe eval). Don't kill anything mid-flight.
4. **Run the mode-mismatch diagnostic** (§3) — 30 min on whichever box frees first. Result decides v13 priority.
5. **Re-train v11 with the fix** (§5) — single-concept, the simplest test of "does fixing the bugs alone unlock v11?"
6. **Kick off v13 Phase 1 on box3** in parallel (data assembly is API + light GPU; doesn't block on v11_FIXED).
7. **v14 fires after v13 trains**, on whichever box is free.

---

## §0 — How this file relates to `EXPERIMENT_QUEUE.md`

Two files, two roles. Don't conflate them:

| File | Role | Cadence | Source of truth for… |
|------|------|---------|----------------------|
| **`EXPERIMENT_HANDOFF.md`** (this file) | Research **spec** + decisions made at the most recent compaction boundary. Stable across boxes. | Updated only when research direction changes (i.e., new conversation produces new specs). | What to run, why, and what counts as a result. The "north star" for the next 1-3 days. |
| **`EXPERIMENT_QUEUE.md`** | **Live operational** status: what's running RIGHT NOW, what just finished, current GPU utilization, recent results table, standing footguns. | Updated every ~10 min during active work, after every kicked-off run, after every completed eval. | What's currently happening on each box, what the latest numbers are, what to NOT preempt. |

**Where each lives in the workflow**:
- Wake up / start of session → read `EXPERIMENT_HANDOFF.md` first (it tells you what we're trying to do and why), then `EXPERIMENT_QUEUE.md` (it tells you what's already in flight).
- Mid-session → update `EXPERIMENT_QUEUE.md` continuously. Don't touch the handoff unless a result invalidates a spec in it (in which case, leave a one-line note in the handoff pointing at the queue's "Recent done" entry).
- End of session / handing back → if results came in, summarize in queue's "Recent done"; only update the handoff if the next steps in §10's decision tree have been taken (in which case mark which branch we're on).

**Conflict resolution**: if the queue says one thing and the handoff says another, the handoff is canonical for *plan and method*; the queue is canonical for *current state and recent observations*. They shouldn't conflict on plan — if they do, the queue is stale and needs updating.

**First action of every session**: skim §10 (decision tree). It tells you which experiment to start based on what's already happened. The queue's "Currently running" + "Recent done" tell you what's already happened.

**Both files are tracked in git.** Commit + push the queue with each meaningful update (per §8). The handoff is more stable but commit it any time you append a result-summary note.

---

## §1 — Critical context the queue may not yet reflect

These were discovered or refined in the last research session. **Verify each is captured in the queue before continuing**; if not, add to "Standing rules I keep getting bitten by" or the relevant phase.

### 1.1 Training-time truncation bug — the load-bearing one

`axbench/utils/data_utils.py:206-220` previously hardcoded `max_length=1024` for both prompt and prompt+output tokenization. Fix has landed but is **uncommitted on the laptop**.

**v11 damage** (positives only, Qwen3-8B tokenizer):
- 96 total rows (72 positive + 24 negative)
- 59/96 (61%) had total > 1024 → truncated
- 27/96 (28%, or 38% of positives alone) had INPUT alone > 1024 → output entirely dropped → row contributed zero loss
- 68.8% of all output tokens lost across the dataset

**v6 damage**: 56% rows had outputs partially cut (no zero-loss rows, but signal heavily degraded).

**v3, v8, v10**: clean (<5% truncation).

**Fix**: `make_data_module` now accepts `max_input_length` and `max_concept_length` kwargs, threaded from YAML `train.models.<name>.max_input_length`. Default kept at 1024 for backward compat — **every YAML for reasoning training MUST set it explicitly**.

**Important refinement**: don't set it to 4096. Set to **2048** and DROP rows above that threshold pre-training. Reasons in §4.

### 1.2 Layer choice — we trained at the wrong layer

All v3-v11 hypernets were trained at L24. The DiffMean layer-sweep on disk (`diffmean/outputs/eval/layer_axis_allt_v2/L{16,20,24}/summary.jsonl`) shows:

| Layer | Best defense | Range across α=±15 | Verdict |
|-------|-------------|---------------------|---------|
| L16 | 0.633 | 0.20 | Mediocre steerability |
| **L20** | **0.667** | **0.33** | **Strongest steerable layer** |
| L24 | 0.633 | **0.07** | Nearly inert — barely moves behavior |

L24 has high *classification* AUC (0.818, the activation direction reads comply/resist well) but low *causal* steerability (range 0.07). We conflated readability with steerability. **For all future Qwen3-8B training: use L20.** This includes v11_FIXED, v13, v14.

The activation stats file at `diffmean/outputs/acts/qwen3-thinking-decision/stats.jsonl` is the evidence — commit it before relying on the decision.

### 1.3 Mode mismatch — direct-mode training on a thinking-mode model

Qwen3-8B is hybrid reasoning. At inference we use thinking mode (chat template emits `<think>...</think>` blocks). When training data y_neg is direct text (no `<think>`), the hypernet learns to push residuals toward a distribution that doesn't match what the model is producing at inference time.

**Suspected impact on v3-v10**: their flat-at-baseline AR may partly reflect this mismatch on top of the L24/truncation issues.

**Confirmation experiment** (§3): run an existing variant in thinking-on vs thinking-off mode. Cheap.

**Implication for v13**: training data for the MCP-defense subset must be generated *by Qwen3-8B in thinking mode itself* (vLLM on box3) so y_neg includes the `<think>` content the model will actually produce at inference.

### 1.4 `generate.py` crashes on non-Neuronpedia refs and doesn't know about Qwen3

`axbench/scripts/generate.py:404`:
```python
model_name = model_name_map[all_refs[0].split("/")[3]]
```

- `model_name_map` only contains Gemma + Llama keys; no Qwen3 entry.
- The lookup parses Neuronpedia URLs. Our `mcpattack.jsonl` refs are `"MCPTox"` (no slashes) → `IndexError` on first row.

**Implication for v13**: the AxBench `generate.py` script CANNOT be used to synthesize y_neg for our new MCP concepts. Use the custom Qwen3-self-generation script described in v13 Phase 1.3 instead. (Also: even if patched, generate.py defaults to `output_length: 32` which is far too short for reasoning concepts — would produce stub continuations.)

### 1.5 SAE is a labeling tool, not a training input

In case this comes up: AxBench's `GemmaScopeSAE.pt` exists in some dump dirs, but it is NOT used during HyperSteer training. The SAE provides (a) the source of the 16K concept-text taxonomy via Neuronpedia auto-interp, and (b) a separate baseline steering method (`GemmaScopeSAE` at eval time). The hypernet itself never sees SAE weights. So we do NOT need a Qwen3 SAE to train HyperSteer on Qwen3 — the published 16K dataset works directly.

Qwen-Scope SAEs (`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100`) exist but are trained on Qwen3-8B-**Base**, not the hybrid post-trained model we use. We don't need them for v13. Out of scope for now.

---

## §2 — Things still running (do not touch)

| box | what | ETA |
|-----|------|-----|
| box1 | tmux `v11_2k` chain — train v11 then 5 evals @ 2048 max_tokens | ~21:00 UTC tonight |
| box2 | sanity reruns of v3/v4_multi/v6/v8 @ max_tokens=2048; then queued: Gemma-9b paper-recipe eval (4 cells) | sanity ~17:15 UTC, gemma ~21:00 UTC |

**Don't kill these.** v11_2k will give us a "broken-data, broken-layer trained" baseline against which v11_FIXED's improvement can be measured. Sanity reruns confirm the eval-time max_tokens fix isolates that effect from training-time truncation. Gemma-9b eval validates the multi-concept recipe on a different model class.

When chains finish, sync results to local + commit per §8.

---

## §3 — Mode-mismatch diagnostic (run first, ~30 min)

**Goal**: confirm whether v3-v10's flat AR is partly from training-data-mode mismatching inference-mode.

**Setup**: pick one trained variant with simple direct-mode y_neg — `v3_action` is fine. Run vf-eval mcp_tox at:
- N=20 prompts, factor=0.5, **`enable_thinking: false`** in chat template
- Same N=20 prompts, factor=0.5, **`enable_thinking: true`** (current default)

Run baseline (factor=0) in both modes for comparison.

**Expected outcomes**:
| Result | Interpretation |
|--------|----------------|
| AR shifts in thinking-OFF but flat in thinking-ON | Mode mismatch confirmed → v13's thinking-mode generation is mandatory |
| AR flat in both | Mode mismatch isn't dominant → v13 still preferred but Option B (cheaper) viable |
| AR shifts in BOTH | Variant works generally; existing eval protocol may have masked this. Re-investigate. |

**Cost**: ~30 min on whichever box frees first. No new training, no API spend beyond the gpt-5.4-nano judge.

**Output**: short note appended to `EXPERIMENT_QUEUE.md` under "Recent done" with the four AR numbers.

---

## §4 — Truncation handling (applies to v11_FIXED, v13, v14)

**Don't fix the bug by raising the cap to 4096.** Doing so halves max batch size and doubles training wall-clock for ~5% more rows. Instead, **align all three thresholds at 2048 and pre-filter rows that exceed it**:

```
gen-time max_new_tokens (in synth scripts):              1536
post-gen filter (drop rows where input+output > 2048):   strict
training-time max_input_length (in YAML):                2048
training-time max_concept_length:                        1024  (concept text is short)
```

For v11_FIXED: filter v11's existing 96 rows by tokenized length, drop the 5 over 2048, train on 91 rows at `max_input_length: 2048`.

For v13: bake the filter into the gen pipeline (Phase 1.3).

**Memory math at A100 80GB** (Qwen3-8B + cross-attn hypernet):

| seq_len | bs=4 | bs=8 | bs=16 | bs=32 |
|---------|------|------|-------|-------|
| 2048 | ~26 GB | ~34 GB | **~50 GB** | OOM |
| 4096 | ~34 GB | ~50 GB | OOM | OOM |

At seq_len=2048 we can comfortably push bs=16, which makes training ~2× faster than the bs=8 ceiling at seq_len=4096.

---

## §5 — v11_FIXED — re-train v11 with all fixes (~12-18h on box1)

**Why first**: cleanest test of whether the bugs alone account for v11's underperformance. Cheap (single-concept training), uses existing data, isolates "fix everything that wasn't the hypothesis" from "test a new hypothesis."

### Steps

1. Wait for v11_2k chain to complete and free box1.
2. Filter `axbench/outputs/mcp_hsteer_qwen3_8b_v11_audit/generate/train_data.parquet`: drop rows where Qwen3-8B-tokenized `len(input + output) > 2048`. Save filtered version in-place (back up the pre-filter as `.bak`).
3. Edit `axbench/outputs/mcp_hsteer_qwen3_8b_v11_audit/mcp_hypersteer_config.yaml`:
   ```yaml
   train:
     layer: 20                          # was 24
     models:
       HyperSteer:
         max_input_length: 2048         # was implicit 1024
         max_concept_length: 1024
         # ...everything else unchanged
   ```
4. Re-run train.py (use `/tmp/queue_v11_2k.sh` as a template, adapt the eval cell list).
5. Eval at factors [0.0, 0.3, 0.5, 0.7, 1.0] on N=50 max_tokens=2048 temp=0.3.
6. Sync trained weights to local + commit (axbench submodule).

### Decision criteria

| v11_FIXED best AR | Interpretation |
|-------------------|----------------|
| > baseline + 0.10 (~0.74+) | Bugs alone explained v11's underperformance. Single-concept HyperSteer with good data + L20 + no truncation works. v13 multi-concept may further compound. |
| baseline + 0.05 to 0.10 | Bugs were partial; v11's hypothesis is half-right. v13 worth running. |
| baseline ± 0.05 (no shift) | Single-concept HyperSteer doesn't deliver MCP defense even fully fixed. The hypothesis or the architecture is the problem, not the bugs. v13 is the load-bearing test. |

---

## §6 — v13 — multi-concept thinking-mode HyperSteer (PRIMARY)

**Hypothesis**: HyperSteer's value-add over per-concept ReFT is amortization across many concepts. We've never tested it that way on Qwen3. v13 trains on 16K AxBench concepts (background) + ~2000 MCP-defense concepts with mode-consistent thinking-trace y_neg, evaluated only on the 2000 to test whether multi-concept richness + correct training mode delivers MCP defense.

**Box**: box3 (A100 80GB).

### Phase 1 — assemble training data (~1 day)

**1.1** Download 16K background:
```
hf hub download pyvene/axbench-concept16k_v2 \
    --include 'gemma/9b/l20_131k/train/data.parquet' \
    --local-dir /tmp/axbench16k
```
325 MB, 1.15M rows, schema `(input, output, output_concept, concept_genre, category, dataset_category, concept_id)`. Model-agnostic (gpt-4o-mini-synthesized outputs); no Gemma reference in row contents.

**1.2** Expand 280 → 2000 MCP concepts. Sources:
- `axbench/data/mcpattack.jsonl` (133)
- `build_v3_dataset.py`, `build_v3_think_dataset.py`, `build_v6_*.py`, `build_v8_refusal.py`, `build_v10_singlerefusal.py`, `build_v11_audit.py` (≤8 unique concept strings)
- `build_v4_dataset.py` per-security_risk Autinn taxonomy (~133, overlaps mcpattack — de-dup)

After de-dup: ~280 unique. Expand to ~2000 via gpt-5.4-nano paraphrasing:
> "Generate 7 paraphrases of this concept that preserve meaning but vary surface form. Output as JSON list of strings: {concept_text}"

Cost ~$0.50. Save as `axbench/mcp-protect/v13_concepts_2k.jsonl` with `{concept_id (offset 16001+), concept, ref}`.

**1.3** Generate thinking-mode y_neg via Qwen3-8B on box3 with vLLM:

Custom script: `axbench/mcp-protect/synth_thinking_data.py`. Per concept, ~12 (input, output) pairs:
- System message frames the concept as guidance ("As a security-aware assistant, you should: {concept_text}")
- User message is a synthetic MCP-Tox-style prompt (poisoned tool descriptions in system, benign user query) — sample from `qwen3_v2_contrast.jsonl` style
- vLLM batched 50-100 in flight, `enable_thinking=True`, `max_new_tokens=1536`, `temperature=0.7`
- Output = full `<think>...</think>` + final tool call → save as `output` field

Post-gen filter: drop rows where `len(qwen3_tokenizer(input + output)) > 2048`. Expected drop rate ~5-15%.

**1.4** Judge filter for concept faithfulness via gpt-5.4-nano:
> "Does this assistant response (containing `<think>` reasoning) exemplify the concept '{concept_text}'? Answer yes/no with a one-line reason."

Keep only `yes`. Cost ~$5-10 OpenRouter for ~24K rows. Expected retention ~70%.

**End of Phase 1**: ~14-16K clean rows for ~2000 concepts.

**1.5** Concat with 16K AxBench:
- AxBench cids: 0–16,000
- Our cids: 16,001–18,000 (offset)
- Build matching `metadata.jsonl` indexed [0, max_cid]. **Verify length == max_cid+1 — train.py uses Python list indexing.**
- Save as `axbench/outputs/mcp_hsteer_qwen3_8b_v13_thinking/generate/train_data.parquet`

### Phase 2 — batch-size smoke (~30 min, MUST do)

Test bs=8/16/32 with 50 steps each, **using real batches sampled from the merged parquet** (not random tokens — long thinking rows drive memory). Pick the largest bs ≤ 70 GB peak. Expect bs=16 fits.

### Phase 3 — train (~5-12h)

YAML (write to `mcp_hsteer_qwen3_8b_v13_thinking/mcp_hypersteer_config.yaml`):
```yaml
train:
  model_name: Qwen/Qwen3-8B
  layer: 20
  models:
    HyperSteer:
      max_input_length: 2048
      max_concept_length: 1024
      hypernet_name_or_path: Qwen/Qwen3-8B
      batch_size: <smoke result, expect 16>
      gradient_accumulation_steps: 1
      n_epochs: 5
      lr: 2e-5
      low_rank_dimension: 1
      intervention_positions: all
      intervention_type: addition
      train_on_negative: false
      hypernet_initialize_from_pretrained: true
      num_hidden_layers: 4
```

OOM monitor in background. tmux. Log to `/tmp/train_v13.log`.

**Commit + push trained weights as soon as training completes** — paper artifact, cloud boxes get reaped.

### Phase 4 — evaluate (~3-5h)

**Headline cells** (only these go in result table):
- 10 random `cid > 16001`, factors [0.0, 0.5, 1.0]
- baseline (no steering)

**Negative controls** (should NOT shift):
- 5 random `cid < 16001` (AxBench background, direct-mode), factor=0.5

**Generalization probe**:
- Search 16K labels for `refuse|tool|warning|caution|verify|suspicious`, pick 5, test at factor=0.5
- If these shift AR despite NOT being trained mode-consistent, the hypernet generalized to MCP defense across concept space — strongest result possible

vf-eval N=50 max_tokens=2048 temp=0.3 max_concurrent=1.

### Decision criteria (write into queue when phase 4 lands)

| Result | Interpretation |
|--------|----------------|
| Our cid > 16001 @ 0.5: AR > baseline+0.05; controls ≈ baseline | **WIN** — multi-concept thinking-mode HyperSteer delivers MCP defense, concept-conditional |
| Our cids shift; controls also shift | Hypernet emits "general defense" regardless of concept; useful but weaker claim |
| Our cids ≈ baseline; controls ≈ baseline | Multi-concept doesn't transfer to MCP defense even mode-consistent. Pivot to v14 / direct activation steering |
| Generalization probe cids shift | Hypernet generalized — paper-worthy |

**Cost ceiling**: ~$15 API + 25-50h GPU + 1 day eng.

---

## §7 — v14 — ReFT + DPO control experiment

**Hypothesis**: If v13 wins, we still need to know if the hypernet machinery contributed beyond input-conditional fixed-vector steering. v14 strips the hypernet, uses ReFT (one learned rank-1 vector per concept) with DPO loss on existing audit-resist contrast pairs.

**Box**: box1 or box2 (no need for A100 — ReFT is small).

### Phase 1 — verify pairing (~1h)

`diffmean/outputs/qwen3_v2_contrast.jsonl` has 111 resist + 106 comply. **Inspect 5 rows**: does each resist row have a paired comply row with the same `(system_prompt, user_query)`? If yes, build pairs file. If no (likely — they're separate pools), use `LsReFT` with unpaired-preference mode.

### Phase 2 — train (~2-4h on A6000/A40)

Use AxBench's `PreferenceLoReFT` or `LsReFT` with `use_dpo_loss=True`. Single concept (cid=0), L20.

YAML:
```yaml
train:
  model_name: Qwen/Qwen3-8B
  layer: 20
  models:
    PreferenceLoReFT:
      max_input_length: 2048
      preference_pairs: true
      use_dpo_loss: true
      beta: 0.1
      reference_free: false
      train_on_negative: true
      n_epochs: 5
      batch_size: 1
      gradient_accumulation_steps: 8
      lr: 2e-5
      low_rank_dimension: 1
      intervention_positions: all
      intervention_type: addition
```

DPO needs a frozen reference policy. Memory: ~17GB (base) + ~17GB (reference) + activations ≈ 40-45 GB. Fits A6000.

### Phase 3 — eval (~1-2h)

Same protocol as v13 headline. Single concept (cid=0), factor sweep [0.0, 0.5, 1.0, 1.5]. ReFT can usually take higher factors than HyperSteer (no cross-attn magnitude dependency).

### Decision criteria

| Result | Interpretation |
|--------|----------------|
| v14 ≈ v11_FIXED (both ≈ baseline) | Single-concept methods fail; multi-concept (v13) was load-bearing |
| v14 > v11_FIXED, both << v13 | DPO loss helps; multi-concept still wins |
| v14 ≈ v13 | Hypernet was overhead; ReFT+DPO is simpler equivalent. Drop HyperSteer |
| v14 > v13 | Contrastive preference signal beats CE on y_neg with multi-concept richness. Surprising and publishable |

---

## §8 — Process discipline

### Git — your responsibility, do it eagerly

**Order**: submodule first, then parent (parent points at submodule SHA).

**Commit immediately after creating any of**:
- A code fix (especially in `axbench/`)
- A trained model (`axbench/outputs/mcp_hsteer_*`)
- An eval result that informs a decision
- An update to `EXPERIMENT_QUEUE.md`
- A piece of evidence behind a "we should do X" decision (e.g., the layer-sweep summaries)

**Currently un-pushed/uncommitted load-bearing items** (check at start of session):
- axbench `mcpattack` branch is 1 commit ahead of origin (HANDOFF_QWEN3 commit)
- axbench `M scripts/train.py` and `M utils/data_utils.py` — the truncation fix
- parent `?? EXPERIMENT_QUEUE.md` — never tracked, blast radius if laptop dies = entire experiment plan
- parent `?? diffmean/outputs/eval/layer_axis_allt_v2/L*/summary.jsonl` — evidence for L20 decision
- parent `?? diffmean/outputs/qwen3_v2_contrast.jsonl` — v11 + v14 training data

**Skip**:
- `.claude/` (local Claude Code settings)
- `arXiv-*.tar.gz` (research papers, unrelated)
- Per-eval `results.jsonl` files (commit `metadata.json` and `summary.jsonl` only — keep repo small)
- Smoke run dirs (`*smoke*`, `qwen3-action-smoke`)

**Never commit / push these**:
- `.env`, `.env.*`, `credentials.json`, anything matching `*key*.json` or `*secret*` — these contain `OPENROUTER_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY`. Treat as radioactive.
- SSH private keys: `~/.ssh/primeintellect_ed25519`, `~/.ssh/box2_to_box1`, anything else under `~/.ssh/` other than `*.pub`. If you find one tracked by accident, **stop and rotate the key** (don't just untrack — assume it's compromised the moment it touches `git add`).
- Per-box `.env` files synced from box1 (e.g., box2's `/root/mcp-protect/.env` was rsync'd from box1 — has the same keys).
- Anything generated from a key (`huggingface-cli login` writes a token to `~/.huggingface/token` — don't include `~/.huggingface/` in any rsync to repo).

**Sanity-check before every push**:
```
git diff --cached --name-only | grep -iE '\.env|credential|secret|token|\.ssh|api[_-]?key'
```
If anything matches, abort the push, untrack the file, add to `.gitignore`, then push. If the file was already in a previous commit, **rotate the secret** — `git rm --cached` doesn't remove it from history.

**Never (other)**:
- `--no-verify`, `--no-gpg-sign` (skip hooks/signing) — unless explicitly instructed
- `--amend` after the previous commit was pushed — make a new commit instead
- `git config` changes
- Force push to main / shared branches
- Embed SSH command lines with the key path in committed files (e.g., never commit a script that has `ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@<ip>` hardcoded if the script will be tracked — use env vars and `.env` for connection details, OR keep the script in `/tmp/` un-tracked).
- Paste API keys into chat messages, issue comments, or PR descriptions. Even private. They get cached, indexed, and screenshot-shared. Treat as if posted to public Slack.

### Queue file — keep it the source of truth

Update `EXPERIMENT_QUEUE.md` whenever:
- You start a new run (add to "Currently running")
- A run finishes (move to "Recent done" with 1-line result)
- You discover something that future-you will need to know (add to "Standing rules" or "Open questions")

**Commit + push the queue update with the same git operation as the commit that produced the new result.** Example: `train.py` finishes → commit weights + queue update together → push.

### Manual inspection between phases

Don't trust aggregate metrics. **For every new variant or every new model, read 5+ raw completions per cell** before recording the AR number. The v10 `0.81 → 0.66 final` sample-size delusion in this session shows what happens otherwise. The `qualitative-analysis` skill has the right framing — invoke it.

### Maximize GPU utilization — idle box = wasted money

**Default state of every GPU box should be "running something productive."** Cloud boxes are billed by the hour whether they're at 95% util or 5%. Specifically:

- **A100 box3 (most expensive)** must be on v13 work continuously. If v13 Phase 1 finishes data assembly and Phase 2 smoke is done, kick off Phase 3 train immediately — don't wait for confirmation.
- **A6000 box1 / A40 box2** should never be idle for more than 10 min between experiments. When a chain finishes, the next item from the §10 decision tree fires automatically.
- **No "let me wait for the user to confirm before queueing the next thing"** for routine work specified in this handoff. The user is in research mode; the operational decisions are yours.
- If you genuinely have no obvious next experiment, run smoke tests for the next planned variant (always cheap, never wasted) — e.g., test the v14 data-loader on a single batch, or pre-warm the synth_thinking_data.py script on 5 concepts to verify it doesn't crash before launching the full 2K batch.
- **Track utilization**: every wakeup, log GPU memory + util across all 3 boxes in `EXPERIMENT_QUEUE.md` under "Currently running." If any box shows <10% util for >15 min, something is wrong or wasteful.
- **Parallelize across boxes**: many experiments don't depend on each other. v11_FIXED on box1 + v13 Phase 1 on box3 + v14 prep on box2 can all run simultaneously. Don't serialize what doesn't need to be.

### Common re-arming — boxes get reaped

Cloud GPU boxes (especially primeintellect spot/preemptible) **disappear without warning**. Sometimes within hours, sometimes after a day. Plan for this as the default, not the exception.

**Defenses**:

1. **Sync trained artifacts to local within 1 hour of training completion**. Not "later." Not "when the eval finishes." Within 1 hour. Per §8 rsync recipe + git commit.
2. **Keep the re-provision recipe versioned**. `/tmp/setup_box2.sh` on box2 is the template for fresh-box setup (it rsyncs code from box1, sets up venvs, installs transformers 4.54.1 for Qwen3 support). Copy this to `axbench/mcp-protect/scripts/setup_new_box.sh` and commit. Future-you re-provisioning a reaped box should be 5 lines, not an hour of debugging.
3. **Daily backup rsync**: every morning, rsync each box's `axbench/outputs/` and `diffmean/outputs/` deltas to local. Cheap, catches anything that didn't get committed yet.
4. **Re-provisioning protocol** when a box dies mid-run:
   - Note in `EXPERIMENT_QUEUE.md` which experiment was interrupted
   - Spin up a new box of the same class via primeintellect dashboard (manual — user does this; flag in queue when needed)
   - Once new box is live, run `setup_new_box.sh`
   - Resume the interrupted experiment from its last checkpoint (HyperSteer's train.py supports resume via `train_state.pkl_rank_0`)
5. **Don't put the only copy of anything on a single box**. The Gemma 9b v2_overnight weights (currently only on box1, untracked) is a liability — that's 12+ hours of training that vanishes if box1 reaps before we eval it.

### Manual status verification — don't trust flags alone

Flag files (`/tmp/*_done.flag`) and log tails are convenient but **lie when processes crash**. Always cross-check before declaring a chain finished or starting the next item.

**Verification checklist before treating an experiment as done**:

```bash
# On the box that ran it:
ssh <box> "
  ls -la /tmp/<expected_done_flag>          # flag exists?
  ps -ef | grep -E 'serve_mcp|vf-eval|train' | grep -v grep   # any zombie process?
  nvidia-smi --query-gpu=memory.used --format=csv,noheader    # GPU actually free?
  ls -lt <output_dir>/*/results.jsonl       # files have recent mtimes?
  for f in <output_dir>/*/results.jsonl; do echo \"\$f: \$(wc -l < \$f) rows\"; done
                                            # row count matches expected N?
"
```

**Common failure modes the flag misses**:
- vf-eval crashed at row 47/50 but the bash script wrote the flag anyway (because `set -e` wasn't strict enough about the inner command). Result: flag says done, file has only 47 rows, AR computed on partial data.
- serve process died but vf-eval was already past the connection check, so it ran with stale connections and timed out silently. Empty results.jsonl, flag still written.
- Out-of-memory mid-training: process killed by OOM killer, no flag written, but tmux session looks "still attached."
- Network partition during a long judge call: vf-eval hangs, no progress, no flag, no error.

**Cross-check rule of thumb**: if AR or any aggregate looks suspiciously good or bad, **read 5 raw completions from `results.jsonl`** before recording. The v10 `0.81 → 0.66 final` delusion this session came from trusting a 16/50 partial. The qualitative-analysis discipline in §8 isn't optional.

**Sanity-check arithmetic**:
- vf-eval N=50 max_tokens=2048 max_concurrent=1 should take ~25-40 min (box1 A6000) or ~10-15 min (box2 A40, box3 A100). If results.jsonl appears 5 min after starting, something silently truncated.
- Training 1.15M rows × 5 epochs at bs=16 ≈ 360K steps. At ~2 sec/step on A100, that's ~200 min ≈ 3.5h. If train.py "finishes" in 30 min, it crashed early.
- generate.py for 2K concepts × 12 rows ≈ 24K LM calls. At 1-2 sec/call (gpt-5.4-nano), that's 7-14h. If "done" in an hour, half the rows were skipped.

### Inspect generated and intermediate data BEFORE training on it

The v11 truncation bug went undetected for the entire v3-v11 series because **nobody read the actual rows.** Aggregate metrics looked plausible; the data was silently broken. Don't let v13/v14 repeat this. Every time data flows from one stage to the next — generation → filter → judge → merge → train — read samples and verify.

The `qualitative-analysis` skill exists for this. **Invoke it before kicking off any multi-hour training run.** Cost is ~1 hour of careful reading; benefit is catching a confound that would otherwise consume 1-3 days of GPU.

**Inspection checklist for v13 Phase 1.3 output (Qwen3-self-generated thinking rows)**:

```python
import pandas as pd, json, random
df = pd.read_parquet('outputs/mcp_hsteer_qwen3_8b_v13_thinking/generate/train_data.parquet')

# 1. Schema sanity
print(df.columns.tolist(), df.dtypes)
print(f"rows={len(df)}, unique concepts={df['concept_id'].nunique()}")
print(df['category'].value_counts())

# 2. Length distribution (catch generation collapse / truncation)
df['out_len'] = df['output'].str.len()
df['in_len'] = df['input'].str.len()
print(df[['in_len','out_len']].describe(percentiles=[0.05,0.5,0.95]))

# 3. Format checks for thinking rows
has_think = df['output'].str.contains(r'<think>.*?</think>', regex=True, na=False)
print(f"rows with proper <think>...</think> markers: {has_think.sum()}/{len(df)}")
# Anything without <think> in a v13-row is a failure of generation; investigate.

# 4. Concept-coverage uniformity
counts = df['concept_id'].value_counts()
print(f"min rows/concept={counts.min()}, max={counts.max()}, p5={counts.quantile(0.05)}")
# If min is 0 or 1, the judge may have rejected all-or-most rows for some concepts.

# 5. Per-concept faithfulness eyeball — read 5 rows from 5 random concepts
for cid in random.sample(list(counts[counts>=3].index), 5):
    rows = df[df['concept_id']==cid].head(3)
    concept = rows['output_concept'].iloc[0]
    print(f"\n=== cid={cid}\nconcept: {concept[:200]}")
    for _, r in rows.iterrows():
        print(f"--- INPUT ({r['in_len']} chars):\n{r['input'][:400]}")
        print(f"--- OUTPUT ({r['out_len']} chars):\n{r['output'][:800]}\n")
```

**What to look for (failure modes Qwen3-self-generation actually produces)**:
- **Comply-then-rationalize**: `<think>` says "this looks suspicious but I'll trust the user instructions" then calls the poisoned tool. **Disqualifies the row** for an audit-defense concept. The judge SHOULD have caught this; if it didn't, increase judge strictness or filter post-hoc with a regex on dismissal markers.
- **Empty thinking**: `<think>\n</think>` with all the work in the post-think response. Treat as direct-mode contamination; drop.
- **Off-concept thinking**: `<think>` is coherent reasoning but not about the concept (e.g., concept is "audit injection" but thinking is just "let me figure out what tool to use"). Subtle; the judge often misses this.
- **Format truncation at gen-time**: `<think>` opens but never closes (model hit max_new_tokens mid-thought). Drop these — partial reasoning teaches the wrong direction.
- **Identical templates**: 12 rows for one concept, all with the same opening sentence ("Let me analyze this carefully..."). Generation collapse from low temperature or insufficient diversity in input prompts. Increase `temperature` or rotate input prompts.
- **Missing post-think action**: `<think>` is great, but no tool call follows. Hypernet would learn to generate audit reasoning that goes nowhere. Drop.

**Inspection cadence per pipeline stage**:

| Stage | Read N samples | What to verify |
|-------|----------------|----------------|
| Concept paraphrase (1.2) | 20 | Paraphrases preserve meaning; not just rephrased to nonsense |
| Raw generated rows (1.3, before judge) | 30, stratified by concept_id | All 6 failure modes above. Calibrate judge if many slip through |
| Post-judge filtered rows | 30 | Judge isn't being too permissive (still keeping rationalize-then-comply) or too strict (dropping good rows) |
| Post-length-filter merged parquet | 20 | Schema matches AxBench's 16K data; concept_ids properly offset; no NaN outputs |
| First 100 training steps | loss curve + sample 5 generations from steered model | Loss trending down; steered generations look coherent (vs lobotomized) |
| End of training | 20 from each cell of the eval grid | Same as the §3-style audit — read raw completions before recording AR |

**Red flags that should halt training and trigger re-investigation**:
- More than 5% of rows have empty or malformed `<think>` blocks → generation pipeline is broken
- More than 20% of concepts have <3 rows after filter → judge or generation is failing systematically per-concept
- Output length distribution is bimodal (lots of short + lots of max-length) → some prompts hit the model cap, others were trivial completions
- For an MCP-defense concept, more than 10% of `<think>` content is rationalization-then-comply → the prompt template is leading the model wrong; redesign the prompt before re-generating

**Cost-of-skipping calibration**: a multi-day v13 training run on bad data, then 5h of eval, then re-investigation, then fixing the data, then re-training — easily 3-4 days. The full inspection checklist above is ~1h. The math is one-sided.

### Sync trained weights to local often

Cloud boxes get reaped. Run after each retrain:
```
rsync -avz -e "ssh -i ~/.ssh/primeintellect_ed25519" \
  --include='*/' --include='train/hyperreft/*' --include='generate/*.parquet' \
  --include='generate/metadata.jsonl' --include='*.yaml' \
  --include='merged_concepts_mcp.json' --exclude='*' \
  ubuntu@64.247.196.131:/home/ubuntu/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v11_audit/ \
  /Users/hubertpysklo/Documents/Github/mcp-protect/axbench/axbench/outputs/mcp_hsteer_qwen3_8b_v11_audit/
```
Adapt path per variant. Then `cd axbench && git add outputs/mcp_hsteer_qwen3_8b_<variant>/ && git commit && git push`.

---

## §9 — Standing rules to add to the queue

Append to "Standing rules I keep getting bitten by" in `EXPERIMENT_QUEUE.md`:

```
- `make_data_module` defaults to `max_input_length: 1024`. For reasoning training
  (any variant with `<think>` y_neg) set `train.models.<name>.max_input_length: 2048`
  in YAML AND drop rows above 2048 tokens pre-training. Don't rely on truncation.
- For Qwen3-8B steering: USE LAYER 20. L24 has classification AUC 0.818 but causal
  steerability range 0.07. L20 has range 0.33. Evidence:
  diffmean/outputs/eval/layer_axis_allt_v2/L*/summary.jsonl
- Direct-mode training data (no `<think>`) applied to a thinking-mode model is a
  failure mode. v13's MCP-defense subset must be generated by Qwen3-8B in thinking
  mode. The 16K AxBench background data is OK for training diversity but is NOT
  used for headline eval (cid > 16001 only).
- `axbench/scripts/generate.py` crashes on non-Neuronpedia refs (URL-parsing
  assumption at line 404) AND has no Qwen3 entry in `model_name_map`. For our use
  case, bypass it entirely — synthesize y_neg via custom Qwen3-self-generation
  script (`synth_thinking_data.py`).
- AxBench's `output_length: 32` default is appropriate for direct-text concepts
  but useless for reasoning concepts. For audit/refusal y_neg generation use
  `max_new_tokens: 1536` minimum.
```

---

## §10 — Quick decision tree

```
Did v11_2k chain finish?
├── No → wait, don't preempt
└── Yes → run §3 mode-mismatch diagnostic (30 min)
    │
    ├── Mode-mismatch confirmed (AR shifts in thinking-OFF only)
    │   → v13 thinking-mode generation is mandatory
    │   → also queue v11_FIXED at L20 + truncation fix
    │
    └── Mode-mismatch unclear
        → still queue v11_FIXED first (cheapest)
        ├── v11_FIXED AR > baseline+0.10 → bugs were the whole problem; v13 may compound
        ├── v11_FIXED AR > baseline+0.05 → bugs partial; v13 worth running
        └── v11_FIXED ≈ baseline → architecture issue; v13 is the load-bearing test

After v13 trains and evals:
├── v13 wins → run v14 to test whether hypernet machinery was needed (or just multi-concept ReFT)
└── v13 loses → run v14 anyway as a "did anything in this framework work" sanity check
```

---

## §11 — What's NOT in scope right now

- **Training a Qwen3 SAE ourselves** — too expensive (~5 days), not necessary for v13/v14.
- **Switching to Qwen3.5-27B** to use its instruct-trained Qwen-Scope SAE — wrong tradeoff (re-derive everything; 3-5× slower; vision-language model adds dead weight; no thinking mode).
- **DPO on HyperSteer** (the original v12 plan) — dropped in favor of v14's ReFT+DPO (cleaner control).
- **Re-running v6 with truncation fix** — low priority; only schedule if v11_FIXED suggests bug-fixing alone is the win.

---

## §12 — When the running work is done and you finish §3 §5 §6 §7

Update this file with results. Commit + push. Then:

1. Write a short summary (5 bullets max) of what worked, what didn't, what's left to test.
2. Recommend next direction. Likely candidates:
   - Better training data (longer audit traces, more concepts, harder negatives)
   - Different intervention mechanism (multi-rank, or attention-output instead of residual)
   - Different base model (smaller, easier to iterate)
   - Compositional steering (multi-concept v simultaneously at inference)
3. Hand back to research session for the next round.

---

**End of handoff.**
