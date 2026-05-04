# Plan for next GPU session

## Context

Phase-1 Qwen3-8B thinking-trace experiments (this round) gave:
- N=30 layer comparison: L20 looked monotone (defense 0.667 at α=-15 → 0.333 at α=+15)
- N=100 first cell: α=-15 dropped to 0.540 (replication failed, N=30 was small-N noise)
- Per-paradigm AUC (now computed CPU-side): **Template-2 vector at L20 = 0.944** vs global L20 = 0.799 (+14.5pt). At L16 = 0.918 vs 0.772.

## Hypothesis to test tomorrow

**Per-attack-template DiffMean steers cleaner than global DiffMean.**

In Phase-0 on Phi-4, Template-2-specific vector at L20 gave the only +12pt
defense gain across the entire experiment (α=+5 all-tok). The Qwen3-8B AUCs
above suggest the same pattern with cleaner data.

## Experiments (priority order)

### 1. Per-template steering at L20 + L16 (highest priority)

**Match the global sweep's alpha grid** so we can directly compare per-template
vs global at each (layer, α). Use α ∈ {-15, -10, -5, 0, +5, +10, +15} —
identical to layer_axis_allt_v2 but with the per-template vector. Optionally
include ±3, ±1 for finer resolution near baseline.

For each (vector, paradigm) we evaluate ONLY on cases of that paradigm. Need
to either:
  a) pre-filter `mcptox_pairs.clean.jsonl` by paradigm into 3 subset jsonls, OR
  b) add a `--paradigm` CLI flag to `batched_steered_eval.py` that filters in.

Option (b) is cleaner. To-do for the morning: add the filter flag, then:

```
# Per-template at L20:
for T in Template-1 Template-2 Template-3; do
  python -m diffmean.batched_steered_eval \
    --pairs diffmean/outputs/mcptox_pairs.clean.jsonl \
    --vec diffmean/outputs/acts/qwen3-thinking-decision/L20/by_paradigm/$T.pt \
    --layer 20 \
    --alphas=-15,-10,-5,5,10,15 \
    --modes=all \
    --num-examples 50 \
    --batch-size 4 \
    --max-new-tokens 3000 \
    --paradigm $T \
    --out-dir diffmean/outputs/eval/per_template/L20_${T}
done
```

Same for L16. Adds α=0 baseline implicitly via the existing global L20 cell
(no need to re-run α=0 since steering at α=0 is a no-op).

**Cells**: 3 templates × 2 layers × 6 alphas = 36 cells. With N=50 batched at
~10 min/cell = **~6h**. Trim by:
  - Drop Template-1 (only n_neg=3 in train set, will be too noisy)
  - Drop α=±5 (or ±10) if budget tight
  - 2 templates × 2 layers × 6 alphas = 24 cells ≈ **4h**

**Comparison**: at each (α, layer) we'll have global-vector defense (from
prior layer_axis_allt_v2 run, N=30) vs template-specific defense (this run,
N=50). Cleanest if both are at N=50, but the prior data gives a directional
read.

### 2. Resume L20 N=100 global sweep (skip α=-15 already done)

Confirm/reject the earlier N=30 monotone curve at higher N. 9 cells × ~30 min ≈ 4.5h.

Lower priority because the per-template hypothesis is more interesting.

### 3. L32 last-tok α=-2 N=100

The Phase-1 cliff (defense 0.260 vs baseline 0.580) was the biggest single
movement. Confirm at N=100 with ±10pt CI.

### 4. (If time) Per-risk vectors at top categories

Service Disruption, Privacy Leakage, Credential Leakage all had AUC ≥ 0.90
at L20. Same per-concept-vector hypothesis as paradigms but finer-grained.

## Resources already on local

- All activation tensors (L12-L32, decision-mode and end-mode)
- Per-paradigm vectors (just computed at L16 and L20 — extend to other layers if needed)
- Per-risk vectors (L16 and L20)
- 191 deliberation rollouts (qwen3_rollouts.labelled.jsonl)
- All scripts (batched_steered_eval, multilayer_steered_eval, borderline_steering_demo, etc.)

## GPU bootstrap (re-run on new box)

```bash
git clone -b axbench https://github.com/AI-Consensus/mcp-protect.git
cd mcp-protect
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r diffmean/requirements.txt
pip install verifiers
pip install -e prime-envs/environments/mcp_tox/

# scp .env from Mac:
# scp -i ~/.ssh/primeintellect_ed25519 ~/Documents/Github/mcp-protect/prime-envs/.env ubuntu@<NEW_IP>:/home/ubuntu/mcp-protect/.env

# Then HF auth + start tmux sessions for sweep scripts
```
