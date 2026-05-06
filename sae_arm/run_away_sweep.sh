#!/usr/bin/env bash
# 4-cell unattended run. Cell 1 ablates the S_out filter on the headline
# cell (thr=0.0). Cells 2-4 test whether the thr=0.05 win on v1×L20×T2
# generalizes along three axes: layer, contrast set, training pool size.
#
# Each cell ~50 min at batch=32; total ~3.3h.
#
# Usage:
#   nohup bash sae_arm/run_away_sweep.sh > sae_arm/run_logs/away.log 2>&1 &
#   disown
#   tail -f sae_arm/run_logs/away.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source sae_arm/.venv/bin/activate

SAE_DIR="$HOME/.cache/huggingface/hub/models--Qwen--SAE-Res-Qwen3-8B-Base-W64K-L0_50/snapshots"
SAE_HASH="$(ls -d "$SAE_DIR"/*/ 2>/dev/null | head -1)"
SAE_HASH="${SAE_HASH%/}"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"; exit 1
fi

LOG_DIR="sae_arm/run_logs/away_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[away-sweep] logs: $LOG_DIR"

# ---- Cell 1: filter ablation. v1 × L20 × T2, thr=0.0.
# Same cached features.json + s_out.json from the supervised run; only Step 3
# (assemble) re-runs. Tells us whether the S_out filter helps or hurts.
echo; echo "[$(date +%H:%M:%S)] === CELL 1: v1 × L20 × T2 × thr=0.0 (filter ablation) ==="
python sae_arm/build_steering_vector.py \
    --set qwen3-thinking-decision --layer 20 --paradigm Template-2 \
    --sae-path "$SAE_HASH/layer20.sae.pt" --threshold 0.0 \
    > "$LOG_DIR/cell1.build.log" 2>&1 \
    && python sae_arm/eval_mcptox.py \
        --set qwen3-thinking-decision --layer 20 --paradigm Template-2 \
        --threshold 0.0 --batch-size 32 --judge-concurrency 32 \
        > "$LOG_DIR/cell1.eval.log" 2>&1 \
    && echo "  cell 1 OK" || echo "  cell 1 FAIL — see $LOG_DIR/cell1.*.log"

# ---- Cell 2: layer generalization. v1 × L16 × T2 × thr=0.05.
# Different SAE (layer 16). Step 1 + 2 build from scratch (different acts,
# different SAE) so this cell takes the full ~50 min including build.
echo; echo "[$(date +%H:%M:%S)] === CELL 2: v1 × L16 × T2 × thr=0.05 (layer generalization) ==="
python sae_arm/build_steering_vector.py \
    --set qwen3-thinking-decision --layer 16 --paradigm Template-2 \
    --sae-path "$SAE_HASH/layer16.sae.pt" --threshold 0.05 \
    > "$LOG_DIR/cell2.build.log" 2>&1 \
    && python sae_arm/eval_mcptox.py \
        --set qwen3-thinking-decision --layer 16 --paradigm Template-2 \
        --threshold 0.05 --batch-size 32 --judge-concurrency 32 \
        > "$LOG_DIR/cell2.eval.log" 2>&1 \
    && echo "  cell 2 OK" || echo "  cell 2 FAIL — see $LOG_DIR/cell2.*.log"

# ---- Cell 3: contrast-set generalization. v2 × L20 × T2 × thr=0.05.
# v2 contrast pairs are length-balanced (v1 was length-confounded). Cleaner
# harvest may produce a less noisy direction.
echo; echo "[$(date +%H:%M:%S)] === CELL 3: v2 × L20 × T2 × thr=0.05 (length-balanced contrast) ==="
python sae_arm/build_steering_vector.py \
    --set qwen3-v2-contrast --layer 20 --paradigm Template-2 \
    --sae-path "$SAE_HASH/layer20.sae.pt" --threshold 0.05 \
    > "$LOG_DIR/cell3.build.log" 2>&1 \
    && python sae_arm/eval_mcptox.py \
        --set qwen3-v2-contrast --layer 20 --paradigm Template-2 \
        --threshold 0.05 --batch-size 32 --judge-concurrency 32 \
        > "$LOG_DIR/cell3.eval.log" 2>&1 \
    && echo "  cell 3 OK" || echo "  cell 3 FAIL — see $LOG_DIR/cell3.*.log"

# ---- Cell 4: pooled-train, T2-eval. v1 × L20 × thr=0.05.
# Build with no --paradigm flag → uses all ~80 v1 contrast pairs (T1+T2+T3).
# Eval with --paradigm-eval Template-2 → tests on the same 50 T2 prompts as
# your other T2 cells, so defense_curve is directly comparable.
echo; echo "[$(date +%H:%M:%S)] === CELL 4: v1 × L20 × pooled-train × T2-eval × thr=0.05 ==="
python sae_arm/build_steering_vector.py \
    --set qwen3-thinking-decision --layer 20 \
    --sae-path "$SAE_HASH/layer20.sae.pt" --threshold 0.05 \
    > "$LOG_DIR/cell4.build.log" 2>&1 \
    && python sae_arm/eval_mcptox.py \
        --set qwen3-thinking-decision --layer 20 \
        --threshold 0.05 --paradigm-eval Template-2 \
        --batch-size 32 --judge-concurrency 32 \
        > "$LOG_DIR/cell4.eval.log" 2>&1 \
    && echo "  cell 4 OK" || echo "  cell 4 FAIL — see $LOG_DIR/cell4.*.log"

echo; echo "[$(date +%H:%M:%S)] === away-sweep done ==="
