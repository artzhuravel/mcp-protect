#!/usr/bin/env bash
# 4-cell run extending the v1 T2 layer curve to L12/L28/L32 + L20 rescue.
# Acts already exist at all 6 layers for v1 (L12/L16/L20/L24/L28/L32).
#
# Cells:
#   1. L12 v1 T2 thr=0  diff   — early-layer (new data point)
#   2. L28 v1 T2 thr=0  diff   — late-layer (new data point)
#   3. L32 v1 T2 thr=0  diff   — final-layer (new data point)
#   4. L20 v1 T2 thr=0.05 diff — paper-critical L20 rescue (filter+magnitude)
#
# Total ~3h at batch=32. Continues past per-cell failures.
#
# Usage:
#   nohup bash sae_arm/run_layer_extension.sh > sae_arm/run_logs/layer_ext.log 2>&1 & disown
#   tail -f sae_arm/run_logs/layer_ext.log

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

LOG_DIR="sae_arm/run_logs/layer_ext_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[layer-ext] logs: $LOG_DIR"

run_cell() {
    local NAME=$1 LAYER=$2 THRESHOLD=$3
    local SAE="$SAE_HASH/layer${LAYER}.sae.pt"
    local ACTS="$REPO_ROOT/sae_arm/acts/qwen3-thinking-decision/L${LAYER}/H_pos.pt"

    if [ ! -f "$SAE" ]; then
        echo "  [$NAME] SKIP: SAE missing ($SAE)"; return 0
    fi
    if [ ! -f "$ACTS" ]; then
        echo "  [$NAME] SKIP: acts missing ($ACTS)"; return 0
    fi

    echo
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] CELL: $NAME (L${LAYER}, thr=${THRESHOLD}, diff)"
    echo "================================================================"

    python sae_arm/build_steering_vector.py \
        --set qwen3-thinking-decision --layer "$LAYER" --paradigm Template-2 \
        --sae-path "$SAE" --threshold "$THRESHOLD" --weighting diff \
        > "$LOG_DIR/${NAME}.build.log" 2>&1 \
    && python sae_arm/eval_mcptox.py \
        --set qwen3-thinking-decision --layer "$LAYER" --paradigm Template-2 \
        --threshold "$THRESHOLD" --weighting diff \
        --batch-size 32 --judge-concurrency 32 \
        > "$LOG_DIR/${NAME}.eval.log" 2>&1 \
    && echo "  [$NAME] OK" || echo "  [$NAME] FAIL — see $LOG_DIR/${NAME}.*.log"
}

# New-layer cells: thr=0 diff (the recipe that wins at L16)
run_cell "L12_T2"  12 0.0  || true
run_cell "L28_T2"  28 0.0  || true
run_cell "L32_T2"  32 0.0  || true

# L20 rescue: filter + magnitude (untested combination at this cell)
run_cell "L20_T2_filtered"  20 0.05 || true

echo
echo "[$(date +%H:%M:%S)] === layer-extension done ==="
