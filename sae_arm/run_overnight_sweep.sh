#!/usr/bin/env bash
# 11-cell unattended sweep at the winning recipe (--threshold 0 --weighting diff)
# across {L16, L20, L24} × {v1, v2} × {T1, T2, T3}, completing the cross with
# what's already on disk (L16 × {v1, v2} × T2).
#
# Each cell ~40 min at batch=48; total ~7.3h. Continues past per-cell failures
# (missing acts, OOM, etc.).
#
# Usage:
#   nohup bash sae_arm/run_overnight_sweep.sh > sae_arm/run_logs/overnight.log 2>&1 &
#   disown
#   tail -f sae_arm/run_logs/overnight.log

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

LOG_DIR="sae_arm/run_logs/overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[overnight] logs: $LOG_DIR"
echo "[overnight] started: $(date)"

# Single-cell runner. Skips gracefully if SAE or acts missing.
run_cell() {
    local NAME=$1 SET=$2 LAYER=$3 PARADIGM=$4
    local SAE="$SAE_HASH/layer${LAYER}.sae.pt"
    local ACTS="$REPO_ROOT/sae_arm/acts/$SET/L${LAYER}/H_pos.pt"

    if [ ! -f "$SAE" ]; then
        echo "  [$NAME] SKIP: SAE not found ($SAE)"; return 0
    fi
    if [ ! -f "$ACTS" ]; then
        echo "  [$NAME] SKIP: acts not found ($ACTS) — set $SET likely lacks L$LAYER"
        return 0
    fi

    echo
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] CELL: $NAME ($SET × L$LAYER × $PARADIGM)"
    echo "================================================================"

    if python sae_arm/build_steering_vector.py \
        --set "$SET" --layer "$LAYER" --paradigm "$PARADIGM" \
        --sae-path "$SAE" --threshold 0.0 --weighting diff \
        > "$LOG_DIR/${NAME}.build.log" 2>&1
    then
        echo "  [$NAME] build OK"
    else
        echo "  [$NAME] build FAIL — see $LOG_DIR/${NAME}.build.log"
        return 1
    fi

    if python sae_arm/eval_mcptox.py \
        --set "$SET" --layer "$LAYER" --paradigm "$PARADIGM" \
        --threshold 0.0 --weighting diff \
        --batch-size 48 --judge-concurrency 32 \
        > "$LOG_DIR/${NAME}.eval.log" 2>&1
    then
        echo "  [$NAME] eval OK"
    else
        echo "  [$NAME] eval FAIL — see $LOG_DIR/${NAME}.eval.log"
        return 1
    fi
}

# ===== Group A: complete the L16 paradigm sweep (winning layer) =====
echo; echo ">>> GROUP A: L16 paradigm extension (T1, T3 across v1, v2)"
run_cell "A1_L16_v1_T1" qwen3-thinking-decision 16 Template-1 || true
run_cell "A2_L16_v1_T3" qwen3-thinking-decision 16 Template-3 || true
run_cell "A3_L16_v2_T1" qwen3-v2-contrast       16 Template-1 || true
run_cell "A4_L16_v2_T3" qwen3-v2-contrast       16 Template-3 || true

# ===== Group B: full 2×3 grid at L20 =====
echo; echo ">>> GROUP B: L20 full grid (v1, v2 × T1, T2, T3)"
run_cell "B1_L20_v1_T1" qwen3-thinking-decision 20 Template-1 || true
run_cell "B2_L20_v1_T2" qwen3-thinking-decision 20 Template-2 || true
run_cell "B3_L20_v1_T3" qwen3-thinking-decision 20 Template-3 || true
run_cell "B4_L20_v2_T1" qwen3-v2-contrast       20 Template-1 || true
run_cell "B5_L20_v2_T2" qwen3-v2-contrast       20 Template-2 || true
run_cell "B6_L20_v2_T3" qwen3-v2-contrast       20 Template-3 || true

# ===== Group C: L24 stretch (v1 only — v2 lacks L24 acts) =====
echo; echo ">>> GROUP C: L24 v1 T2"
run_cell "C1_L24_v1_T2" qwen3-thinking-decision 24 Template-2 || true

echo
echo "[$(date +%H:%M:%S)] === overnight sweep done ==="
echo "Logs: $LOG_DIR"
