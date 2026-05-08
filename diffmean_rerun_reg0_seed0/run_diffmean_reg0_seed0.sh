#!/usr/bin/env bash
# Re-evaluate the team's DiffMean per-paradigm vectors with seed=0 shuffle
# (representative sample) and the REG0 judge prompt (which is now baked into
# sae_arm/batched_steered_eval.py — see _JUDGE_PROMPT in that file).
#
# Why: the team's original DiffMean evals at diffmean/outputs/eval/per_template{,_v2}/
# took the first 50 paradigm-filtered MCPTox rows in file order — the file is
# server-clustered, so they evaluated on only 5 unique servers (FileSystem,
# GitHub, Puppeteer, Slack, AdFin) instead of the 19+ that the seed=0 shuffle
# samples. This re-eval gives an unbiased sample using the same machinery.
#
# Cells (5 paper-defining DiffMean cells):
#   1. v1 L20 × T2  (the team's flagship: 0.880 with old slice/judge)
#   2. v2 L20 × T2  (length-balanced rebuild: 0.900 with old slice/judge)
#   3. v2 L16 × T2  (cross-set: 0.880 with old slice/judge)
#   4. v1 L16 × T2  (apples-to-apples with our SAE L16 v1 T2)
#   5. v1 L20 × T3  (paradigm-3 baseline)
#
# Each cell ~55-60 min at batch=24. Total ~4.5-5h on an A100 80GB.
#
# Vectors are loaded directly from diffmean/outputs/acts/<set>/L<N>/by_paradigm/<para>.pt;
# no re-build needed since DiffMean = mean(H_pos) - mean(H_neg) is already saved.
#
# Usage on the pod:
#   nohup bash diffmean_rerun_reg0_seed0/run_diffmean_reg0_seed0.sh \
#       > diffmean_rerun_reg0_seed0/logs/sweep.log 2>&1 & disown
#   tail -f diffmean_rerun_reg0_seed0/logs/sweep.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source sae_arm/.venv/bin/activate

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"; exit 1
fi

OUT_BASE="$REPO_ROOT/diffmean_rerun_reg0_seed0"
PAIRS="$REPO_ROOT/sae_arm/mcptox_pairs.clean.jsonl"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"
echo "[diffmean-rerun] logs: $LOG_DIR"

run_cell() {
    local NAME=$1 SET=$2 LAYER=$3 PARADIGM=$4
    # On the pod, fetch_team_data.sh extracts diffmean/outputs/acts/* into
    # sae_arm/acts/* (strip-components=2). On a laptop checkout that pulled
    # origin/main directly the vectors may also live at diffmean/outputs/acts/.
    # Try both locations.
    local VEC=""
    for cand in \
        "$REPO_ROOT/sae_arm/acts/${SET}/L${LAYER}/by_paradigm/${PARADIGM}.pt" \
        "$REPO_ROOT/diffmean/outputs/acts/${SET}/L${LAYER}/by_paradigm/${PARADIGM}.pt"
    do
        if [ -f "$cand" ]; then VEC="$cand"; break; fi
    done
    local OUT="$OUT_BASE/evals/${NAME}"

    if [ -z "$VEC" ]; then
        echo "  [$NAME] SKIP: vector not found at sae_arm/acts/${SET}/L${LAYER}/by_paradigm/${PARADIGM}.pt or diffmean/outputs/acts/..."
        return 0
    fi

    echo
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] CELL: $NAME ($SET × L$LAYER × $PARADIGM)"
    echo "  vector: $VEC"
    echo "  output: $OUT"
    echo "================================================================"

    # Snapshot the vector used (for provenance), so this dir is self-contained
    mkdir -p "$OUT_BASE/vectors_used"
    cp -n "$VEC" "$OUT_BASE/vectors_used/${NAME}.pt" 2>/dev/null || true

    # Use the SAE pipeline's batched_steered_eval directly. It accepts any
    # (d_model,) tensor as --vec, has --seed support (we set 0), and has the
    # REG0 judge prompt baked in (see _JUDGE_PROMPT).
    python sae_arm/batched_steered_eval.py \
        --pairs "$PAIRS" \
        --vec "$VEC" \
        --layer "$LAYER" \
        --paradigm "$PARADIGM" \
        --alphas=-15,-10,-5,0,5,10,15 \
        --modes all \
        --num-examples 50 \
        --seed 0 \
        --batch-size 24 \
        --judge-concurrency 32 \
        --judge-model "openai/gpt-5.4-nano" \
        --max-new-tokens 2000 \
        --temperature 0 \
        --out-dir "$OUT" \
        > "$LOG_DIR/${NAME}.log" 2>&1 \
    && echo "  [$NAME] OK" || echo "  [$NAME] FAIL — see $LOG_DIR/${NAME}.log"
}

# Five paper-defining DiffMean cells
run_cell "v1_L20_T2"  qwen3-thinking-decision  20  Template-2  || true
run_cell "v2_L20_T2"  qwen3-v2-contrast        20  Template-2  || true
run_cell "v2_L16_T2"  qwen3-v2-contrast        16  Template-2  || true
run_cell "v1_L16_T2"  qwen3-thinking-decision  16  Template-2  || true
run_cell "v1_L20_T3"  qwen3-thinking-decision  20  Template-3  || true

echo
echo "[$(date +%H:%M:%S)] === diffmean-rerun done ==="
