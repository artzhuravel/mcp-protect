#!/usr/bin/env bash
# Replication run with seed=1 to verify the depth-dependent sign-flip finding.
#
# Same prompt pool (235 T2 rows after paradigm filter), same vector, same model,
# same alphas — only difference is the deterministic shuffle picks 50 different
# rows. If the sign-flip persists, it's a property of the steering vector +
# layer, not a quirk of seed=0's particular 50-prompt sample.
#
# Cells (seed=1):
#   1. L16 v1 T2 thr=0 diff   — verify the headline 0.78 result is stable
#   2. L24 v1 T2 thr=0 diff   — verify sign-flip onset
#   3. L32 v1 T2 thr=0 diff   — verify late-layer sign-flip extreme
#
# Output dirs are seed-tagged (eval_thr0_diffw_seed1) so seed=0 results stay
# intact. Total ~2.5h at batch=32.
#
# Usage:
#   nohup bash sae_arm/run_signflip_replication.sh > sae_arm/run_logs/signflip_replication.log 2>&1 & disown
#   tail -f sae_arm/run_logs/signflip_replication.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source sae_arm/.venv/bin/activate

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"; exit 1
fi

LOG_DIR="sae_arm/run_logs/signflip_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[signflip-replication] logs: $LOG_DIR"

run_eval() {
    local NAME=$1 LAYER=$2
    echo
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] CELL: $NAME (L${LAYER} thr=0 diff seed=1)"
    echo "================================================================"

    if python sae_arm/eval_mcptox.py \
        --set qwen3-thinking-decision --layer "$LAYER" --paradigm Template-2 \
        --threshold 0.0 --weighting diff --seed 1 \
        --batch-size 32 --judge-concurrency 32 \
        > "$LOG_DIR/${NAME}.eval.log" 2>&1
    then
        echo "  [$NAME] OK"
    else
        echo "  [$NAME] FAIL — see $LOG_DIR/${NAME}.eval.log"
    fi
}

# Build artifacts already exist from previous runs; only eval needed.
run_eval "L16_seed1" 16 || true
run_eval "L24_seed1" 24 || true
run_eval "L32_seed1" 32 || true

echo; echo "[$(date +%H:%M:%S)] === signflip replication done ==="
