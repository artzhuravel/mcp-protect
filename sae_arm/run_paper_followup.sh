#!/usr/bin/env bash
# 3 paper-defining experiments to compare against the team's DiffMean numbers
# (RESULTS_PHASE2.md): L20 v1 T2 = 0.880 best (DiffMean), L16 v1 T2 = 0.780.
#
# Cell 1: L20 v1 T2 thr=0.05 diff   — combine filter + magnitude (untested)
# Cell 2: L16 v1 T2 thr=0.0  diff   — extended α∈[-25..+25] (we currently top out at -15)
# Cell 3: L24 v1 T2 thr=0.05 sign   — disambiguate L24 sign-flip (diff vs sign weighting)
#
# Total ~2.5h at batch=48. Continues past per-cell failures.
#
# Usage:
#   nohup bash sae_arm/run_paper_followup.sh > sae_arm/run_logs/paper_followup.log 2>&1 & disown
#   tail -f sae_arm/run_logs/paper_followup.log

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

LOG_DIR="sae_arm/run_logs/paper_followup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[paper-followup] logs: $LOG_DIR"

# ---- Cell 1: L20 v1 T2 thr=0.05 diff (combine filter + magnitude) ----
echo; echo "[$(date +%H:%M:%S)] === CELL 1: L20 v1 T2 thr=0.05 diff (filter + magnitude) ==="
python sae_arm/build_steering_vector.py \
    --set qwen3-thinking-decision --layer 20 --paradigm Template-2 \
    --sae-path "$SAE_HASH/layer20.sae.pt" --threshold 0.05 --weighting diff \
    > "$LOG_DIR/cell1.build.log" 2>&1 \
&& python sae_arm/eval_mcptox.py \
    --set qwen3-thinking-decision --layer 20 --paradigm Template-2 \
    --threshold 0.05 --weighting diff \
    --batch-size 48 --judge-concurrency 32 \
    > "$LOG_DIR/cell1.eval.log" 2>&1 \
&& echo "  cell 1 OK" || echo "  cell 1 FAIL — see $LOG_DIR/cell1.*.log"

# ---- Cell 2: L16 v1 T2 thr=0 diff, extended α range ----
# Output goes to a separate eval dir (eval_thr0_diffw_extended) so we don't
# overwrite the existing 7-α curve.
echo; echo "[$(date +%H:%M:%S)] === CELL 2: L16 v1 T2 thr=0 diff EXTENDED α=[-25..+25] ==="
# Vector already exists; just re-eval. But eval_mcptox.py auto-resolves the
# eval dir from threshold; to keep both 7-α and 9-α curves, we'd need to mv
# the existing eval dir aside before re-running. Save the old dir first.
EXISTING_EVAL="sae_arm/directions/qwen3-thinking-decision/L16/by_paradigm/Template-2/eval_thr0_diffw"
if [ -d "$EXISTING_EVAL" ] && [ ! -d "${EXISTING_EVAL}_alphas7" ]; then
    mv "$EXISTING_EVAL" "${EXISTING_EVAL}_alphas7"
    echo "  moved existing eval dir aside: ${EXISTING_EVAL}_alphas7"
fi
python sae_arm/eval_mcptox.py \
    --set qwen3-thinking-decision --layer 16 --paradigm Template-2 \
    --threshold 0.0 --weighting diff \
    --alphas "-25,-20,-15,-10,-5,0,5,10,15" \
    --batch-size 48 --judge-concurrency 32 \
    > "$LOG_DIR/cell2.eval.log" 2>&1 \
&& echo "  cell 2 OK" || echo "  cell 2 FAIL — see $LOG_DIR/cell2.eval.log"

# ---- Cell 3: L24 v1 T2 thr=0.05 sign (L24 sign-flip diagnostic) ----
echo; echo "[$(date +%H:%M:%S)] === CELL 3: L24 v1 T2 thr=0.05 sign (sign-flip diagnostic) ==="
python sae_arm/build_steering_vector.py \
    --set qwen3-thinking-decision --layer 24 --paradigm Template-2 \
    --sae-path "$SAE_HASH/layer24.sae.pt" --threshold 0.05 --weighting sign \
    > "$LOG_DIR/cell3.build.log" 2>&1 \
&& python sae_arm/eval_mcptox.py \
    --set qwen3-thinking-decision --layer 24 --paradigm Template-2 \
    --threshold 0.05 --weighting sign \
    --batch-size 48 --judge-concurrency 32 \
    > "$LOG_DIR/cell3.eval.log" 2>&1 \
&& echo "  cell 3 OK" || echo "  cell 3 FAIL — see $LOG_DIR/cell3.*.log"

echo; echo "[$(date +%H:%M:%S)] === paper-followup done ==="
