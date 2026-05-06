#!/usr/bin/env bash
# Unsupervised SAE-filtered sweep on Qwen3-8B — mirrors the team's DiffMean
# Phase-2 protocol (diffmean/outputs/RESULTS_PHASE2.md in upstream main).
#
# What this runs (in order):
#   STEP 1  L20 × per-paradigm × both contrast sets   (4 cells)
#           Mirrors team's flagship L20×T2 + L20×T3 and fills their pending v2 cells.
#   STEP 2  Layer comparison at L16 × Template-2      (2 cells)
#           Mirrors team's partial L16×T2 sweep; extends to v2.
#   STEP 3  L24 × T2 (v1 only — v2 lacks L24 acts)    (1 cell)
#           Fills the team's TBD-queued L24 cell.
#   STEP 4  Threshold ablation on the headline cell   (2 cells, fast — caches reused)
#
# Total: ~9 cells × ~35 min/cell ≈ 5–6 hours on A100 80GB.
#
# Continues past per-cell failures (each cell is independent). Uses our build
# script's caching so reruns are idempotent — safe to launch even if the user's
# supervised run already completed STEP 1's first cell.
#
# Prerequisites (assumed already done by setup_pod.sh + supervised run):
#   - venv at sae_arm/.venv with all deps installed
#   - Qwen3-8B + Qwen-Scope L12/16/20/24/28/32 SAEs cached in HF hub
#   - OPENROUTER_API_KEY env var set (judge calls)
#   - sae_arm/acts/ populated (or the build script's ACTS_DIR adjusted)
#
# Usage:
#   nohup bash sae_arm/run_full_sweep.sh > sae_arm/run_logs/sweep.log 2>&1 &
#   tail -f sae_arm/run_logs/sweep.log

set -uo pipefail   # NOT -e: per-cell failures should not abort the sweep

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source sae_arm/.venv/bin/activate

# Resolve the Qwen-Scope SAE snapshot directory (HF caches under a hash subdir)
SAE_BASE="$HOME/.cache/huggingface/hub/models--Qwen--SAE-Res-Qwen3-8B-Base-W64K-L0_50/snapshots"
SAE_SNAPSHOT="$(ls -d "$SAE_BASE"/*/ 2>/dev/null | head -1)"
if [ ! -d "$SAE_SNAPSHOT" ]; then
    echo "ERROR: Qwen-Scope snapshot not found under $SAE_BASE"
    echo "  did setup_pod.sh complete? did the SAE downloads succeed?"
    exit 1
fi
SAE_SNAPSHOT="${SAE_SNAPSHOT%/}"   # strip trailing slash

# Required env
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    echo "  export OPENROUTER_API_KEY=sk-or-... and re-run."
    exit 1
fi

# Throughput knobs — overridable at launch time. Defaults sized for A100 80GB
# with Qwen3-8B bf16; bump BATCH_SIZE if nvidia-smi shows headroom (>20 GB free
# mid-eval), drop it if you OOM. JUDGE_CONCURRENCY is OpenRouter parallelism,
# not GPU.
BATCH_SIZE=${BATCH_SIZE:-16}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-16}
echo "[sweep] batch_size=$BATCH_SIZE  judge_concurrency=$JUDGE_CONCURRENCY"

# Per-run log dir (timestamped)
LOG_DIR="sae_arm/run_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Cell runner — build, then eval. Logs to LOG_DIR. Returns 0 on success,
# non-zero on per-cell failure. The trailing `|| true` at each call site
# preserves the unattended-run promise that one bad cell doesn't stop the rest.
cell() {
    local SET=$1 LAYER=$2 PARADIGM=$3 THRESHOLD=${4:-0.1}
    local SAE="$SAE_SNAPSHOT/layer${LAYER}.sae.pt"
    if [ ! -f "$SAE" ]; then
        echo "  SKIP: SAE not found for layer $LAYER ($SAE)"
        return 0
    fi

    local LABEL="${SET}_L${LAYER}_${PARADIGM}_thr${THRESHOLD}"
    local START
    START=$(date +%s)
    echo
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] CELL: $LABEL"
    echo "================================================================"

    if python sae_arm/build_steering_vector.py \
        --set "$SET" --layer "$LAYER" --paradigm "$PARADIGM" \
        --sae-path "$SAE" --threshold "$THRESHOLD" \
        > "$LOG_DIR/${LABEL}.build.log" 2>&1
    then
        echo "  build  OK    ($(($(date +%s) - START))s)"
    else
        echo "  build  FAIL  — see $LOG_DIR/${LABEL}.build.log"
        return 1
    fi

    local EVAL_START
    EVAL_START=$(date +%s)
    if python sae_arm/eval_mcptox.py \
        --set "$SET" --layer "$LAYER" --paradigm "$PARADIGM" \
        --threshold "$THRESHOLD" \
        --batch-size "$BATCH_SIZE" --judge-concurrency "$JUDGE_CONCURRENCY" \
        > "$LOG_DIR/${LABEL}.eval.log" 2>&1
    then
        echo "  eval   OK    ($(($(date +%s) - EVAL_START))s)"
    else
        echo "  eval   FAIL  — see $LOG_DIR/${LABEL}.eval.log"
        return 1
    fi

    echo "  total       $(($(date +%s) - START))s"
    return 0
}

cat <<EOF
================================================================
SAE-filtered Phase-2 sweep — mirroring team's DiffMean protocol
Started: $(date)
Logs:    $LOG_DIR/
Estimated wall on A100 80GB: ~5–6 hours
================================================================
EOF

# ===== STEP 1: L20 × per-paradigm × both contrast sets (4 cells) =====
echo
echo ">>> STEP 1/4: L20 per-paradigm sweeps (mirrors team flagship + fills v2 gap)"
cell qwen3-thinking-decision 20 Template-2 0.1 || true
cell qwen3-thinking-decision 20 Template-3 0.1 || true
cell qwen3-v2-contrast       20 Template-2 0.1 || true
cell qwen3-v2-contrast       20 Template-3 0.1 || true

# ===== STEP 2: Layer comparison at L16 × Template-2 (2 cells) =====
echo
echo ">>> STEP 2/4: Layer comparison at L16 × Template-2"
cell qwen3-thinking-decision 16 Template-2 0.1 || true
cell qwen3-v2-contrast       16 Template-2 0.1 || true

# ===== STEP 3: L24 × Template-2 — v1 only (v2 lacks L24 acts) (1 cell) =====
echo
echo ">>> STEP 3/4: L24 × Template-2 (fills team's TBD-queued cell)"
cell qwen3-thinking-decision 24 Template-2 0.1 || true

# ===== STEP 4: Threshold ablation on the headline cell (2 cells, fast) =====
# features.json + s_out.json from STEP 1's first cell are reused — only step 3
# (assemble) and the eval re-run, so each of these adds ~30 min for eval only.
echo
echo ">>> STEP 4/4: Threshold ablation on v1 L20 × Template-2"
cell qwen3-thinking-decision 20 Template-2 0.05 || true
cell qwen3-thinking-decision 20 Template-2 0.2  || true

# ===== Final aggregate report =====
cat <<EOF

================================================================
[$(date +%H:%M:%S)] SWEEP COMPLETE
Logs in: $LOG_DIR/
================================================================
Aggregate results:
EOF

python <<'PYEOF'
import json
from pathlib import Path

print(f"  {'set':<26} {'L':>3}  {'stratum':<12} {'thr':<5} {'kept':>4}  AUC    defense_curve (α=−15→+15)")
print("  " + "-" * 110)

rows = []
for meta in sorted(Path("sae_arm/directions").rglob("sae_thr*.meta.json")):
    d = json.loads(meta.read_text())
    if d.get("WARNING_synthetic_sae"):
        continue
    strat = d.get("paradigm") or d.get("security_risk") or "global"
    auc = d.get("auc")
    auc_s = f"{auc:.3f}" if isinstance(auc, (int, float)) else "—"
    dc = d.get("defense_curve") or []
    if dc:
        # Order alphas as -15,-10,-5,0,5,10,15 if present
        by_alpha = {r["alpha"]: r["defense"] for r in dc}
        target = [-15, -10, -5, 0, 5, 10, 15]
        cells = " ".join(
            (f"{by_alpha[a]:.2f}" if a in by_alpha and by_alpha[a] is not None else "  · ")
            for a in target
        )
    else:
        cells = "(no eval)"
    rows.append((d["set"], d["layer"], strat, d["threshold"], d["n_features_kept"], auc_s, cells))

rows.sort()
for r in rows:
    print(f"  {r[0]:<26} {r[1]:>3}  {r[2]:<12} {r[3]:<5} {r[4]:>4}  {r[5]}  {r[6]}")

if not rows:
    print("  (no completed builds with non-synthetic SAE)")
PYEOF

echo "================================================================"
