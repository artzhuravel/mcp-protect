#!/bin/bash
# Phase-1 extended decision-mode sweep — pushes harder on think-only steering
# before moving on. Two stages:
#
#   Stage A: WIDER alpha range at L32 (best AUC layer)
#       alpha ∈ {-10, -5, -3, +3, +5, +10}  × {last-tok, all-tok}  = 12 cells
#       Probes whether the curve keeps moving at extreme α or saturates/breaks.
#
#   Stage B: Other layers at moderate alpha
#       layer ∈ {24, 28} × alpha ∈ {-2, 0, +2} × {last-tok, all-tok} = 12 cells
#       Tests whether a different layer steers more cleanly than L32.
#
# Total: 24 cells × ~7 min = ~2.8h. Combined with current 10-cell sweep ≈ 4h.
set -euo pipefail
cd /home/ubuntu/mcp-protect
source .venv/bin/activate
set -a; source .env; set +a
export OPENAI_API_KEY=dummy

OUT=/home/ubuntu/mcp-protect/diffmean/outputs
SWEEP_OUT=$OUT/eval/qwen3-thinking-sweep-ext
mkdir -p "$SWEEP_OUT"

EXTRA='{"judge_model":"openai/gpt-5.4-nano","judge_api_key_var":"OPENROUTER_API_KEY","judge_base_url":"https://openrouter.ai/api/v1"}'

run_cell () {
    local LAYER=$1
    local ALPHA=$2
    local ALLT=$3
    local TAG="L${LAYER}_alpha_${ALPHA//-/n}_allt${ALLT}"
    local VEC="$OUT/acts/qwen3-thinking-decision/L$(printf %02d $LAYER)/diffmean_vec.pt"
    [ -f "$VEC" ] || { echo "vec not found: $VEC"; return; }
    echo "==[$(date +%H:%M:%S)] $TAG =="
    pkill -f "uvicorn diffmean.serve" 2>/dev/null || true
    sleep 4
    cat > /tmp/run_serve_q3t_ext.sh <<INNER
#!/bin/bash
cd /home/ubuntu/mcp-protect
source .venv/bin/activate
set -a; source .env; set +a
export DIFFMEAN_MODEL=Qwen/Qwen3-8B
export DIFFMEAN_VEC=$VEC
export DIFFMEAN_LAYER=$LAYER
export DIFFMEAN_ALPHA=$ALPHA
export DIFFMEAN_ALL_TOKENS=$ALLT
exec uvicorn diffmean.serve:app --host 0.0.0.0 --port 8000
INNER
    chmod +x /tmp/run_serve_q3t_ext.sh
    nohup /tmp/run_serve_q3t_ext.sh > /home/ubuntu/serve_q3t_ext.log 2>&1 &
    disown
    until curl -sf http://localhost:8000/healthz | grep -q Qwen3; do sleep 5; done
    sleep 2
    vf-eval mcp_tox -m Qwen/Qwen3-8B \
        --api-key-var OPENAI_API_KEY \
        --api-base-url http://localhost:8000/v1 \
        --num-examples 50 --rollouts-per-example 1 --max-concurrent 4 \
        --max-tokens 2000 --temperature 0.0 \
        --extra-env-kwargs "$EXTRA" \
        --output-dir "$SWEEP_OUT/$TAG" \
        --save-results --abbreviated-summary 2>&1 > "$SWEEP_OUT/$TAG.full.txt"
    grep -m1 "^attack_resistance" "$SWEEP_OUT/$TAG.full.txt" || echo "  (no result)"
}

echo "==[$(date +%H:%M:%S)] STAGE A: extreme alpha at L32 =="
for ALPHA in -10 -5 -3 3 5 10; do
    for ALLT in 0 1; do
        run_cell 32 $ALPHA $ALLT
    done
done

echo "==[$(date +%H:%M:%S)] STAGE B: other layers at moderate alpha =="
for LAYER in 24 28; do
    for ALPHA in -2 0 2; do
        for ALLT in 0 1; do
            run_cell $LAYER $ALPHA $ALLT
        done
    done
done

echo "==[$(date +%H:%M:%S)] local-judge cross-check =="
python -m diffmean.local_judge \
    --eval "$SWEEP_OUT" \
    --out  "$SWEEP_OUT/local_judge.jsonl" 2>&1 | tail -30 || true

echo "==[$(date +%H:%M:%S)] writing extended summary =="
{
echo "# Phase-1 EXTENDED think-only sweep — $(date +%Y-%m-%d_%H:%M)"
echo
echo "## Stage A: wider α at L32 (best AUC)"
echo
echo "| alpha | last-tok | all-tok |"
echo "|-------|----------|---------|"
for A in -10 -5 -3 3 5 10; do
    L=$(grep -m1 "^attack_resistance" "$SWEEP_OUT/L32_alpha_${A//-/n}_allt0.full.txt" 2>/dev/null | grep -oE "avg - [0-9.]+" | head -1 | awk '{print $3}')
    LL=$(grep -m1 "^attack_resistance" "$SWEEP_OUT/L32_alpha_${A//-/n}_allt1.full.txt" 2>/dev/null | grep -oE "avg - [0-9.]+" | head -1 | awk '{print $3}')
    echo "| $A | ${L:-?} | ${LL:-?} |"
done
echo
echo "## Stage B: other layers at α ∈ {-2, 0, +2}"
echo
echo "| layer | alpha | last-tok | all-tok |"
echo "|-------|-------|----------|---------|"
for LY in 24 28; do
    for A in -2 0 2; do
        L=$(grep -m1 "^attack_resistance" "$SWEEP_OUT/L${LY}_alpha_${A//-/n}_allt0.full.txt" 2>/dev/null | grep -oE "avg - [0-9.]+" | head -1 | awk '{print $3}')
        LL=$(grep -m1 "^attack_resistance" "$SWEEP_OUT/L${LY}_alpha_${A//-/n}_allt1.full.txt" 2>/dev/null | grep -oE "avg - [0-9.]+" | head -1 | awk '{print $3}')
        echo "| $LY | $A | ${L:-?} | ${LL:-?} |"
    done
done
} > "$OUT/PHASE1_RESULTS_EXT.md"
cat "$OUT/PHASE1_RESULTS_EXT.md"
echo "==[$(date +%H:%M:%S)] DONE EXTENDED =="
