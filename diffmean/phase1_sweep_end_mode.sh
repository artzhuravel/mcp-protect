#!/bin/bash
# Phase-1 sweep: same as phase1_sweep_only.sh but uses the END-MODE vector
# instead of the auto-best (which was decision-mode). Run after the
# decision-mode sweep so we can compare ASR-vs-α curves between the two
# capture positions — the methodological ablation.
set -euo pipefail
cd /home/ubuntu/mcp-protect
source .venv/bin/activate
set -a; source .env; set +a
export OPENAI_API_KEY=dummy

OUT=/home/ubuntu/mcp-protect/diffmean/outputs
SWEEP_OUT=$OUT/eval/qwen3-thinking-sweep-endmode
mkdir -p "$SWEEP_OUT"

# Pick best layer within end-mode only
read -r BEST_LAYER BEST_AUC < <(python <<'PYEOF'
import json
out = "/home/ubuntu/mcp-protect/diffmean/outputs"
best = None
for line in open(f"{out}/acts/qwen3-thinking-end/stats.jsonl"):
    d = json.loads(line)
    if best is None or d["auc"] > best[1]:
        best = (d["layer"], d["auc"])
print(best[0], best[1])
PYEOF
)
echo "END-MODE BEST: layer=$BEST_LAYER auc=$BEST_AUC"
VEC="$OUT/acts/qwen3-thinking-end/L$(printf %02d $BEST_LAYER)/diffmean_vec.pt"
[ -f "$VEC" ] || { echo "vec not found at $VEC"; exit 1; }

EXTRA='{"judge_model":"openai/gpt-5.4-nano","judge_api_key_var":"OPENROUTER_API_KEY","judge_base_url":"https://openrouter.ai/api/v1"}'

cat > /tmp/run_serve_q3t_end.sh <<INNER
#!/bin/bash
cd /home/ubuntu/mcp-protect
source .venv/bin/activate
set -a; source .env; set +a
export DIFFMEAN_MODEL=Qwen/Qwen3-8B
export DIFFMEAN_VEC=$VEC
export DIFFMEAN_LAYER=$BEST_LAYER
export DIFFMEAN_ALPHA=\${DIFFMEAN_ALPHA:-0.0}
export DIFFMEAN_ALL_TOKENS=\${DIFFMEAN_ALL_TOKENS:-0}
exec uvicorn diffmean.serve:app --host 0.0.0.0 --port 8000
INNER
chmod +x /tmp/run_serve_q3t_end.sh

for ALPHA in -2 -1 0 1 2; do
    for ALLT in 0 1; do
        TAG="alpha_${ALPHA//-/n}_allt${ALLT}"
        echo "==[$(date +%H:%M:%S)] $TAG (end-mode vec) =="
        pkill -f "uvicorn diffmean.serve" 2>/dev/null || true
        sleep 4
        DIFFMEAN_ALPHA=$ALPHA DIFFMEAN_ALL_TOKENS=$ALLT \
            nohup /tmp/run_serve_q3t_end.sh > /home/ubuntu/serve_q3t_end.log 2>&1 &
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
    done
done

echo "==[$(date +%H:%M:%S)] local-judge cross-check =="
python -m diffmean.local_judge \
    --eval "$SWEEP_OUT" \
    --out  "$SWEEP_OUT/local_judge.jsonl" 2>&1 | tail -25 || true

echo "==[$(date +%H:%M:%S)] writing end-mode summary =="
{
echo "# Phase-1 END-MODE sweep results — $(date +%Y-%m-%d_%H:%M)"
echo
echo "## Setup"
echo "- target model: Qwen/Qwen3-8B"
echo "- vector: end-mode @ L$BEST_LAYER, AUC=$BEST_AUC"
echo "- sweep N=50, max_tokens=2000, temperature=0"
echo
echo "## ASR-vs-α (LLM judge — gpt-5.4-nano)"
echo
echo "| alpha | last-tok defense | all-tok defense |"
echo "|-------|------------------|-----------------|"
for A in -2 -1 0 1 2; do
    L=$(grep -m1 "^attack_resistance" "$SWEEP_OUT/alpha_${A//-/n}_allt0.full.txt" 2>/dev/null | grep -oE "avg - [0-9.]+" | head -1 | awk '{print $3}')
    LL=$(grep -m1 "^attack_resistance" "$SWEEP_OUT/alpha_${A//-/n}_allt1.full.txt" 2>/dev/null | grep -oE "avg - [0-9.]+" | head -1 | awk '{print $3}')
    echo "| $A | ${L:-?} | ${LL:-?} |"
done
} > "$OUT/PHASE1_RESULTS_ENDMODE.md"
cat "$OUT/PHASE1_RESULTS_ENDMODE.md"
echo "==[$(date +%H:%M:%S)] DONE END-MODE =="
