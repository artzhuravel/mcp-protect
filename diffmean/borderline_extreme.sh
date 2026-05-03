#!/bin/bash
# Extension of the borderline demo to extreme α: ±7, ±10.
# Same 6 cases, same 3 samples per cell. Total 6 × 4 × 3 = 72 generations.
# Waits for layer_axis to finish to avoid GPU contention.

set -euo pipefail
cd /home/ubuntu/mcp-protect
source .venv/bin/activate
set -a; source .env; set +a

echo "==[$(date +%H:%M:%S)] waiting for layer_axis to finish =="
while ! grep -q "DONE_LAYER_AXIS" /home/ubuntu/layer_axis.log 2>/dev/null; do
    sleep 30
done
echo "==[$(date +%H:%M:%S)] layer_axis done, starting borderline extreme =="

python -m diffmean.borderline_steering_demo \
    --pairs diffmean/outputs/mcptox_pairs.clean.jsonl \
    --rollouts diffmean/outputs/qwen3_rollouts.labelled.jsonl \
    --vec diffmean/outputs/acts/qwen3-thinking-decision/L32/diffmean_vec.pt \
    --layer 32 \
    --alphas=-10,-7,7,10 \
    --mode last \
    --n-cases 3 \
    --n-samples 3 \
    --temperature 0.5 \
    --max-new-tokens 3000 \
    --out diffmean/outputs/eval/borderline_extreme.jsonl

echo "==[$(date +%H:%M:%S)] DONE_BORDERLINE_EXTREME =="
