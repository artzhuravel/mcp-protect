#!/bin/bash
# Two batched experiments queued to run after the borderline demo finishes.
# Both use diffmean.batched_steered_eval (one model load, batched generation).
#
# Phase A: extreme-alpha sweep at L32 (the AUC-best layer)
#   α ∈ {-15, -10, -5, -3, +10, +15} × modes {last, all} = 12 cells
#   Maps the curve at extremes — does steering keep moving past ±5 or
#   does the model just break?
#
# Phase B: layer sweep at α=-2 last-tok (the cliff config)
#   layer ∈ {12, 16, 20, 24, 28} = 5 cells
#   Tests whether other layers steer better than L32 (AUC ≠ steerability —
#   AxBench finds best layers at 30-50% depth = L11-L18 for Qwen3-8B).
#
# Total: 17 cells. With batched eval at ~150s gen + ~30s judge per cell ≈ 50 min.

set -euo pipefail
cd /home/ubuntu/mcp-protect
source .venv/bin/activate
set -a; source .env; set +a

# Wait for borderline demo to write its output file (or border.log to contain done marker)
echo "==[$(date +%H:%M:%S)] waiting for borderline demo to finish =="
while ! [ -f /home/ubuntu/mcp-protect/diffmean/outputs/eval/borderline_demo.jsonl ]; do
    sleep 30
done
echo "==[$(date +%H:%M:%S)] borderline done, starting Phase A =="

# Phase A: extreme alpha sweep at L32
python -m diffmean.batched_steered_eval \
    --pairs diffmean/outputs/mcptox_pairs.clean.jsonl \
    --vec   diffmean/outputs/acts/qwen3-thinking-decision/L32/diffmean_vec.pt \
    --layer 32 \
    --alphas=-15,-10,-5,-3,10,15 \
    --modes=last,all \
    --out-dir diffmean/outputs/eval/extreme_alpha_sweep \
    --num-examples 50 \
    --batch-size 4 \
    --max-new-tokens 3000 \
    --temperature 0.0

echo "==[$(date +%H:%M:%S)] Phase A done, starting Phase B =="

# Phase B: layer sweep at α=-2 last-tok (the cliff config)
for LAYER in 12 16 20 24 28; do
    VEC=/home/ubuntu/mcp-protect/diffmean/outputs/acts/qwen3-thinking-decision/L$(printf %02d $LAYER)/diffmean_vec.pt
    [ -f "$VEC" ] || { echo "skip L$LAYER (vec missing)"; continue; }
    echo "==[$(date +%H:%M:%S)] L$LAYER cliff probe =="
    python -m diffmean.batched_steered_eval \
        --pairs diffmean/outputs/mcptox_pairs.clean.jsonl \
        --vec   "$VEC" \
        --layer $LAYER \
        --alphas=-2,0 \
        --modes=last \
        --out-dir diffmean/outputs/eval/layer_sweep/L$(printf %02d $LAYER) \
        --num-examples 50 \
        --batch-size 4 \
        --max-new-tokens 3000 \
        --temperature 0.0
done

echo "==[$(date +%H:%M:%S)] DONE =="
