#!/bin/bash
# Layer-axis steering sweep: are middle layers cleaner steerers than L32?
#
# L32 has the highest AUC (0.825) but might not be the best steering layer.
# AxBench finds best layers at 30-50% depth (L11-L18 for Qwen3-8B's 36 layers).
# Middle layers L16/L20/L24 had AUCs 0.772/0.799/0.818 — slightly lower as
# probes but possibly better as causal steering directions.
#
# Grid: L ∈ {16, 20, 24} × α ∈ {-10, -5, +5, +10} × mode = all-tok = 12 cells.
# N=50, max_new_tokens=3000 (long-enough thinking traces for reliable judging).
#
# Waits for post_border_pipeline to finish so we don't fight for the GPU.

set -euo pipefail
cd /home/ubuntu/mcp-protect
source .venv/bin/activate
set -a; source .env; set +a

echo "==[$(date +%H:%M:%S)] waiting for post_border pipeline to finish =="
while ! grep -q "==.*\] DONE ==" /home/ubuntu/post_border.log 2>/dev/null; do
    sleep 30
done
echo "==[$(date +%H:%M:%S)] post_border done, starting layer-axis sweep =="

for LAYER in 16 20 24; do
    VEC=/home/ubuntu/mcp-protect/diffmean/outputs/acts/qwen3-thinking-decision/L$(printf %02d $LAYER)/diffmean_vec.pt
    [ -f "$VEC" ] || { echo "skip L$LAYER (vec missing)"; continue; }
    echo "==[$(date +%H:%M:%S)] L$LAYER all-tok sweep =="
    python -m diffmean.batched_steered_eval \
        --pairs diffmean/outputs/mcptox_pairs.clean.jsonl \
        --vec   "$VEC" \
        --layer $LAYER \
        --alphas=-10,-5,5,10 \
        --modes=all \
        --out-dir diffmean/outputs/eval/layer_axis_allt/L$(printf %02d $LAYER) \
        --num-examples 50 \
        --batch-size 4 \
        --max-new-tokens 3000 \
        --temperature 0.0
done

echo "==[$(date +%H:%M:%S)] DONE_LAYER_AXIS =="
