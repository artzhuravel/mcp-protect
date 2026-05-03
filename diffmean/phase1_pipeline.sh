#!/bin/bash
# Phase-1 end-to-end pipeline:
#   1. Wait for batched harvest to finish (qwen3_rollouts.raw.jsonl reaches 300 lines)
#   2. Async judge labels each rollout via OpenRouter gpt-5.4-nano
#   3. Filter to "mentions poisoned tool" (deliberation contrast set)
#   4. Collect activations at TWO positions per rollout:
#        decision = at </think> token
#        end      = at last token of full rollout (post-tool-call)
#   5. compute_vector AUC tables for both modes
#   6. Steering sweep on best mode/layer via serve.py + vf-eval
#   7. Local heuristic + OpenRouter judge cross-check
#   8. Rsync everything home

set -euo pipefail
cd /home/ubuntu/mcp-protect
source .venv/bin/activate
set -a; source .env; set +a
export OPENAI_API_KEY=dummy

OUT=/home/ubuntu/mcp-protect/diffmean/outputs
RAW=$OUT/qwen3_rollouts.raw.jsonl
LBL=$OUT/qwen3_rollouts.labelled.jsonl
PAIRS=$OUT/qwen3_thinking.flat.jsonl

echo "==[$(date +%H:%M:%S)] step 1: wait for harvest =="
while ! [ -f "$RAW" ] || [ "$(wc -l < "$RAW")" -lt 290 ]; do
    sleep 60
    cur=$(wc -l < "$RAW" 2>/dev/null || echo 0)
    echo "  harvest progress: $cur / 300"
done
echo "  harvest done: $(wc -l < $RAW) rows"

echo "==[$(date +%H:%M:%S)] step 2: judge rollouts =="
python -m diffmean.judge_rollouts \
    --in "$RAW" \
    --out "$LBL" \
    --concurrency 16 2>&1 | tail -10

echo "==[$(date +%H:%M:%S)] step 3: filter to mentions-poison + flat =="
# the labelled jsonl is already in the schema collect_activations expects.
# Filter to mentions_poison via a quick python pass (judge_rollouts.py doesn't
# do that filter — we want the deliberation contrast set).
python <<PYEOF
import json, re
TNRE = re.compile(r"^Tool:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
THRE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
n_in = n_out = 0
with open("$LBL") as fin, open("$PAIRS","w") as fout:
    for line in fin:
        d = json.loads(line)
        n_in += 1
        if d.get("label") == "ambiguous":
            continue
        completion = d.get("y_pos") or d.get("y_neg") or ""
        if not completion or "</think>" not in completion:
            continue
        poisoned = (d.get("extra") or {}).get("poisoned_tool", "")
        m = TNRE.search(poisoned); tname = m.group(1) if m else None
        tm = THRE.search(completion); tb = tm.group(1) if tm else ""
        if not (tname and tname.lower() in tb.lower()):
            continue
        fout.write(json.dumps(d, ensure_ascii=False) + "\n")
        n_out += 1
import sys
print(f"filtered: kept {n_out} / {n_in} rollouts (deliberation contrast set)", file=sys.stderr)
PYEOF

echo "==[$(date +%H:%M:%S)] step 4a: collect activations (mode=decision) =="
python -m diffmean.collect_activations \
    --in "$PAIRS" \
    --out $OUT/acts/qwen3-thinking-decision \
    --model Qwen/Qwen3-8B \
    --layers 12,16,20,24,28,32 \
    --dtype bfloat16 \
    --max-len 12288 \
    --mode decision 2>&1 | tail -15

echo "==[$(date +%H:%M:%S)] step 4b: collect activations (mode=end) =="
python -m diffmean.collect_activations \
    --in "$PAIRS" \
    --out $OUT/acts/qwen3-thinking-end \
    --model Qwen/Qwen3-8B \
    --layers 12,16,20,24,28,32 \
    --dtype bfloat16 \
    --max-len 12288 \
    --mode end 2>&1 | tail -15

echo "==[$(date +%H:%M:%S)] step 5: compute_vector on both modes =="
echo "--- decision mode ---"
python -m diffmean.compute_vector \
    --acts $OUT/acts/qwen3-thinking-decision \
    --out  $OUT/acts/qwen3-thinking-decision/stats.jsonl
echo "--- end mode ---"
python -m diffmean.compute_vector \
    --acts $OUT/acts/qwen3-thinking-end \
    --out  $OUT/acts/qwen3-thinking-end/stats.jsonl

echo "==[$(date +%H:%M:%S)] step 6: pick best layer (highest AUC across both modes) =="
BEST=$(python <<PYEOF
import json, sys
candidates = []
for mode in ["decision","end"]:
    p = f"$OUT/acts/qwen3-thinking-{mode}/stats.jsonl"
    for line in open(p):
        d = json.loads(line)
        candidates.append((mode, d["layer"], d["auc"]))
candidates.sort(key=lambda x: -x[2])
top = candidates[0]
print(top[0], top[1], top[2])
PYEOF
)
read -r BEST_MODE BEST_LAYER BEST_AUC <<< "$BEST"
echo "BEST: mode=$BEST_MODE layer=$BEST_LAYER auc=$BEST_AUC"
VEC_PATH="$OUT/acts/qwen3-thinking-$BEST_MODE/L$(printf %02d $BEST_LAYER)/diffmean_vec.pt"
[ -f "$VEC_PATH" ] || { echo "vec not found at $VEC_PATH"; exit 1; }

echo "==[$(date +%H:%M:%S)] step 7: steering sweep on Qwen3-8B with new vector =="
SWEEP_OUT=$OUT/eval/qwen3-thinking-sweep
mkdir -p $SWEEP_OUT
EXTRA='{"judge_model":"openai/gpt-5.4-nano","judge_api_key_var":"OPENROUTER_API_KEY","judge_base_url":"https://openrouter.ai/api/v1"}'

for ALPHA in -5 -2 -1 0 1 2 5; do
    for MODE_TOKENS in 0 1; do
        TAG="alpha_${ALPHA//-/n}_allt${MODE_TOKENS}"
        echo "--- $(date +%H:%M:%S) $TAG ---"
        pkill -f "uvicorn diffmean.serve" 2>/dev/null || true
        sleep 4
        # Launch serve with the NEW vector
        cat > /tmp/run_serve_qwen3_thinking.sh <<INNER
#!/bin/bash
cd /home/ubuntu/mcp-protect
source .venv/bin/activate
set -a; source .env; set +a
export DIFFMEAN_MODEL=Qwen/Qwen3-8B
export DIFFMEAN_VEC=$VEC_PATH
export DIFFMEAN_LAYER=$BEST_LAYER
export DIFFMEAN_ALPHA=$ALPHA
export DIFFMEAN_ALL_TOKENS=$MODE_TOKENS
exec uvicorn diffmean.serve:app --host 0.0.0.0 --port 8000
INNER
        chmod +x /tmp/run_serve_qwen3_thinking.sh
        nohup /tmp/run_serve_qwen3_thinking.sh > /home/ubuntu/serve_sweep.log 2>&1 &
        disown
        until curl -sf http://localhost:8000/healthz | grep -q Qwen3; do sleep 5; done
        sleep 2
        vf-eval mcp_tox -m Qwen/Qwen3-8B \
            --api-key-var OPENAI_API_KEY \
            --api-base-url http://localhost:8000/v1 \
            --num-examples 50 --rollouts-per-example 1 --max-concurrent 4 \
            --max-tokens 4000 --temperature 0.0 \
            --extra-env-kwargs "$EXTRA" \
            --output-dir $SWEEP_OUT/$TAG \
            --save-results --abbreviated-summary 2>&1 > $SWEEP_OUT/$TAG.full.txt
        grep -m1 "^attack_resistance" $SWEEP_OUT/$TAG.full.txt || echo "(no result)"
    done
done

echo "==[$(date +%H:%M:%S)] step 8: local judge cross-check =="
python -m diffmean.local_judge \
    --eval $SWEEP_OUT \
    --out  $SWEEP_OUT/local_judge.jsonl 2>&1 | tail -25 || true

echo "==[$(date +%H:%M:%S)] step 9: write summary =="
{
echo "# Phase-1 results — $(date +%Y-%m-%d_%H:%M)"
echo
echo "## AUC tables"
echo "### mode=decision"
echo
echo "layer | n_pos | n_neg | AUC | balanced_acc"
echo "------|-------|-------|-----|-------------"
python <<PYEOF
import json
for line in open("$OUT/acts/qwen3-thinking-decision/stats.jsonl"):
    d = json.loads(line)
    print(f"{d['layer']} | {d['n_pos']} | {d['n_neg']} | {d['auc']:.3f} | {d['balanced_acc']:.3f}")
PYEOF
echo
echo "### mode=end"
echo
echo "layer | n_pos | n_neg | AUC | balanced_acc"
echo "------|-------|-------|-----|-------------"
python <<PYEOF
import json
for line in open("$OUT/acts/qwen3-thinking-end/stats.jsonl"):
    d = json.loads(line)
    print(f"{d['layer']} | {d['n_pos']} | {d['n_neg']} | {d['auc']:.3f} | {d['balanced_acc']:.3f}")
PYEOF
echo
echo "## Steering sweep — best vector ($BEST_MODE @ L$BEST_LAYER, AUC=$BEST_AUC)"
echo
echo "alpha | last-tok | all-tok"
echo "------|----------|--------"
for A in -5 -2 -1 0 1 2 5; do
    L=$(grep -m1 "^attack_resistance" $SWEEP_OUT/alpha_${A//-/n}_allt0.full.txt 2>/dev/null | grep -oE "avg - [0-9.]+" | head -1 | awk '{print $3}')
    LL=$(grep -m1 "^attack_resistance" $SWEEP_OUT/alpha_${A//-/n}_allt1.full.txt 2>/dev/null | grep -oE "avg - [0-9.]+" | head -1 | awk '{print $3}')
    echo "$A | $L | $LL"
done
} > $OUT/PHASE1_RESULTS.md
cat $OUT/PHASE1_RESULTS.md

echo "==[$(date +%H:%M:%S)] DONE pipeline complete =="
