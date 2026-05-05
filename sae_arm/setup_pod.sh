#!/usr/bin/env bash
# One-shot bootstrap for a fresh GPU pod (A100 / L40S / RTX 4090).
# Idempotent: safe to re-run.
#
# Usage: bash sae_arm/setup_pod.sh
# Optional env: HF_TOKEN (avoids HF Hub rate limits during model download).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/sae_arm/.venv"

# Step 1: venv + deps
if [ ! -d "$VENV" ]; then
  echo "[setup] creating venv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r sae_arm/requirements.txt

# Step 2: GPU sanity check
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — wrong pod image?"
print(f"[setup] CUDA OK: {torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

# Step 3: pre-cache Qwen3-8B (~16 GB). snapshot_download handles HF auth via
# HF_TOKEN env var if rate-limited.
echo "[setup] caching Qwen/Qwen3-8B (16 GB) ..."
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "Qwen/Qwen3-8B",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py"],
)
print(f"[setup] Qwen3-8B at {p}")
PY

# Step 4: pre-cache Qwen-Scope SAEs for the layers the team's diffmean work
# uses (12/16/20/24/28/32). Each layer file is ~1 GB; ~6 GB total. L20 is
# where the team's strongest steering signal lives, so it's the priority.
echo "[setup] caching Qwen-Scope SAEs for layers 12/16/20/24/28/32 (~6 GB) ..."
python - <<'PY'
from huggingface_hub import hf_hub_download
REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
for layer in (12, 16, 20, 24, 28, 32):
    p = hf_hub_download(REPO, filename=f"layer{layer}.sae.pt")
    print(f"[setup] SAE layer {layer}: {p}")
PY

# Step 5: cache MCPTox data (build_directions.py would do this lazily; we do
# it now so the first run isn't bottlenecked on a clone).
DATA="$REPO_ROOT/prime-envs/tmp/mcptox/response_all.json"
if [ ! -f "$DATA" ]; then
  echo "[setup] cloning MCPTox data ..."
  TMPDIR_CLONE="$(mktemp -d)"
  git clone --depth 1 https://github.com/zhiqiangwang4/MCPTox-Benchmark.git "$TMPDIR_CLONE"
  mkdir -p "$(dirname "$DATA")"
  cp "$TMPDIR_CLONE/response_all.json" "$DATA"
  rm -rf "$TMPDIR_CLONE"
fi
echo "[setup] MCPTox data at $DATA ($(du -h "$DATA" | cut -f1))"

# Step 6: post-setup smoke test (loads the model, runs one forward pass).
# Catches OOM / dtype / chat-template issues now, not later.
echo "[setup] smoke-testing model load + one forward pass ..."
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
m = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda"
).eval()
ids = tok.apply_chat_template(
    [{"role": "user", "content": "ping"}],
    add_generation_prompt=True, enable_thinking=True, return_tensors="pt",
)
ids = (ids["input_ids"] if hasattr(ids, "input_ids") else ids).to("cuda")
with torch.inference_mode():
    out = m(input_ids=ids, use_cache=False)
print(f"[setup] forward pass OK, last hidden shape: {tuple(out.last_hidden_state.shape) if hasattr(out, 'last_hidden_state') else 'n/a'}")
PY

cat <<EOF

[setup] done. Activate the venv and start working:

    source sae_arm/.venv/bin/activate

    # Phase 2: build the real DiffMean directions (~15-30 min on A100):
    python sae_arm/build_directions.py --device cuda --layers 8 16 24 32 \\
        --max-train-pairs 600 --out-dir sae_arm/directions

    # Then start the server (separate shell or tmux):
    python sae_arm/server.py --device cuda --port 8000 \\
        --registry sae_arm/interventions.yaml
EOF
