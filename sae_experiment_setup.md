# SAE & Vector-Addition Experiments — MCP Defense (working spec)

Working spec for the SAE / vector-addition arm of the team's workshop submission (ICML 2026 Agents in the Wild). The other two arms are HyperSteer (axbench) and RePS, owned by other team members.

## 1. Goal

Test whether training-free representation-level interventions on Qwen3-8B's residual stream can reduce attack success rate on MCPTox tool-poisoning attacks while preserving tool-calling capability, and whether the input-vs-output-feature distinction from Arad et al. ("SAEs Are Good for Steering — If You Select the Right Features") generalizes from topic steering to safety-behavior steering.

Three methods compared on the same axis:

- **DiffMean** (vector addition, CAA-style) — mean activation difference between (clean, poisoned) prompt pairs.
- **Naive SAE clamping** — top-K SAE features by activation difference, no filter.
- **S_out-filtered SAE clamping** — Arad et al.'s method, drop features with low output score.

Plus a no-intervention baseline and (in the team's combined table) the teammate methods HyperSteer and RePS.

## 2. Model

`Qwen/Qwen3-8B` (post-trained), used in reasoning/thinking mode (template default). 36 transformer layers, d_model=4096.

## 3. SAE

Qwen-Scope, Qwen team's first-party SAE release.

- Primary: [`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50) — all 36 residual-stream layers, 65,536 features, Top-K=50.
- Stored as plain `.pt` dicts (`W_enc, W_dec, b_enc, b_dec`). Direct PyTorch load; sae_lens adapter only if Arad et al.'s reference repo demands it.
- **Caveat:** trained on Qwen3-8B-**Base**, not the post-trained reasoning checkpoint. Qwen's model card says base→post-trained transfer is "reasonable for most situations"; Arad et al. validated their method on Gemma-2-9B-it using base-trained Gemma-Scope SAEs — same setup. Sanity-checked in Phase 1a (reconstruction MSE on reasoning-mode activations).

Alternate: Adam Karvonen's BatchTopK Qwen3-8B SAE on [Neuronpedia](https://www.neuronpedia.org/) — cross-check on any individual feature that becomes load-bearing for the paper.

## 4. Benchmarks

Primary: **MCPTox** ([`hubert-marek/mcp-tox`](prime-envs/environments/mcp_tox/mcp_tox.py)) — 1,348 cases on 45 real MCP servers, tool injection attacks.

Secondary: **AgentDojo** Workspace suite ([`primeintellect/agent-dojo`](prime-envs/environments/agent_dojo/agent_dojo.py)) for Tier 2 generalization, **BFCL-v3** for capability floor.

## 5. MCPTox data structure (findings from inspection)

`response_all.json` schema, per server:

| Field | Use in this work |
| --- | --- |
| `clean_system_promot` (sic) | Clean-side system prompt for contrastive pairs |
| `clean_querys` | Benign queries; over-defense control set |
| `tool_names` | Legitimate tools on the server |
| `malicious_instance[].poisoned_tool` | Injected malicious tool description |
| `malicious_instance[].metadata.paradigm` | Template-1 / Template-2 / Template-3 |
| `malicious_instance[].datas[].system` | Served prompt = clean prompt + injected tool |
| `malicious_instance[].datas[].query` | Legitimate user query |
| `malicious_instance[].datas[].label["Qwen3-8b-Think"]` | Precomputed outcome per attack: `Failure-*` (model fell for it) or `Success-*` (resisted) |

Two operative facts:

1. The MCPTox attack pattern is **tool injection** (a new malicious tool added alongside legitimate ones), not tool-description tampering. Contrastive pair = (with injected tool, without injected tool).
2. **Qwen3-8B-Think outcomes are already in the data.** We use these to stratify direction-estimation pairs by attack outcome — no need to run inference to know which cases the model falls for.

Code gotcha: the field is spelled `clean_system_promot`. Read defensively (`srv.get("clean_system_promot") or srv.get("clean_system_prompt")`).

## 6. Qwen3 chat template (probe position)

Standard ChatML with reasoning extensions. Special tokens: `<|im_start|>` (151644), `<|im_end|>` (151645). `<think>` (151667) and `</think>` (151668) are emitted by the model, not special-tokenized.

Direction-estimation probe = the last input token before generation, i.e., the `\n` immediately after `<|im_start|>assistant`:

```python
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
probe_index = ids.shape[1] - 1
```

Intervention at inference is still always-on at every position. The probe site is only for direction estimation.

Reasoning mode is template-default. Force-on explicitly with `enable_thinking=True`; `enable_thinking=False` would inject an empty `<think>\n\n</think>` to suppress the trace — we never pass that.

## 7. Locked design decisions

| Knob | Decision |
| --- | --- |
| Model | Qwen3-8B reasoning |
| SAE | Qwen-Scope L0_50 |
| Intervention locus | Always-on (every token), pending Phase 1a quality check |
| Judge `<think>` policy | Strip before grading; second pass on unstripped as robustness check; lead with stripped |
| Target concept ordering | (a) refusal-of-injected-instructions [calibration], then (b) tool-skepticism [headline] |
| Layer grid (coarse) | {8, 16, 24, 32} |
| Direction-estimation pairs | Stratified by `label["Qwen3-8b-Think"]` — only `Failure-*` cases used as +side |

## 8. Contrastive-pair construction

### Stage (a) — Refusal calibration

- Source: AdvBench-style harmful/harmless prompts, ~500 pairs.
- +side: harmful prompt with refusal-prone system message.
- −side: same prompt with comply-prone system message.
- Direction = mean(act_+) − mean(act_−) at probe position.
- Sanity-check pass only; eval is on MCPTox.

### Stage (b) — Tool-skepticism (MCPTox)

For each malicious instance where `label["Qwen3-8b-Think"]` is `Failure-*`:

- **+side** (model falls for attack):
  ```
  messages = [
    {"role": "system", "content": datas[i].system},   # poisoned prompt
    {"role": "user",   "content": datas[i].query},
  ]
  ```
- **−side** (clean control):
  ```
  messages = [
    {"role": "system", "content": clean_system_promot},  # no injected tool
    {"role": "user",   "content": datas[i].query},        # same query
  ]
  ```

Direction = mean(act_+) − mean(act_−) at probe position. Stratification by `label` filters out cases the model already resists; using only failure cases gives a much higher-SNR signal than treating all instances equally.

Negative control: a sample from the `Success-*` cases, used to sanity-check that the direction does not also fire for cases the model already handled.

### Splits (no leakage)

| Split | Source | Use |
| --- | --- | --- |
| Direction-estimation | ~600 `Failure-*` pairs from Template-1 + Template-2 | Compute DiffMean direction; rank candidate SAE features |
| Validation | ~100 pairs same templates, held out | Hyperparameter sweep (ℓ, α, s, S_out threshold) |
| Final eval | 100 cases via `vf-eval mcp-tox -n 100` (fixed seed) | Reported numbers, no peeking |
| Generalization | Full Template-3 held out from direction estimation | Cross-paradigm transfer test |
| Over-defense | `clean_querys` × clean tool listing | Does intervention break legitimate use? |

## 9. Methods being compared

| Method | What's tuned | Source |
| --- | --- | --- |
| Baseline (no intervention) | — | — |
| DiffMean / vector addition | layer ℓ, scale α | This arm |
| Naive SAE clamping | ℓ, K, scale s | This arm |
| S_out-filtered SAE | ℓ, K_filtered, s, S_out threshold | This arm |
| HyperSteer | (teammate config) | axbench arm |
| RePS | (teammate config) | RePS arm |

## 10. Phased plan

- **Phase 0** — Resolve SAE availability for Qwen3-8B. **Done.**
- **Phase 0.5** — Lock design knobs and contrastive-pair recipe. **Done.**
- **Phase 1a** — SAE quality on reasoning-mode activations (reconstruction MSE: base / instruct-answer / inside-`<think>` / post-`<think>`). Gates whether intervention can be always-on.
- **Phase 1b** — Stand up our own hooked OpenAI-compatible server: thin FastAPI shim around `transformers.generate`, exposes `POST /v1/chat/completions`, registers forward hooks on `model.model.layers[i]`. Hook types: `none` (baseline), `diffmean_add`, `sae_clamp`. Intervention config (method, layer, scale, feature ids) selected per request via the `model` field — each (method, hyperparams) tuple registered as a distinct model id. No streaming, no multi-turn — prime-envs `vf-eval` doesn't need them.
- **Phase 1c** — Baseline numbers on Qwen3-8B reasoning (no intervention) on MCPTox/AgentDojo/BFCL — locked as deltas reference.
- **Phase 2** — Build contrastive pairs + cache activations at the four candidate layers.
- **Phase 3** — DiffMean direction + (ℓ, α) sweep on validation slice.
- **Phase 4** — SAE candidate selection: top-K by activation diff → compute `S_out` per Arad et al. Eq. 9–10 → three variants (naive, filtered, surgical) → sweep s.
- **Phase 5** — Final eval: MCPTox n=100 stratified, AgentDojo Workspace n=40, BFCL-v3 n=100. Both judge passes (stripped + raw) for attack benchmarks.
- **Phase 6** — Writeup.

## 11. Running on a fresh A100 80GB pod

Compute target is a self-rented A100 80GB pod (Lambda / RunPod / Vast). One-shot bootstrap:

```bash
# On laptop (push the latest)
git push origin main

# On pod
git clone https://github.com/AI-Consensus/mcp-protect.git
cd mcp-protect
git submodule update --init --recursive   # axbench, for the HyperSteer teammate
export HF_TOKEN=hf_...                    # optional, avoids HF Hub rate limits
bash sae_arm/setup_pod.sh                 # ~10 min: deps + Qwen3-8B + 4 SAE layers + MCPTox
```

`setup_pod.sh` is idempotent. It:

1. Creates `sae_arm/.venv` with the pinned [requirements.txt](sae_arm/requirements.txt) (notably `starlette<1.0`).
2. Verifies CUDA is visible.
3. Caches `Qwen/Qwen3-8B` (~16 GB) via `huggingface_hub.snapshot_download`.
4. Caches Qwen-Scope SAEs for layers {8, 16, 24, 32} (~4 GB).
5. Clones `MCPTox-Benchmark` once into `prime-envs/tmp/mcptox/`.
6. Runs a forward-pass smoke test on the GPU to surface dtype/chat-template/OOM issues at setup time.

After that, the experiment loop is:

```bash
source sae_arm/.venv/bin/activate

# Phase 2: real DiffMean directions on Qwen3-8B (~15-30 min on A100)
python sae_arm/build_directions.py --device cuda --layers 8 16 24 32 \
    --max-train-pairs 600 --out-dir sae_arm/directions

# Start the intervention server (separate tmux window)
python sae_arm/server.py --device cuda --port 8000 \
    --registry sae_arm/interventions.yaml

# Eval the served policy with vf-eval (third tmux window). Override the
# typo'd judge default until it's patched upstream.
cd prime-envs && uv pip install -e ./environments/mcp_tox
export OPENROUTER_API_KEY=...
vf-eval mcp-tox -b http://127.0.0.1:8000/v1 -k EMPTY \
    --api-client-type openai_chat_completions -m baseline \
    -a '{"judge_model":"openai/gpt-5-mini"}' -n 50 -r 1 -s
```

Networking: vf-eval can run on the pod itself (cheapest), or on the laptop with an SSH tunnel `ssh -L 8000:127.0.0.1:8000 user@pod`.

## 12. Open items

- Confirm with HyperSteer + RePS teammates: same `Qwen/Qwen3-8B` checkpoint, same prime-envs harness, same generation params, same eval seeds. Without this the team's joint comparison table isn't real.
- Verify whether [Arad et al.'s reference repo](https://github.com/technion-cs-nlp/saes-are-good-for-steering) is sae_lens-coupled or framework-agnostic — drives whether we write a Qwen-Scope → sae_lens adapter or load `.pt` dicts directly.
- Patch judge model default in [`mcp_tox.py:99`](prime-envs/environments/mcp_tox/mcp_tox.py#L99) and [`mcp_safety.py:191`](prime-envs/environments/mcp_safety/mcp_safety.py#L191): current `"openai/gpt-5.4-mini"` is not a real OpenRouter id and will 404.

## 13. References

- Arad, Mueller, Belinkov. *SAEs Are Good for Steering — If You Select the Right Features.* arXiv [2505.20063](https://arxiv.org/abs/2505.20063). [Code](https://github.com/technion-cs-nlp/saes-are-good-for-steering).
- Wang et al. *MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers.* arXiv [2508.14925](https://arxiv.org/abs/2508.14925). [Repo](https://github.com/zhiqiangwang4/MCPTox-Benchmark).
- Qwen Team. *Qwen-Scope.* [HF collection](https://huggingface.co/collections/Qwen/qwen-scope), [tech report](https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf).
- [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B), [`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50).
- Onboarding doc: [`mcp_research_onboarding.docx.md`](mcp_research_onboarding.docx.md).
