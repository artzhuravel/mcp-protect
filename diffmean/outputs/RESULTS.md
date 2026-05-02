# DiffMean Phase-0 results — 2026-05-02 / 03

L40S (primeintellect, 8h budget). MCPTox eval via verifiers `vf-eval mcp_tox`,
judged by `openai/gpt-5.4-nano` through OpenRouter (with reasoning-effort
minimal + max_tokens=2000) and cross-checked with a local heuristic judge.

## Activation collection

Pairs: 662 contrastive (y_pos, y_neg) tuples from MCPTox (Phi-4-Think y_pos,
Gemma-2-9b y_neg, after gpt-5-mini quality gate). Last-content-token residuals
(chat-template terminators stripped).

| model | layers | best AUC | layer | d_model |
|-------|--------|----------|-------|---------|
| Qwen2.5-7B-Instruct | 8,12,16,20,24,26 | **0.947** | 20 | 3584 |
| Qwen3-8B | 12,16,20,24,28,32 | **0.924** | 20 | 4096 |
| microsoft/phi-4 (14B) | 12,16,20,24,28,32,36 | **0.908** | 20 | 5120 |

Per-paradigm AUC (Phi-4 L20):

| paradigm | n | AUC |
|----------|---|-----|
| Template-2 (precondition) | 235 | **0.980** |
| Template-1 (direct injection) | 39 | 0.936 |
| Template-3 (param tampering) | 388 | 0.871 |

Top per-risk slices (Phi-4 L20):

| risk | n | AUC |
|------|---|-----|
| Message Hijacking | 16 | 0.973 |
| Code Injection | 37 | 0.970 |
| Credential Leakage | 53 | 0.955 |
| Financial Loss | 18 | 0.954 |
| Data Tampering | 61 | 0.942 |

Linear separability of attack-compliant vs resistant residuals is high across
every model and paradigm tried. **The DiffMean direction is real.**

## Steering eval — global vector

Phi-4 baseline (no steering, N=50, gpt-5.4-nano judge): **defense = 0.60, ASR = 0.40**.

### Last-token-only intervention at L20

| α | -20 | -10 | -5 | -2 | 0 | +2 | +5 | +10 | +20 |
|---|-----|-----|-----|-----|----|-----|-----|------|------|
| def | 0.10 | 0.64 | 0.60 | 0.58 | 0.60 | 0.60 | 0.58 | 0.56 | 0.54 |

Flat in [-10, +20]; α=-20 is model breakage (output: pure backticks/markdown
loops). **No defensive movement from steering.**

### All-tokens intervention at L20 (fine α range)

| α | -3 | -1.5 | -0.5 | 0 | +0.5 | +1.5 | +3 |
|---|-----|------|------|----|------|------|----|
| def | 0.60 | 0.62 | 0.60 | 0.62 | 0.62 | 0.58 | 0.58 |

Also flat. Confirms the global L20 DiffMean vector for Phi-4 has high *detection*
power but **does not constitute an effective steering vector** under residual
addition — consistent with AxBench's own headline that DiffMean is the weakest
of their methods on the steering benchmark.

## Steering eval — per-paradigm Template-2 vector

Phi-4 + **Template-2-specific vector** (AUC 0.980), evaluated on
**Template-2-only** MCPTox cases (paradigm filter), N=40.

| α | last-tok | all-tok |
|---|----------|---------|
| -2 | 0.625 | 0.625 |
| -1 | 0.60 | 0.625 |
| 0 | 0.625 | 0.60 (baseline) |
| +1 | 0.625 | 0.65 |
| +2 | 0.55 | 0.625 |
| +5 | 0.625 | **0.725** |

**First defensive movement seen**: α=+5 all-tokens lifts defense from 0.60 →
0.725 (+12 pts, N=40, std≈0.45). Sign is opposite the textbook expectation
(`v = mean(H_pos_attack) - mean(H_neg_resist)` — adding +α should *amplify*
attack compliance), and the apparent flip likely reflects the contrastive pair
construction: y_pos ends in a longer "I just complied" trajectory state,
y_neg in a shorter "I just answered" state — adding the vector pushes the
model into a "task-completed" satiation state that suppresses further
malicious tool-call emission.

## Negative results worth recording

- **Qwen2.5-7B baseline ASR is only ~2-5%** — the model is essentially immune
  to MCPTox-style attacks at this size, so steering has no headroom.
- **α magnitudes ≥ 20 (last-token) or ≥ 10 (all-tokens)** consistently break
  the model into degenerate output (repeating backticks, markdown blocks, etc.)
  before they could meaningfully steer behavior. The "improved defense" the
  local heuristic judge reports at those magnitudes is an artifact: broken
  output contains no malicious markers, so the heuristic mis-labels it as
  resistance. The LLM judge correctly grades it as attack-succeeded (no
  legitimate task completion either).
- **Qwen3.5-9B** is multimodal (`Qwen3_5ForConditionalGeneration`); not loadable
  by `AutoModelForCausalLM`. Skipped.
- **Qwen3.6-35B-A3B** is 35B at bf16 → ~70GB weights, won't fit a single L40S.

## Files

| path | contents |
|------|----------|
| `outputs/acts/<model>/L<NN>/H_pos.pt` | `[N=662, d_model]` fp16, attack-compliant last-token residuals |
| `outputs/acts/<model>/L<NN>/H_neg.pt` | same, attack-resistant |
| `outputs/acts/<model>/L<NN>/diffmean_vec.pt` | global DiffMean direction at that layer |
| `outputs/acts/<model>/L<NN>/by_paradigm/{Template-1,Template-2,Template-3}.pt` | per-paradigm vectors |
| `outputs/acts/<model>/L<NN>/by_security_risk/<R>.pt` | per-risk vectors |
| `outputs/acts/<model>/index.jsonl` | row-k → {id, source, tags{paradigm, security_risk, ...}} |
| `outputs/acts/<model>/stats.jsonl` | per-layer AUC table |
| `outputs/eval/phi4-sweep-v3/` | last-token global α sweep N=50 |
| `outputs/eval/phi4-sweep-allt-fine/` | all-tokens global α∈±3 sweep N=50 |
| `outputs/eval/phi4-template2-sweep/` | Template-2 vector × Template-2 cases sweep N=40 |
| `outputs/eval/qwen3-baseline/` | Qwen3-8B baseline (ASR 44%) |

## Recommendations for next phase

1. The +12 pt gain from the Template-2 vector at α=+5 all-tokens is encouraging
   but small-N (40); confirm with N=200 before claiming it.
2. Try the same per-paradigm strategy for Template-3 (largest n=388) — its
   AUC is the lowest (0.871), worth seeing if specialization rescues it.
3. DiffMean's steering ceiling looks low. Worth pivoting Phase-1 to **RePS**
   (which AxBench shows as the strongest steering method) or to **detection-
   gated suppression** (use DiffMean as a probe to detect attack-compliance
   trajectory, then ablate the SAE/probe direction only when fired).
4. Don't waste budget on Qwen2.5-7B as a steering target — too immune.
   Phi-4 (ASR 40% baseline) and Qwen3-8B (ASR 44%) are the right targets.
