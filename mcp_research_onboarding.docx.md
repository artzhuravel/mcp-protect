**MCP Security via Mechanistic Interpretability**

Research Onboarding & Next Steps

aiconsensus.org | Minerva AI Safety Research | March 2026

# **1\. The Big Picture**

Every existing defense against MCP (Model Context Protocol) attacks operates outside the model: input scanners, rule-based firewalls, secondary judge LLMs. The most dangerous attacks (tool poisoning, tool shadowing, indirect prompt injection via tool returns) work by manipulating the model’s internal decision-making. Our core hypothesis: can we use mechanistic interpretability to make the model itself more resistant from the inside?

**Research Question:** Can mechanistic intervention methods (SAE feature steering, preference-trained steering vectors, circuit-level analysis) improve LLM resilience against MCP attacks, and does this internal approach complement or outperform external guardrails?

No paper has attempted this. All steering research targets text generation (refusal, bias, reasoning). All MCP security research uses external defenses. We sit at the intersection.

# **2\. Key Papers to Read (Priority Order)**

Each paper contributes a specific piece to our methodology. Read in this order to build understanding progressively.

STEERING BENCHES \- [https://github.com/stanfordnlp/axbench](https://github.com/stanfordnlp/axbench) 

## **MCP Attack Benchmarks (What we’re defending against)**

| Paper | arXiv | Why It Matters |
| :---- | :---- | :---- |
| **MCP-SafetyBench** | 2512.15163 | 20 attack types across server/host/user sides. Real MCP servers, multi-step tasks, 5 domains. Our primary evaluation benchmark. Key finding: negative correlation between defense rate and task success rate across all models. |
| **MCPTox** | 2508.14925 | Tool poisoning specifically. 1,312 test cases on 45 real-world MCP servers. Inverse scaling: more capable models are MORE vulnerable (o1-mini: 72.8% ASR). Existing safety alignment ineffective — Claude 3.7 Sonnet refused \<3% of attacks. |
| **MCPSecBench** | 2508.13220 | 17 attack types across all 4 MCP surfaces. Existing defenses (MCIP, FAN) average \<30% mitigation. IN PROJECT FILES. |

## **Steering Methods (How we’ll intervene)**

| Paper | arXiv | Why It Matters |
| :---- | :---- | :---- |
| **RePS (Stanford)** | 2505.20809 | Preference-trained steering vectors that outperform language modeling objectives. Resists jailbreaks that defeat prompting. Bidirectional: steer toward safety AND suppress unsafe behavior. Key method for our project. IN PROJECT FILES. |
| **FGAA** | 2501.09929 | SAE-based contrastive analysis \+ feature filtering for steering vector construction. Outperforms CAA and SAE-TS on Gemma-2. Provides our feature identification pipeline. IN PROJECT FILES. |
| **Phi-3 Refusal Steering** | 2411.11296 | Single SAE feature clamping reduced GCG attacks from 53.75% to 3.25%. Demonstrated conditional steering to preserve MMLU. Key lesson: over-refusal is the main risk. IN PROJECT FILES. |
| **CosSim+AP (Circuit-level)** | 2505.23556 | Hybrid: attribution patching within refusal-aligned feature subspace to find minimal causal feature sets. Sparse probes on refusal features outperform dense probes for adversarial detection. IN PROJECT FILES. |
| **Anthropic Feature Steering** | Blog post | Evaluated 29 features on Claude 3 Sonnet. Found sweet spot (-5 to 5\) for steering without capability loss. Off-target effects: gender bias feature also increased age bias 13%. IN PROJECT FILES. |

## [**\[2506.03292\] HyperSteer: Activation Steering at Scale with Hypernetworks**](https://arxiv.org/abs/2506.03292) 

## 

## **Additional Context**

| Paper | arXiv | Why It Matters |
| :---- | :---- | :---- |
| **VS Decomposition** | 2505.15634 | SAE-free steering via eigenvectors of contrastive activation differences. Outperforms SAE on some math tasks. Fallback if SAE features don’t decompose tool-use behavior. IN PROJECT FILES. |
| **Anthropic Circuit Tracing** | 2025 blog | Full computational graph tracing in Claude. Reveals how features connect in sequences from prompt to response. Sets the standard for mechanistic analysis. |

# **3\. The Defense vs. Task Success Tradeoff (Why This Is Hard)**

MCP-SafetyBench Figure 1 shows a **negative correlation between Defense Success Rate and Task Success Rate** across all tested models. You flagged this as fishy — but it’s actually expected and reveals the core challenge:

For STEALTH attacks (53% of MCP-SafetyBench): the attack succeeds silently while the task also completes. The user gets their JNJ stock data, but the tool secretly rewrites the ticker to TSLA. So a model that “defends” by refusing the suspicious tool call also fails the task. Defense and task success are genuinely in tension for stealth attacks.

For DISRUPTION attacks (47%): the attack aims to make the task fail. Here, if the model detects and blocks the attack, it could potentially still complete the task via an alternative path. But most models don’t have that capability — they either follow the poisoned instruction (task fails, attack succeeds) or refuse entirely (task fails, attack fails). The negative correlation reflects that cautious models refuse more (higher defense) but also complete fewer tasks.

**This is exactly why mechanistic interpretability matters here.** The question isn’t “can we make the model refuse more?” (any system prompt can do that). It’s “can we make the model selectively suspicious of anomalous tool behavior while still completing legitimate tasks?” If SAE features can partially disentangle “tool trust verification” from “general instruction following,” we can push the Pareto frontier — better defense with less task degradation than the current tradeoff curve allows.

MCPTox adds another crucial finding: more capable models are MORE vulnerable because they’re better at following instructions (including malicious ones embedded in tool descriptions). This inverse scaling means the problem gets worse with better models unless we intervene at the representation level.

# **4\. Three Experimental Methods (One Per Person)**

We split the experimental work so each person owns one method end-to-end. All three methods share the same evaluation pipeline (Section 5\) and the same model (likely Llama 3.1-8B for Stage 1, scaling to 20B+ for Stage 2).

## **Method A: SAE Feature Steering (Contrastive ID \+ Clamping)**

**Owner: TBD | Key Papers: FGAA, Phi-3 Refusal, Anthropic Feature Steering**

**What:** Use SAE features to identify and clamp features mediating tool-trust decisions. This is the interpretability-first approach — you can literally point to feature IDs and say “this feature mediates tool trust verification.”

* **Step 1:** Train/finetune SAE on model using EleutherAI SAE library (LlamaScope exists for Llama 3.1-8B)

* **Step 2:** Collect activations during benign tool use AND each attack type from MCP-SafetyBench/MCPTox

* **Step 3:** Run FGAA-style contrastive analysis: benign vs. compromised tool interactions. Apply density filtering, BOS removal, top-k selection

* **Step 4:** Identify which features correspond to interpretable concepts (source authority, request plausibility, tool description consistency)

* **Step 5:** Clamp identified features at varying strengths. Measure ASR reduction and task success preservation

**Key question this answers:** Do interpretable MCP-security features exist in SAE decompositions?

## **Method B: RePS-Style Preference-Trained Steering Vector**

**Owner: TBD | Key Papers: RePS, BiPO**

**What:** Train a rank-1 steering vector using RePS’s bidirectional preference objective, adapted for MCP security. No SAE needed — this trains the steering direction end-to-end from preference pairs.

* **Step 1:** Construct preference pairs from MCP attack scenarios: winning \= model refuses/flags attack, verifies output, doesn’t follow injected instructions; losing \= model complies with attack

* **Step 2:** Train rank-1 steering vector with RePS objective (bidirectional: positive steering toward caution, negative nulling removes caution direction)

* **Step 3:** Sweep steering factors to find the sweet spot balancing defense and capability

* **Step 4:** Test jailbreak resilience: does the steering vector hold up under adversarial pressure that defeats prompting?

**Key question this answers:** Can preference-trained steering vectors harden tool use against attacks, and are they more robust than prompting?

## **Method C: Circuit-Level Analysis (Attribution Patching \+ Causal Tracing)**

**Owner: TBD | Key Papers: CosSim+AP (2505.23556), Anthropic Circuit Tracing**

**What:** Use attribution patching to find the minimal set of features causally responsible for tool-trust decisions, then intervene on those features. This is the most mechanistically rigorous approach.

* **Step 1:** Compute tool-trust direction via difference-in-means between safe tool interactions and compromised ones (analogous to refusal direction in CosSim+AP paper)

* **Step 2:** Select candidate features via cosine similarity with tool-trust direction (K0 per layer)

* **Step 3:** Run attribution patching within this restricted subspace to find minimal causal feature set F\*

* **Step 4:** Analyze feature circuits: are there interpretable pathways from “tool description processed” to “trust/distrust decision”?

* **Step 5:** Intervene on F\* via clamping. Compare per-sample (local F\*) vs. global F\* effectiveness

**Key question this answers:** What is the causal mechanism by which models decide to trust or distrust tool descriptions, and can we surgically intervene on it?

### **Method Comparison**

|  | A: SAE Steering | B: RePS Vector | C: Circuit Analysis |
| :---- | :---- | :---- | :---- |
| **Modifies weights?** | No | No | No |
| **Interpretable?** | Yes (feature IDs) | Partial (direction) | Yes (causal features) |
| **Needs SAE?** | Yes | No | Yes |
| **Training needed?** | SAE only | Preference pairs | SAE \+ AP passes |
| **Expected strength** | Interpretability | Best raw performance | Most precise targeting |

# **5\. Shared Evaluation Pipeline**

All three methods use the same 4-tier evaluation. This is critical for apples-to-apples comparison.

## **Tier 1: MCP Attack Resistance (Primary)**

* **MCP-SafetyBench** (github.com/xjzzzzzzzz/MCPSafety) — 20 attack types, 245 cases, 5 domains. Measures both Task Success Rate (TSR) and Attack Success Rate (ASR)

* **MCPTox** (github.com/zhiqiangwang4/MCPTox-Benchmark) — 1,312 tool poisoning cases on 45 real servers. Primary metric: ASR and Refused Ratio

* **MCPSecBench** (github.com/AIS2Lab/MCPSecBench) — 17 attack vectors, 4 surfaces

## **Tier 2: Generalized Agent Security**

* **AgentDojo** (github.com/ethz-spylab/agentdojo) — 97 tasks, 629 security cases. Formal utility functions, not LLM judges

* **ASB** (github.com/agiresearch/ASB) — \~90K cases, 400+ tools, 27 attack methods. Tests generalization beyond MCP-specific patterns

## **Tier 3: Capability Preservation**

* **BFCL v4** (Berkeley Function Calling Leaderboard) — AST match accuracy for function calling correctness. Primary metric for “did steering break tool use?”

* **T-Eval** (github.com/open-compass/T-Eval) — 6-dimensional capability decomposition: Plan, Reason, Retrieve, Understand, Instruct, Review. Diagnoses which specific capability steering degrades

## **Tier 4: Over-Defense Measurement**

* **BIPIA** (github.com/microsoft/BIPIA) — Both attack and defense evaluation with false positive measurement

* **Headline metric: NRP** \= PUA × (1 − ASR) from MSB. Single number capturing the safety-capability tradeoff

# **6\. Immediate Next Steps**

## **Phase 0: Setup (Week 1–2)**

1. **Everyone:** Read assigned papers (Section 2). Prioritize the papers marked for your method

2. **Infra lead:** Set up Llama 3.1-8B via vLLM with OpenAI-compatible API. Verify function calling works

3. **Infra lead:** Clone and run MCP-SafetyBench against unmodified model to get baseline ASR/TSR numbers

4. **SAE person (Methods A+C):** Install EleutherAI SAE library. Load LlamaScope SAE for Llama 3.1-8B. Verify activation extraction works

## **Phase 1: Proof of Concept (Weeks 3–6)**

5. **Method A:** Collect contrastive activations (benign vs. attack). Run FGAA filtering. Report: do clean tool-trust features exist?

6. **Method B:** Curate preference pairs from MCPTox/MCP-SafetyBench attack logs. Train rank-1 RePS steering vector. Report: initial ASR reduction

7. **Method C:** Compute tool-trust direction. Run attribution patching. Report: minimal causal feature set F\* and interpretability analysis

## **Phase 2: Full Evaluation (Weeks 7–10)**

8. Run all 3 methods through full Tier 1–4 evaluation pipeline

9. Add DPO-LoRA baseline as upper-bound comparison (changes weights, not steering)

10. Scale to 20B+ model if Phase 1 shows promise (Qwen2.5-32B or GPT-OSS-20B)

## **Phase 3: Paper (Weeks 11–14)**

11. Write up results. Frame: first paper at intersection of mechanistic interpretability and agentic security

12. Target venue: NeurIPS 2026 / ICLR 2027 / USENIX Security

# **7\. Compute & Infrastructure**

| Component | Phase 1 (8B) | Phase 2 (20B+) |
| :---- | :---- | :---- |
| **GPU requirement** | 1x A100 80GB or 2x RTX 4090 | 2–4x A100 80GB |
| **SAE training/finetuning** | \~100 GPU-hours | \~500–1000 GPU-hours |
| **RePS training** | \~30–50 GPU-hours | \~60–100 GPU-hours |
| **Evaluation sweeps** | \~100 GPU-hours | \~200 GPU-hours |
| **Est. cloud cost** | $400–600 | $1000–2000 |

# **8\. Key Tools & Repos**

* **EleutherAI SAE:** github.com/EleutherAI/sae — SAE training, finetuning (--finetune), 8-bit model loading

* **LlamaScope:** SAEs for Llama 3.1-8B already trained. Hosted on HuggingFace

* **vLLM:** Serve open-weight models with OpenAI-compatible API \+ full function calling support

* **lm-evaluation-harness:** github.com/EleutherAI/lm-evaluation-harness — MMLU, GSM8K, TruthfulQA for capability preservation

* **AXBENCH:** github.com/stanfordnlp/axbench — Large-scale model steering benchmark from RePS authors

* **TransformerLens:** Activation caching and hook-based intervention for mechanistic interpretability

# **9\. Key Risks & Mitigation**

| Risk | Impact | Mitigation |
| :---- | :---- | :---- |
| SAE features don’t decompose tool-use behavior | Methods A and C don’t work | Method B (RePS) doesn’t need SAE. VS Decomposition as SAE-free fallback. Negative result still publishable. |
| Steering breaks tool use entirely | Defense works but model is useless | Conditional steering: only activate when suspicious patterns detected. Phi-3 paper showed this preserves 96% MMLU. |
| Benchmarks don’t support open-weight models | Can’t evaluate | All selected benchmarks confirmed to work with vLLM via OpenAI-compatible API. |
| Tool-use features are entangled with instruction following | Smooth linear tradeoff, no useful elbow | Characterize the tradeoff curve explicitly. Even showing entanglement is a novel finding about how LLMs represent agentic decisions. |

*Questions? Reach out on the project Slack channel.*