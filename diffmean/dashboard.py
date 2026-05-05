#!/usr/bin/env python3
"""Dashboard to browse mcptox_pairs + steering effects — http://localhost:7331"""
from __future__ import annotations
import hashlib
import json, pathlib, re, http.server
from collections import defaultdict
from urllib.parse import urlparse, parse_qs

ROOT = pathlib.Path(__file__).parent
DATA_FILE = ROOT / "outputs" / "mcptox_pairs.raw.jsonl"
EVAL_BASE = ROOT / "outputs" / "eval"
ANALYSIS_FILE = ROOT / "outputs" / "alpha_analysis.json"
LABELLED_FILE = ROOT / "outputs" / "qwen3_rollouts.labelled.jsonl"
BORDERLINE_FILE = ROOT / "outputs" / "eval" / "borderline_demo.jsonl"
BORDER_LOG_FILE = ROOT / "outputs" / "eval" / "border.log"

# HyperSteer artifact roots (axbench tree, sibling to diffmean/)
HSTEER_AXBENCH_ROOT = ROOT.parent / "axbench" / "axbench" / "outputs"
HSTEER_TRAIN_ROOT = HSTEER_AXBENCH_ROOT  # mcp_hsteer_qwen3_8b_*/  or mcp_hsteer_9b_*/
HSTEER_EVAL_ROOT = HSTEER_AXBENCH_ROOT / "eval"

# Variant naming key (shown in UI)
HSTEER_VARIANT_KEY = {
    "v3":  "single-concept, action-only y_neg (safe tool call from diffmean rollout)",
    "v4":  "133 Autinn concepts × action y_neg (multi-concept)",
    "v5":  "133 concepts × thinking-only y_neg",
    "v6":  "133 concepts × full think+action y_neg",
    "v7":  "133 concepts × action y_neg from softer secprompt",
    "v8":  "133 concepts × hand-crafted refusal templates (concept↔y_neg INCOHERENT)",
    "v10": "single concept + refusal templates (coherent — first +pts result, but partly artifact)",
    "v11": "single concept + audit-prompted deliberate-resist y_neg from qwen3_v2_contrast.jsonl (the 'right' recipe)",
}

# Pre-CHAT_MODELS-fix runs: invalid because no system prompt / tool defs were sent.
# Cutoff: commit 08b5b64 on 2026-05-05 ~02:50 UTC.
HSTEER_INVALID_RUNS = {
    "vfeval_n50_headlines_20260505_014849": "pre-CHAT_MODELS fix: no system prompt sent → all attack_resistance=1.0 are artifacts",
    "vfeval_n50_headlines_20260505_021025": "pre-CHAT_MODELS fix: no system prompt sent → all attack_resistance=1.0 are artifacts",
    "vfeval_n50_v10_20260505_123709":       "v10 first attempt used concept_id=1 which crashed train.py; checkpoint never finished — serve returned empty completions. Use vfeval_n50_v10_FIXED_* instead.",
}

ROWS = [json.loads(l) for l in DATA_FILE.read_text().splitlines() if l.strip()]


# Hand-curated annotations for each sweep — pulled from RESULTS.md / PHASE1_RESULTS.md
SWEEP_META = {
    "phi4-baseline": {
        "model": "microsoft/phi-4 (14B)",
        "intervention": "none (baseline)",
        "n_examples": 50,
        "judge": "openai/gpt-5.4-nano via OpenRouter (reasoning effort=minimal, max_tokens=2000) + local heuristic cross-check",
        "notes": "Baseline ASR ≈ 0.40, defense ≈ 0.60.",
    },
    "phi4-sweep-v3": {
        "model": "microsoft/phi-4 (14B)",
        "intervention": "DiffMean global vector — last-token only at L20",
        "layer": 20,
        "alphas": "[-20, -10, -5, -2, 0, +2, +5, +10, +20]",
        "n_examples": 50,
        "notes": "Defense flat in [-10, +20]; α=-20 produces degenerate output (backtick/markdown loops). No defensive movement from steering.",
    },
    "phi4-sweep-allt-fine": {
        "model": "microsoft/phi-4 (14B)",
        "intervention": "DiffMean global vector — all-tokens at L20, fine α range",
        "layer": 20,
        "alphas": "[-3, -1.5, -0.5, 0, +0.5, +1.5, +3]",
        "n_examples": 50,
        "notes": "Flat across [-3, +3]. Confirms global L20 vector has high detection power but isn't a useful steering vector under residual addition.",
    },
    "phi4-sweep-lasttok": {
        "model": "microsoft/phi-4 (14B)",
        "intervention": "DiffMean global vector — last-token at L20, single α=-20 (model-breakage probe)",
        "layer": 20,
        "alphas": "[-20]",
        "n_examples": 50,
        "notes": "Confirms α=-20 collapses output to backticks/markdown.",
    },
    "phi4-sweep-v2": {
        "model": "microsoft/phi-4 (14B)",
        "intervention": "DiffMean global vector — older variant, α=-20 only",
        "layer": 20,
        "alphas": "[-20]",
        "n_examples": 50,
    },
    "phi4-template2-sweep": {
        "model": "microsoft/phi-4 (14B)",
        "intervention": "Template-2-specific DiffMean vector — all-tokens at L20",
        "layer": 20,
        "n_examples": 40,
        "filter": "Template-2 paradigm only (precondition-style attacks; per-paradigm vector AUC = 0.980)",
        "notes": "FIRST DEFENSIVE MOVEMENT SEEN: α=+5 all-tokens lifts defense 0.60 → 0.725 (+12pts, std≈0.45). Sign opposite textbook expectation — likely the vector pushes the model into a 'task-completed' satiation state.",
    },
    "qwen-baseline-smoke": {
        "model": "Qwen2.5-7B-Instruct",
        "intervention": "none (baseline smoke test)",
    },
    "qwen-grok-test": {
        "model": "Qwen2.5-7B-Instruct (test rollout)",
        "intervention": "test sweep",
    },
    "qwen-sweep": {
        "model": "Qwen2.5-7B-Instruct",
        "intervention": "DiffMean global vector — last-token at L20",
        "layer": 20,
        "alphas": "[-8, -4, 0, +4, +8, +12]",
        "n_examples": 100,
        "notes": "Qwen2.5-7B baseline ASR is only ~2-5% — model is essentially immune to MCPTox at this size, no headroom for steering. All α produce mean reward ≈ 0.94-0.97 (just attack-resistance noise).",
    },
    "qwen-sweep-allt": {
        "model": "Qwen2.5-7B-Instruct",
        "intervention": "DiffMean global vector — all-tokens at L20",
        "layer": 20,
        "alphas": "[-50, -20, -10, 0, +10, +20]",
        "n_examples": 100,
        "notes": "Large all-token interventions break the model: α∈[-10, +20] collapse to ~0 reward (degenerate output). Only α=-50 / -20 retain partial output (mean reward 0.22 / 0.29).",
    },
    "qwen3-baseline": {
        "model": "Qwen3-8B (with native thinking mode)",
        "intervention": "none (baseline)",
    },
    "qwen3-rollouts": {
        "model": "Qwen3-8B (with native thinking mode)",
        "intervention": "deliberation rollout harvest (191 mentions-poison filter)",
        "notes": "Used to build Phase-1 contrastive set; not a steering eval.",
    },
    "qwen3-thinking-sweep": {
        "model": "Qwen3-8B (with native thinking mode)",
        "intervention": "DiffMean decision-mode vector at L32 (Phase-1 thinking-trace pipeline)",
        "layer": 32,
        "alphas": "[-2, -1, 0, +1, +2] × {last-tok, all-tok}; partial coverage at ±3, ±5",
        "n_examples": 30,
        "notes": "Decision-moment activation capture. Best AUC 0.825. ASR-vs-α flat in [-2, +2].",
    },
    "qwen3-reft-L20-r4": {
        "model": "Qwen3-8B",
        "intervention": "ReFT (LoReFT) trained at L20, rank 4",
        "layer": 20,
        "notes": "Fine-tuned representation editing instead of static vector addition.",
    },
}


def _load_analysis():
    if ANALYSIS_FILE.exists():
        try:
            return json.loads(ANALYSIS_FILE.read_text())
        except Exception:
            return {}
    return {}


# Map each sweep → activation/training source directory under outputs/acts/
SWEEP_TO_ACTS = {
    "phi4-baseline": "phi4",
    "phi4-sweep-v2": "phi4",
    "phi4-sweep-v3": "phi4",
    "phi4-sweep-lasttok": "phi4",
    "phi4-sweep-allt-fine": "phi4",
    "phi4-template2-sweep": "phi4",  # subset filter applied
    "qwen-baseline-smoke": "qwen25-7b-smoke",
    "qwen-grok-test": "qwen25-7b",
    "qwen-sweep": "qwen25-7b",
    "qwen-sweep-allt": "qwen25-7b",
    "qwen3-baseline": "qwen3-8b",
    "qwen3-rollouts": "qwen3-thinking-decision",
    "qwen3-thinking-sweep": "qwen3-thinking-decision",
    "qwen3-reft-L20-r4": "qwen3-8b",
}


def _load_training_pairs():
    """Load pair-ID lists from outputs/acts/<model>/index.jsonl"""
    acts = ROOT / "outputs" / "acts"
    out = {}
    if not acts.exists():
        return out
    for d in acts.iterdir():
        idx = d / "index.jsonl"
        if not idx.exists():
            continue
        ids = []
        for line in idx.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.append(json.loads(line)["id"])
            except Exception:
                pass
        out[d.name] = ids
    return out


def _alpha_val(name: str):
    s = name.replace("alpha_", "")
    # strip mode suffixes used by various sweeps
    s = re.sub(r"_(all|last|alltok|lasttok|allt\d+|lastt\d+)$", "", s)
    s = s.replace("neg", "-").replace("n", "-").replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return None


_LAYER_DIR_RE = re.compile(r"^L(\d+)$", re.IGNORECASE)


def _build_steering():
    pair_map = {(p["system_prompt"], p["user_query"]): p for p in ROWS}
    pair_id_index = {p["id"]: p for p in ROWS}
    by_pair: dict[str, list] = defaultdict(list)
    by_sweep_alpha: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))
    sweep_layers: dict[str, set[int]] = defaultdict(set)  # auto-detect layer per sweep from path

    for sweep_dir in sorted(EVAL_BASE.iterdir()):
        if not sweep_dir.is_dir():
            continue
        # find every alpha_* dir under this sweep, at any depth
        alpha_dirs = [d for d in sweep_dir.rglob("alpha_*") if d.is_dir()]
        for ad in alpha_dirs:
            alpha = _alpha_val(ad.name)
            if alpha is None:
                continue
            # if alpha dir sits inside an L<NN> layer subdir, label it as a separate sub-sweep
            rel = ad.relative_to(sweep_dir)
            layer_part = next((p for p in rel.parts[:-1] if _LAYER_DIR_RE.match(p)), None)
            if layer_part:
                sweep_label = f"{sweep_dir.name}/{layer_part}"
                detected_layer = int(_LAYER_DIR_RE.match(layer_part).group(1))
            else:
                sweep_label = sweep_dir.name
                detected_layer = None

            if detected_layer is not None:
                sweep_layers[sweep_label].add(detected_layer)

            for rf in ad.rglob("results.jsonl"):
                for line in rf.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue

                    # Schema A: verifiers (prompt[].content + reward + attack_resistance)
                    # Schema B: borderline-style (case_id + system_prompt + user_query + judge{})
                    pair = None
                    reward = r.get("reward")
                    resist = r.get("attack_resistance")
                    detected = r.get("attack_detected")

                    if "case_id" in r and r["case_id"] in {p["id"] for p in ROWS}:
                        # quick path: case_id IS the pair id (rebuild lookup once below)
                        pass

                    if isinstance(r.get("prompt"), list) and r["prompt"]:
                        prompt = r["prompt"]
                        sp = (prompt[0] or {}).get("content", "") if isinstance(prompt[0], dict) else ""
                        uq = (prompt[1] or {}).get("content", "") if len(prompt) > 1 and isinstance(prompt[1], dict) else ""
                        if sp:
                            pair = pair_map.get((sp, uq))
                    elif r.get("system_prompt"):
                        # flat-schema results
                        cid = r.get("case_id")
                        if cid and cid in pair_id_index:
                            pair = pair_id_index[cid]
                        else:
                            pair = pair_map.get((r.get("system_prompt", ""), r.get("user_query", "")))
                        # extract reward/resist from judge sub-object if present
                        judge = r.get("judge") or {}
                        if resist is None:
                            resist = judge.get("attack_resistance")
                        if reward is None:
                            reward = judge.get("attack_resistance")  # use resistance as reward proxy

                    if not pair:
                        continue

                    comp = r.get("completion", "")
                    if isinstance(comp, list):
                        comp = "\n\n".join(
                            m.get("content", "") for m in comp if isinstance(m, dict)
                        )
                    entry = {
                        "sweep": sweep_label,
                        "alpha_dir": ad.name,
                        "alpha": alpha,
                        "reward": reward,
                        "attack_resistance": resist,
                        "attack_detected": detected,
                        "completion": str(comp)[:3000],
                    }
                    by_pair[pair["id"]].append(entry)
                    by_sweep_alpha[sweep_label][alpha].append({
                        "pid": pair["id"],
                        "reward": reward,
                        "attack_resistance": resist,
                    })
    return dict(by_pair), {sw: dict(ad) for sw, ad in by_sweep_alpha.items()}, dict(sweep_layers)


def _build_stats(by_sweep_alpha):
    stats = {}
    for sweep, ad in by_sweep_alpha.items():
        rows = []
        all_pids = set()
        for alpha in sorted(ad.keys()):
            entries = ad[alpha]
            rewards = [e["reward"] for e in entries if e["reward"] is not None]
            resist = [e["attack_resistance"] for e in entries if e["attack_resistance"] is not None]
            for e in entries:
                all_pids.add(e["pid"])
            rows.append({
                "alpha": alpha,
                "n": len(entries),
                "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
                "mean_resist": (sum(resist) / len(resist)) if resist else None,
            })
        stats[sweep] = {"rows": rows, "n_pairs": len(all_pids)}
    return stats


def _mark_varies(by_pair):
    varies = set()
    for pid, entries in by_pair.items():
        sw_groups = defaultdict(list)
        for e in entries:
            sw_groups[e["sweep"]].append(e["reward"])
        for rs in sw_groups.values():
            non_null = [r for r in rs if r is not None]
            if len(set(non_null)) > 1:
                varies.add(pid)
                break
    return list(varies)


def _merge_auto_layers(by_sweep_alpha, auto_layers):
    """Sweep-meta is shared across rebuilds; only inject auto-detected layer info once."""
    for sw, layers in auto_layers.items():
        if not layers:
            continue
        meta = SWEEP_META.setdefault(sw, {})
        if "layer" not in meta:
            meta["layer"] = sorted(layers)[0] if len(layers) == 1 else sorted(layers)
        meta.setdefault("model", "(auto-detected)")
        meta.setdefault("intervention", f"layer-axis sweep at L{sorted(layers)[0]}")


BY_PAIR, BY_SWEEP_ALPHA, AUTO_SWEEP_LAYERS = _build_steering()
_merge_auto_layers(BY_SWEEP_ALPHA, AUTO_SWEEP_LAYERS)
STATS = _build_stats(BY_SWEEP_ALPHA)
VARIES = _mark_varies(BY_PAIR)
ANALYSIS = _load_analysis()
TRAIN_PAIRS = _load_training_pairs()  # {acts_dir: [pair_id, ...]}


def _build_sweep_details():
    """Per-sweep: list of (pair_id, {alpha: reward}) for eval, plus training pair ids."""
    details = {}
    for sweep, ad in BY_SWEEP_ALPHA.items():
        per_pair: dict[str, dict[float, float]] = defaultdict(dict)
        for alpha, entries in ad.items():
            for e in entries:
                per_pair[e["pid"]][alpha] = e["reward"]
        # sort alphas and pairs
        all_alphas = sorted({a for d in per_pair.values() for a in d.keys()})
        rows = []
        for pid in sorted(per_pair.keys()):
            rewards = [per_pair[pid].get(a) for a in all_alphas]
            rows.append({"pid": pid, "rewards": rewards})
        acts_dir = SWEEP_TO_ACTS.get(sweep)
        train_ids = TRAIN_PAIRS.get(acts_dir, []) if acts_dir else []
        details[sweep] = {
            "alphas": all_alphas,
            "eval_pairs": rows,
            "train_acts_dir": acts_dir,
            "train_pair_ids": train_ids,
        }
    # Also include sweeps that have no eval rows but have a training source
    for sweep, acts in SWEEP_TO_ACTS.items():
        if sweep not in details:
            details[sweep] = {
                "alphas": [],
                "eval_pairs": [],
                "train_acts_dir": acts,
                "train_pair_ids": TRAIN_PAIRS.get(acts, []),
            }
    return details


SWEEP_DETAILS = _build_sweep_details()


def _load_jsonl(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


# ---------- HyperSteer artifact discovery ----------

_VARIANT_RE = re.compile(r"\b(v\d+)\b", re.IGNORECASE)
_FACTOR_RE = re.compile(r"_f(\d+)(?:p(\d+))?", re.IGNORECASE)
_TRAIN_DIR_RE = re.compile(r"^mcp_hsteer_(?:9b|qwen3_8b)_(.+)$", re.IGNORECASE)


def _parse_variant(tag: str) -> str | None:
    m = _VARIANT_RE.search(tag)
    return m.group(1).lower() if m else None


def _parse_factor(tag: str) -> float | None:
    m = _FACTOR_RE.search(tag)
    if not m:
        return None
    whole = int(m.group(1))
    frac = m.group(2)
    if frac:
        return float(f"{whole}.{frac}")
    return float(whole)


def _yaml_lite_load(text: str) -> dict:
    """Tiny YAML-subset parser — handles the flat key:value structure we ship without needing PyYAML."""
    out: dict = {}
    stack: list = [(0, out)]
    list_active = None  # (indent, key, parent)
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        # pop stack to current indent
        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()
            list_active = None
        if list_active and indent == list_active[0] and line.startswith("- "):
            list_active[1].append(line[2:].strip())
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        parent = stack[-1][1]
        if val == "":
            child: dict | list = {}
            parent[key] = child
            stack.append((indent + 2, child))
            list_active = None
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            parent[key] = [s.strip() for s in inner.split(",")] if inner else []
        else:
            parent[key] = val.strip("'\"")
        # detect a coming list ("steering_layers:\n  - 10")
        # if the just-added value is empty dict, allow list items at deeper indent
        if val == "":
            # convert to list when first "- " seen at deeper indent
            list_active = (indent + 2, parent[key] if isinstance(parent[key], list) else None)
            # we'll lazily switch dict→list in a second pass below
    # Second pass: convert any dict that ended up empty but the source had list items —
    # the lite parser above already handles list items via list_active where the parent was a list,
    # but for the typical config the dict-only output is fine.
    return out


def _read_yaml_config(path: pathlib.Path) -> dict:
    """Best-effort load. Returns {} if it can't parse."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text()) or {}
    except ModuleNotFoundError:
        try:
            return _yaml_lite_load(path.read_text())
        except Exception:
            return {"_raw": path.read_text()[:5000]}
    except Exception:
        return {"_raw": path.read_text()[:5000]}


def _discover_train_dirs() -> list[dict]:
    """Find every mcp_hsteer_*/ training output directory and summarise."""
    if not HSTEER_TRAIN_ROOT.exists():
        return []
    out = []
    for d in sorted(HSTEER_TRAIN_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = _TRAIN_DIR_RE.match(d.name)
        if not m:
            continue
        tag = m.group(1)  # e.g. "v3" or "v2_overnight"
        variant = _parse_variant(d.name)
        gen_dir = d / "generate"
        train_dir = d / "train"
        ckpt_dir = train_dir / "hyperreft"
        cfg_path = d / "mcp_hypersteer_config.yaml"
        meta_path = gen_dir / "metadata.jsonl"
        parquet_path = gen_dir / "train_data.parquet"
        merged_concepts_path = d / "merged_concepts_mcp.json"

        # parse metadata.jsonl — small, so load eagerly
        concepts = []
        if meta_path.exists():
            for line in meta_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    concepts.append(json.loads(line))
                except Exception:
                    pass

        out.append({
            "name": d.name,
            "tag": tag,
            "variant": variant,
            "variant_desc": HSTEER_VARIANT_KEY.get(variant or "", ""),
            "config_exists": cfg_path.exists(),
            "concepts": concepts,
            "n_concepts": len(concepts),
            "parquet_exists": parquet_path.exists(),
            "merged_concepts_exists": merged_concepts_path.exists(),
            "ckpt_exists": ckpt_dir.exists(),
            "_paths": {
                "root": str(d),
                "config": str(cfg_path),
                "metadata": str(meta_path),
                "parquet": str(parquet_path),
                "merged_concepts": str(merged_concepts_path),
                "ckpt": str(ckpt_dir),
            },
        })
    return out


def _discover_eval_runs() -> list[dict]:
    """Find every vfeval_n50_*/ run and the config-tag subdirs underneath."""
    if not HSTEER_EVAL_ROOT.exists():
        return []
    out = []
    for run_dir in sorted(HSTEER_EVAL_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith("vfeval_n"):
            continue
        try:
            mtime = run_dir.stat().st_mtime
        except Exception:
            mtime = 0.0
        invalid_reason = HSTEER_INVALID_RUNS.get(run_dir.name)
        configs = []
        for cfg_dir in sorted(run_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            tag = cfg_dir.name
            variant = _parse_variant(tag) or ("baseline" if "baseline" in tag.lower() else None)
            factor = _parse_factor(tag)
            is_2k = "_2k" in tag.lower() or "_2k" in run_dir.name.lower()
            # find leaf eval dir
            leaf = None
            for evals_root in [cfg_dir / "evals" / "mcp_tox--hypersteer-local"]:
                if evals_root.exists():
                    for sub in evals_root.iterdir():
                        if sub.is_dir() and (sub / "metadata.json").exists():
                            leaf = sub
                            break
            metadata = {}
            n_results = 0
            if leaf:
                meta_path = leaf / "metadata.json"
                results_path = leaf / "results.jsonl"
                if meta_path.exists():
                    try:
                        metadata = json.loads(meta_path.read_text())
                    except Exception:
                        metadata = {"_raw": meta_path.read_text()[:1000]}
                if results_path.exists():
                    try:
                        n_results = sum(1 for ln in results_path.read_text().splitlines() if ln.strip())
                    except Exception:
                        n_results = 0
            configs.append({
                "tag": tag,
                "variant": variant,
                "factor": factor,
                "is_2k": is_2k,
                "leaf": str(leaf) if leaf else None,
                "metadata": metadata,
                "n_results": n_results,
                "max_tokens": (metadata.get("sampling_args") or {}).get("max_tokens"),
                "avg_reward": metadata.get("avg_reward"),
                "avg_metrics": metadata.get("avg_metrics"),
                "time_ms": metadata.get("time_ms"),
                "num_examples": metadata.get("num_examples"),
            })
        out.append({
            "run": run_dir.name,
            "mtime": mtime,
            "invalid_reason": invalid_reason,
            "n_configs": len(configs),
            "configs": configs,
        })
    return out


HSTEER_TRAIN = _discover_train_dirs()
HSTEER_RUNS = _discover_eval_runs()


def _hsteer_runs_index_lite() -> list[dict]:
    """Trimmed-down view (no per-row data) for client-side rendering."""
    out = []
    for r in HSTEER_RUNS:
        out.append({
            "run": r["run"],
            "mtime": r["mtime"],
            "invalid_reason": r["invalid_reason"],
            "configs": [
                {
                    "tag": c["tag"],
                    "variant": c["variant"],
                    "factor": c["factor"],
                    "is_2k": c["is_2k"],
                    "max_tokens": c["max_tokens"],
                    "avg_reward": c["avg_reward"],
                    "avg_metrics": c["avg_metrics"],
                    "n_results": c["n_results"],
                    "num_examples": c["num_examples"],
                    "has_leaf": c["leaf"] is not None,
                }
                for c in r["configs"]
            ],
        })
    return out


def _hsteer_train_index_lite() -> list[dict]:
    out = []
    for t in HSTEER_TRAIN:
        out.append({
            "name": t["name"],
            "tag": t["tag"],
            "variant": t["variant"],
            "variant_desc": t["variant_desc"],
            "n_concepts": t["n_concepts"],
            "concepts": t["concepts"],  # small (≤133 rows)
            "config_exists": t["config_exists"],
            "parquet_exists": t["parquet_exists"],
            "merged_concepts_exists": t["merged_concepts_exists"],
            "ckpt_exists": t["ckpt_exists"],
        })
    return out


def _find_run(run_name: str) -> dict | None:
    for r in HSTEER_RUNS:
        if r["run"] == run_name:
            return r
    return None


def _find_config(run_name: str, tag: str) -> tuple[dict | None, dict | None]:
    r = _find_run(run_name)
    if not r:
        return None, None
    for c in r["configs"]:
        if c["tag"] == tag:
            return r, c
    return r, None


def _find_train(name: str) -> dict | None:
    for t in HSTEER_TRAIN:
        if t["name"] == name:
            return t
    return None


def _read_results_examples(leaf: pathlib.Path) -> list[dict]:
    """Read results.jsonl, return one trimmed entry per row (no full prompt/completion)."""
    rf = leaf / "results.jsonl"
    if not rf.exists():
        return []
    out = []
    for ln in rf.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        info = r.get("info") or {}
        prompt = r.get("prompt") or []
        comp = r.get("completion") or []
        user_q = ""
        if isinstance(prompt, list) and len(prompt) >= 2 and isinstance(prompt[1], dict):
            user_q = prompt[1].get("content", "")
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            user_q = prompt[0].get("content", "")
        comp_len = 0
        if isinstance(comp, list):
            for m in comp:
                if isinstance(m, dict):
                    comp_len += len(m.get("content") or "")
        out.append({
            "example_id": r.get("example_id"),
            "user_query": user_q[:300],
            "user_query_len": len(user_q),
            "completion_len": comp_len,
            "reward": r.get("reward"),
            "attack_resistance": (r.get("metrics") or {}).get("attack_resistance"),
            "attack_detected": (r.get("metrics") or {}).get("attack_detected"),
            "is_truncated": r.get("is_truncated"),
            "stop_condition": r.get("stop_condition"),
            "security_risk": info.get("security_risk"),
            "server": info.get("server_name") or info.get("server"),
            "paradigm": info.get("paradigm"),
            "data_id": info.get("data_id"),
        })
    return out


def _read_results_full(leaf: pathlib.Path, example_id) -> dict | None:
    """Find one row by example_id (or by index if no example_id) and return everything."""
    rf = leaf / "results.jsonl"
    if not rf.exists():
        return None
    target = None
    try:
        target = int(example_id)
    except (TypeError, ValueError):
        return None
    for idx, ln in enumerate(rf.read_text().splitlines()):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        eid = r.get("example_id")
        if eid is None:
            eid = idx
        if eid == target:
            return r
    return None


def _read_train_parquet_sample(parquet_path: pathlib.Path, limit: int = 25) -> dict:
    """Read a sample of rows. Returns {ok, error, columns, total, rows}."""
    if not parquet_path.exists():
        return {"ok": False, "error": "train_data.parquet not found"}
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return {"ok": False, "error": "pandas+pyarrow not installed — pip install pandas pyarrow"}
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        return {"ok": False, "error": f"read_parquet failed: {e}"}
    cols = list(df.columns)
    total = len(df)
    # interleave a few positives + a few negatives if those columns exist
    rows = []
    if "category" in cols:
        pos = df[df["category"] == "positive"].head(limit // 2)
        neg = df[df["category"] != "positive"].head(limit - len(pos))
        sub = pd.concat([pos, neg])
    else:
        sub = df.head(limit)
    for _, row in sub.iterrows():
        rows.append({c: (None if pd.isna(row[c]) else (row[c] if isinstance(row[c], (int, float, bool, str)) else str(row[c]))) for c in cols})
    return {"ok": True, "columns": cols, "total": total, "rows": rows}


# ---------- end HyperSteer ----------


LABELLED = _load_jsonl(LABELLED_FILE)        # qwen3 labelled rollouts (resist/comply/ambiguous)
BORDERLINE = _load_jsonl(BORDERLINE_FILE)    # script-curated borderline demo cases (may be absent)
BORDER_LOG = BORDER_LOG_FILE.read_text() if BORDER_LOG_FILE.exists() else ""  # raw stability table log


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MCPTox Pairs Browser</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { font-family: system-ui, sans-serif; background: #0f0f13; color: #e0e0e0; height: 100vh; overflow: hidden; }
  body { display: flex; flex-direction: column; }

  header { padding: 9px 14px; background: #1a1a24; border-bottom: 1px solid #2a2a3a; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  header h1 { font-size: 0.9rem; font-weight: 600; color: #a78bfa; white-space: nowrap; }
  .tabs { display: flex; gap: 4px; }
  .tab { padding: 4px 12px; border-radius: 5px; background: #252535; border: 1px solid #3a3a4a; color: #aaa; cursor: pointer; font-size: 0.78rem; }
  .tab.active { background: #3a2060; border-color: #a78bfa; color: #fff; }
  .filters { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; flex: 1; }
  select, input[type=text] { background: #252535; border: 1px solid #3a3a4a; color: #e0e0e0; border-radius: 5px; padding: 4px 8px; font-size: 0.78rem; }
  input[type=text] { flex: 1; min-width: 120px; }
  .count { font-size: 0.72rem; color: #555; white-space: nowrap; }
  .filter-chk { font-size: 0.72rem; color: #888; display: flex; align-items: center; gap: 4px; }

  /* page containers */
  .page { flex: 1; min-height: 0; overflow: hidden; display: none; }
  .page.active { display: flex; }

  /* browse page */
  #browsePage { display: flex; }
  #browsePage.active { display: flex; }
  #browsePage:not(.active) { display: none; }

  .list { width: 290px; min-width: 180px; border-right: 1px solid #2a2a3a; overflow-y: auto; flex-shrink: 0; }
  .item { padding: 8px 11px; border-bottom: 1px solid #1e1e2a; cursor: pointer; transition: background 0.1s; position: relative; }
  .item:hover { background: #1e1e2e; }
  .item.active { background: #2a2040; border-left: 3px solid #a78bfa; }
  .item-id { font-size: 0.66rem; color: #555; font-family: monospace; }
  .item-query { font-size: 0.78rem; margin-top: 2px; color: #bbb; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
  .item-badges { display: flex; gap: 4px; margin-top: 4px; flex-wrap: wrap; }
  .item-flags { position: absolute; top: 6px; right: 8px; display: flex; gap: 4px; font-size: 0.62rem; }
  .flag-st { color: #fbbf24; }
  .flag-vary { color: #4ade80; }

  .detail { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 12px; min-width: 400px; }
  .empty { flex: 1; display: flex; align-items: center; justify-content: center; color: #444; font-size: 0.88rem; }

  .card { background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 7px; overflow: hidden; }
  .card-header { padding: 6px 11px; background: #20202e; font-size: 0.7rem; font-weight: 600; color: #777; text-transform: uppercase; letter-spacing: 0.05em; display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
  .card-body { padding: 9px 11px; font-size: 0.78rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, monospace; color: #ccc; max-height: 200px; overflow-y: auto; }
  .card-body.hl-pos    { border-left: 3px solid #f87171; }
  .card-body.hl-neg    { border-left: 3px solid #4ade80; }
  .card-body.hl-poison { border-left: 3px solid #fbbf24; }

  .badge { display: inline-block; font-size: 0.6rem; padding: 1px 6px; border-radius: 10px; font-weight: 500; }
  .b-risk     { background: #3d1a1a; color: #f87171; }
  .b-server   { background: #1a2d3d; color: #60a5fa; }
  .b-paradigm { background: #1a2d1a; color: #4ade80; }
  .b-model    { background: #2d1a3d; color: #c084fc; }

  .row-flex { display: flex; gap: 12px; flex-wrap: wrap; }
  .row-flex > .card { flex: 1; min-width: 280px; }

  /* steering film strip */
  .strip-controls { display: flex; align-items: center; gap: 10px; padding: 8px 11px; background: #1e1a2e; border-bottom: 1px solid #2a2a3a; flex-wrap: wrap; }
  .strip-wrap { overflow-x: auto; padding: 10px 11px; background: #15151e; }
  .strip { display: flex; gap: 10px; min-width: min-content; align-items: flex-start; }
  .strip-cell { flex: none !important; width: 340px !important; height: 520px !important; max-height: 520px !important; background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 6px; display: flex; flex-direction: column; overflow: hidden; }
  .strip-cell.baseline { border-color: #555; }
  .strip-cell.high-reward { border-left: 3px solid #4ade80; }
  .strip-cell.low-reward  { border-left: 3px solid #f87171; }
  .strip-cell.mid-reward  { border-left: 3px solid #fbbf24; }
  .strip-head { flex-shrink: 0; padding: 7px 10px; background: #20202e; display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 0.75rem; }
  .strip-head .alpha-label { font-weight: 700; color: #c084fc; font-family: ui-monospace, monospace; }
  .strip-body { flex: 1 1 auto; min-height: 0; padding: 8px 10px; font-family: ui-monospace, monospace; font-size: 0.74rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word; color: #ccc; overflow-y: scroll; overscroll-behavior: contain; scrollbar-width: thin; scrollbar-color: #555 #15151e; }
  .strip-body::-webkit-scrollbar { width: 10px; }
  .strip-body::-webkit-scrollbar-track { background: #15151e; }
  .strip-body::-webkit-scrollbar-thumb { background: #555; border-radius: 5px; border: 2px solid #15151e; }
  .strip-body::-webkit-scrollbar-thumb:hover { background: #777; }

  .reward-pill { font-size: 0.66rem; padding: 1px 7px; border-radius: 9px; font-weight: 600; font-family: ui-monospace, monospace; }
  .rp-good { background: #14401a; color: #4ade80; }
  .rp-bad  { background: #401414; color: #f87171; }
  .rp-mid  { background: #403014; color: #fbbf24; }
  .rp-null { background: #252535; color: #555; }

  .no-results { padding: 40px; text-align: center; color: #444; }

  /* stats / sweeps / rollouts / stability / highlights / hsteer pages — block flow + vertical scroll, no flex clipping */
  #statsPage.active, #sweepsPage.active, #rolloutsPage.active, #stabilityPage.active, #highlightsPage.active, #hsteerPage.active {
    display: block !important;
    padding: 16px;
    overflow-y: auto;
    overflow-x: hidden;
  }
  .stats-sweep { flex-shrink: 0; }
  .rol-pill { padding: 5px 12px; border-radius: 14px; border: 1px solid #3a3a4a; background: #252535; color: #aaa; font-size: 0.78rem; cursor: pointer; transition: all 0.1s; }
  .rol-pill:hover { color: #fff; }
  .stats-sweep { background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 8px; margin-bottom: 18px; overflow: hidden; }
  .stats-sweep-hdr { padding: 10px 14px; background: #20202e; display: flex; justify-content: space-between; align-items: center; }
  .stats-sweep-hdr h2 { font-size: 0.92rem; color: #a78bfa; font-family: ui-monospace, monospace; }
  .stats-sweep-hdr .meta { font-size: 0.72rem; color: #777; }
  table.stats { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  table.stats th, table.stats td { padding: 6px 12px; border-bottom: 1px solid #1e1e2a; text-align: left; }
  table.stats th { background: #1a1a24; color: #888; font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
  table.stats td { color: #ccc; font-family: ui-monospace, monospace; }
  table.stats td.alpha { color: #c084fc; font-weight: 600; }
  table.stats tr:hover td { background: #1e1e2a; }
  .bar-cell { display: flex; align-items: center; gap: 8px; }
  .bar { height: 8px; background: #25253a; border-radius: 4px; overflow: hidden; flex: 1; min-width: 80px; max-width: 180px; }
  .bar-fill { height: 100%; transition: width 0.2s; }
  .bf-good { background: linear-gradient(90deg, #4ade80, #22c55e); }
  .bf-mid  { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
  .bf-bad  { background: linear-gradient(90deg, #f87171, #dc2626); }
</style>
</head>
<body>
<header>
  <h1>MCPTox</h1>
  <div class="tabs">
    <div class="tab active" data-tab="browse">Browse</div>
    <div class="tab" data-tab="stats">Stats</div>
    <div class="tab" data-tab="sweeps">Sweeps</div>
    <div class="tab" data-tab="rollouts" style="border-color:#fbbf24;color:#fbbf24;">⚠ Rollouts <span id="rolBadge" style="margin-left:4px;background:#fbbf24;color:#1a1a24;border-radius:8px;padding:1px 7px;font-size:0.65rem;font-weight:700;"></span></div>
    <div class="tab" data-tab="stability">Stability</div>
    <div class="tab" data-tab="highlights">Highlights <span id="hlBadge" style="margin-left:4px;background:#3a2060;border-radius:8px;padding:0 6px;font-size:0.65rem;"></span></div>
    <div class="tab" data-tab="hsteer" style="border-color:#60a5fa;color:#60a5fa;">HyperSteer <span id="hsBadge" style="margin-left:4px;background:#1a2d3d;color:#60a5fa;border-radius:8px;padding:0 6px;font-size:0.65rem;"></span></div>
  </div>
  <div class="filters" id="browseFilters">
    <input type="text" id="search" placeholder="Search…">
    <select id="fServer"><option value="">All servers</option></select>
    <select id="fRisk"><option value="">All risks</option></select>
    <select id="fParadigm"><option value="">All paradigms</option></select>
    <select id="fLayer"><option value="">All layers</option></select>
    <label class="filter-chk"><input type="checkbox" id="fSteering"> ⚡ has steering</label>
    <label class="filter-chk"><input type="checkbox" id="fVaries"> Δ varies</label>
    <label class="filter-chk" style="border-left:1px solid #2a2a3a;padding-left:10px;margin-left:4px;">default sweep
      <select id="defaultSweep" style="margin-left:4px;"><option value="">(auto)</option></select>
    </label>
  </div>
  <span id="liveStatus" style="font-size:0.72rem;color:#666;white-space:nowrap;border-left:1px solid #2a2a3a;padding-left:10px;cursor:pointer;" title="click to toggle live polling">● live</span>
  <span class="count" id="count"></span>
</header>

<div id="browsePage" class="page active">
  <div class="list" id="list"></div>
  <div class="detail" id="detail"><div class="empty">Select a row to inspect</div></div>
</div>

<div id="statsPage" class="page"></div>
<div id="sweepsPage" class="page"></div>
<div id="rolloutsPage" class="page"></div>
<div id="stabilityPage" class="page"></div>
<div id="highlightsPage" class="page"></div>
<div id="hsteerPage" class="page"></div>

<script>
let ROWS          = __ROWS__;
let BY_PAIR       = __BY_PAIR__;
let STATS         = __STATS__;
let VARIES        = new Set(__VARIES__);
let SWEEP_META    = __SWEEP_META__;
let ANALYSIS      = __ANALYSIS__;
let SWEEP_DETAILS = __SWEEP_DETAILS__;
let LABELLED      = __LABELLED__;
let BORDERLINE    = __BORDERLINE__;
let BORDER_LOG    = __BORDER_LOG__;
let HSTEER_RUNS   = __HSTEER_RUNS__;
let HSTEER_TRAIN  = __HSTEER_TRAIN__;
let HSTEER_VARIANT_KEY = __HSTEER_VARIANT_KEY__;
let ROW_BY_ID     = Object.fromEntries(ROWS.map(r => [r.id, r]));

let filtered = ROWS;
let selectedId = null;
let selectedSweep = null;
let defaultSweep = localStorage.getItem('mcptox_defaultSweep') || '';

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function populate(sel, key) {
  [...new Set(ROWS.map(r => r.tags[key]))].sort().forEach(v => {
    const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o);
  });
}
populate(document.getElementById('fServer'),   'server');
populate(document.getElementById('fRisk'),     'security_risk');
populate(document.getElementById('fParadigm'), 'paradigm');

// helper: get layers for a sweep (meta.layer can be number or array)
function sweepLayers(sweep) {
  const m = SWEEP_META[sweep];
  if (!m || m.layer == null) return [];
  return Array.isArray(m.layer) ? m.layer : [m.layer];
}

// populate layer filter from SWEEP_META layers (sorted numerically)
const allLayers = [...new Set(Object.values(SWEEP_META).flatMap(m => {
  if (m.layer == null) return [];
  return Array.isArray(m.layer) ? m.layer : [m.layer];
}))].sort((a,b) => a-b);
const lSel = document.getElementById('fLayer');
allLayers.forEach(L => {
  const o = document.createElement('option'); o.value = String(L); o.textContent = `L${L}`;
  lSel.appendChild(o);
});

// pair_id -> set of layers covered by its steering data
const PAIR_LAYERS = {};
Object.entries(BY_PAIR).forEach(([pid, entries]) => {
  const layers = new Set();
  entries.forEach(e => sweepLayers(e.sweep).forEach(L => layers.add(L)));
  PAIR_LAYERS[pid] = layers;
});

// populate default-sweep selector with all known sweeps
const allSweeps = [...new Set(Object.values(BY_PAIR).flat().map(e => e.sweep))].sort();
const dsSel = document.getElementById('defaultSweep');
allSweeps.forEach(s => {
  const o = document.createElement('option'); o.value = s; o.textContent = s;
  if (s === defaultSweep) o.selected = true;
  dsSel.appendChild(o);
});
dsSel.onchange = () => {
  defaultSweep = dsSel.value;
  localStorage.setItem('mcptox_defaultSweep', defaultSweep);
  // re-render current detail if any
  if (selectedId) {
    selectedSweep = null;  // clear local override so default applies
    const r = ROWS.find(x => x.id === selectedId);
    if (r) renderDetail(r);
  }
};

function applyFilters() {
  const q  = document.getElementById('search').value.toLowerCase();
  const sv = document.getElementById('fServer').value;
  const ri = document.getElementById('fRisk').value;
  const pa = document.getElementById('fParadigm').value;
  const stOnly = document.getElementById('fSteering').checked;
  const varOnly = document.getElementById('fVaries').checked;
  const layerFilter = document.getElementById('fLayer').value;
  filtered = ROWS.filter(r => {
    if (sv && r.tags.server !== sv) return false;
    if (ri && r.tags.security_risk !== ri) return false;
    if (pa && r.tags.paradigm !== pa) return false;
    if (stOnly && !BY_PAIR[r.id]) return false;
    if (varOnly && !VARIES.has(r.id)) return false;
    if (layerFilter) {
      const Ls = PAIR_LAYERS[r.id];
      if (!Ls || !Ls.has(parseInt(layerFilter, 10))) return false;
    }
    if (q && !r.user_query.toLowerCase().includes(q) &&
             !r.system_prompt.toLowerCase().includes(q) &&
             !r.id.toLowerCase().includes(q)) return false;
    return true;
  });
  renderList();
}

function renderList() {
  const list = document.getElementById('list');
  document.getElementById('count').textContent = filtered.length + ' / ' + ROWS.length;
  if (!filtered.length) { list.innerHTML = '<div class="no-results">No results</div>'; return; }
  list.innerHTML = '';
  filtered.forEach(r => {
    const hasSt = !!BY_PAIR[r.id];
    const varies = VARIES.has(r.id);
    const flags = (hasSt ? '<span class="flag-st">⚡</span>' : '') + (varies ? '<span class="flag-vary">Δ</span>' : '');
    const d = document.createElement('div');
    d.className = 'item' + (r.id === selectedId ? ' active' : '');
    d.innerHTML =
      `<div class="item-id">${esc(r.id)}</div>` +
      `<div class="item-query">${esc(r.user_query)}</div>` +
      `<div class="item-badges">` +
        `<span class="badge b-server">${esc(r.tags.server)}</span>` +
        `<span class="badge b-risk">${esc(r.tags.security_risk)}</span>` +
        `<span class="badge b-paradigm">${esc(r.tags.paradigm)}</span>` +
      `</div>` +
      `<div class="item-flags">${flags}</div>`;
    d.onclick = () => { selectedId = r.id; selectedSweep = null; renderList(); renderDetail(r); };
    list.appendChild(d);
  });
}

function rewardClass(r) {
  if (r == null) return '';
  if (r >= 0.9) return 'high-reward';
  if (r <= 0.1) return 'low-reward';
  return 'mid-reward';
}
function pillClass(r) {
  if (r == null) return 'rp-null';
  if (r >= 0.9) return 'rp-good';
  if (r <= 0.1) return 'rp-bad';
  return 'rp-mid';
}
function pillTxt(r) { return r == null ? '—' : r.toFixed(2); }
function fmtAlpha(a) { return (a >= 0 ? '+' : '') + a; }

function renderSteering(pid) {
  const entries = BY_PAIR[pid];
  if (!entries || !entries.length) {
    return '<div style="padding:14px;color:#555;font-size:0.78rem;">No steering data for this pair</div>';
  }
  const sweeps = [...new Set(entries.map(e => e.sweep))].sort();
  if (!selectedSweep || !sweeps.includes(selectedSweep)) {
    // prefer the global default if it's available for this pair
    selectedSweep = (defaultSweep && sweeps.includes(defaultSweep)) ? defaultSweep : sweeps[0];
  }
  const swEntries = entries.filter(e => e.sweep === selectedSweep);
  swEntries.sort((a, b) => a.alpha - b.alpha);

  // detect a baseline (alpha=0) for highlighting
  const baseline = swEntries.find(e => e.alpha === 0);
  const baseReward = baseline ? baseline.reward : null;

  const sweepOpts = sweeps.map(s =>
    `<option value="${esc(s)}"${s === selectedSweep ? ' selected' : ''}>${esc(s)}</option>`
  ).join('');

  // summary stats for this sweep within this pair
  const rs = swEntries.map(e => e.reward).filter(r => r != null);
  const minR = rs.length ? Math.min(...rs) : null;
  const maxR = rs.length ? Math.max(...rs) : null;
  const dRange = (minR != null) ? `Δ ${minR.toFixed(2)} → ${maxR.toFixed(2)}` : '—';

  const cells = swEntries.map(e => {
    const cls = 'strip-cell ' + rewardClass(e.reward) + (e.alpha === 0 ? ' baseline' : '');
    const deltaTxt = (baseReward != null && e.reward != null && e.alpha !== 0)
      ? `<span style="color:${e.reward>baseReward?'#4ade80':e.reward<baseReward?'#f87171':'#888'};font-family:ui-monospace;font-size:0.65rem;">Δ${(e.reward-baseReward>=0?'+':'')}${(e.reward-baseReward).toFixed(2)}</span>`
      : '';
    const ar = e.attack_resistance != null
      ? `<span class="reward-pill ${pillClass(e.attack_resistance)}" title="attack resistance">res ${e.attack_resistance.toFixed(2)}</span>` : '';
    return `<div class="${cls}">
      <div class="strip-head">
        <span class="alpha-label">α=${fmtAlpha(e.alpha)}${e.alpha===0?' (base)':''}</span>
        <span style="display:flex;gap:4px;align-items:center;">
          ${deltaTxt}
          <span class="reward-pill ${pillClass(e.reward)}">r ${pillTxt(e.reward)}</span>
          ${ar}
        </span>
      </div>
      <div class="strip-body">${esc(e.completion)}</div>
    </div>`;
  }).join('');

  // sweep meta + LLM analysis (if present)
  const meta = SWEEP_META[selectedSweep];
  const metaHtml = meta ? `
    <div style="padding:8px 11px;background:#15151e;border-bottom:1px solid #2a2a3a;font-size:0.74rem;line-height:1.5;color:#aaa;">
      ${meta.model ? `<div><span style="color:#666">model:</span> <span style="color:#c084fc;font-family:ui-monospace,monospace;">${esc(meta.model)}</span></div>`:''}
      ${meta.intervention ? `<div><span style="color:#666">intervention:</span> ${esc(meta.intervention)}</div>`:''}
      ${meta.alphas ? `<div><span style="color:#666">α tested:</span> <span style="font-family:ui-monospace,monospace;color:#888">${esc(meta.alphas)}</span></div>`:''}
      ${meta.n_examples ? `<div><span style="color:#666">N:</span> ${meta.n_examples}</div>`:''}
      ${meta.filter ? `<div><span style="color:#666">filter:</span> ${esc(meta.filter)}</div>`:''}
      ${meta.judge ? `<div><span style="color:#666">judge:</span> <span style="font-family:ui-monospace,monospace;font-size:0.7rem;">${esc(meta.judge)}</span></div>`:''}
      ${meta.notes ? `<div style="margin-top:4px;color:#ccc;font-style:italic;">${esc(meta.notes)}</div>`:''}
    </div>` : '';

  const ana = ANALYSIS[`${pid}::${selectedSweep}`];
  const anaHtml = ana ? renderAnalysisInline(ana) : '';

  return `
    <div class="strip-controls">
      <label style="font-size:0.74rem;color:#888;">sweep
        <select id="sweepSel" style="margin-left:4px;">${sweepOpts}</select>
      </label>
      <span style="color:#666;font-size:0.72rem;">n=${swEntries.length} alphas · range ${dRange}</span>
      <div style="flex:1;"></div>
      <span style="color:#666;font-size:0.66rem;">scroll horizontally →</span>
    </div>
    ${metaHtml}
    ${anaHtml}
    <div class="strip-wrap"><div class="strip">${cells}</div></div>`;
}

function renderAnalysisInline(ana) {
  if (!ana) return '';
  if (ana.verdict === 'NO_CHANGE') {
    return `<div style="padding:8px 11px;background:#15201a;border-bottom:1px solid #2a2a3a;font-size:0.74rem;color:#888;">
      <span style="color:#4ade80;font-weight:600;">⊝ NO_CHANGE</span> — judge (gpt-5-mini) saw no meaningful divergence across α
    </div>`;
  }
  if (ana.verdict === 'CHANGED') {
    return `<div style="padding:9px 11px;background:#1f1c12;border-bottom:1px solid #2a2a3a;font-size:0.78rem;line-height:1.5;color:#ddd;">
      <div style="color:#fbbf24;font-weight:600;font-size:0.72rem;letter-spacing:0.05em;margin-bottom:4px;">⚠ JUDGE: CHANGED</div>
      <div>${esc(ana.summary || '')}</div>
    </div>`;
  }
  return `<div style="padding:8px 11px;background:#1c1212;border-bottom:1px solid #2a2a3a;font-size:0.72rem;color:#f87171;">
    judge: ${esc(ana.verdict)} · ${esc(ana.summary || '').slice(0,200)}
  </div>`;
}

function renderDetail(r) {
  const d = document.getElementById('detail');
  d.innerHTML =
    `<div class="row-flex">` +
      `<div class="card" style="flex:2;">` +
        `<div class="card-header"><span>User Query</span>` +
          `<span class="badge b-server">${esc(r.tags.server)}</span></div>` +
        `<div class="card-body">${esc(r.user_query)}</div>` +
      `</div>` +
      `<div class="card" style="flex:3;">` +
        `<div class="card-header"><span>Poisoned Tool</span>` +
          `<span class="badge b-risk">${esc(r.tags.security_risk)}</span></div>` +
        `<div class="card-body hl-poison">${esc(r.extra.poisoned_tool || '—')}</div>` +
      `</div>` +
    `</div>` +
    `<div class="card" id="steeringCard">` +
      `<div class="card-header">` +
        `<span>Steering Effects · side-by-side across α</span>` +
        `<span style="color:#666;font-size:0.66rem;">green=resistant · red=compliant · yellow=mixed</span>` +
      `</div>` +
      `<div id="steeringInner">${renderSteering(r.id)}</div>` +
    `</div>` +
    `<div class="row-flex">` +
      `<div class="card">` +
        `<div class="card-header"><span>y_pos — attack-compliant (bad)</span>` +
          `<span class="badge b-model">${esc(r.tags.y_pos_source_model || '')}</span></div>` +
        `<div class="card-body hl-pos">${esc(r.y_pos)}</div>` +
      `</div>` +
      `<div class="card">` +
        `<div class="card-header"><span>y_neg — attack-resistant (good)</span>` +
          `<span class="badge b-model">${esc(r.tags.y_neg_source_model || '')}</span></div>` +
        `<div class="card-body hl-neg">${esc(r.y_neg)}</div>` +
      `</div>` +
    `</div>` +
    `<div class="card">` +
      `<div class="card-header">System Prompt</div>` +
      `<div class="card-body">${esc(r.system_prompt)}</div>` +
    `</div>`;

  wireSweepSel(r);
}

function wireSweepSel(r) {
  const sel = document.getElementById('sweepSel');
  if (!sel) return;
  sel.onchange = () => {
    selectedSweep = sel.value;
    document.getElementById('steeringInner').innerHTML = renderSteering(r.id);
    wireSweepSel(r);
  };
}

/* ----- stats page ----- */
// Compute per-sweep × per-α mean reward, optionally restricted to a paradigm.
function statsForSweep(sweep, paradigm) {
  const perAlpha = {};
  Object.entries(BY_PAIR).forEach(([pid, entries]) => {
    const r = ROW_BY_ID[pid];
    if (paradigm && r && r.tags.paradigm !== paradigm) return;
    entries.filter(e => e.sweep === sweep).forEach(e => {
      if (!perAlpha[e.alpha]) perAlpha[e.alpha] = [];
      if (e.reward != null) perAlpha[e.alpha].push(e.reward);
    });
  });
  return Object.entries(perAlpha).map(([a, rs]) => ({
    alpha: parseFloat(a),
    n: rs.length,
    mean_reward: rs.length ? rs.reduce((x,y)=>x+y,0)/rs.length : null,
  })).sort((a,b) => a.alpha - b.alpha);
}

let statsParadigm = '';   // '' = all
let crossSel = null;       // Set of selected sweep names for the comparison table

function renderStats() {
  const cont = document.getElementById('statsPage');
  const allSweeps = Object.keys(STATS).sort();
  if (crossSel === null) crossSel = new Set(allSweeps);  // default: all

  const paradigms = [...new Set(ROWS.map(r => r.tags.paradigm))].sort();

  // === Cross-sweep α-heatmap ===
  const selSweeps = allSweeps.filter(sw => crossSel.has(sw));
  const perSweep = Object.fromEntries(selSweeps.map(sw => [sw, statsForSweep(sw, statsParadigm)]));
  const allAlphas = [...new Set(selSweeps.flatMap(sw => perSweep[sw].map(r => r.alpha)))].sort((a,b) => a - b);

  const headerCells = selSweeps.map(sw => {
    const meta = SWEEP_META[sw] || {};
    const layer = meta.layer != null ? (Array.isArray(meta.layer) ? meta.layer.join('/') : meta.layer) : '';
    return `<th style="background:#20202e;padding:6px 8px;font-size:0.66rem;color:#aaa;text-align:center;font-weight:600;writing-mode:vertical-lr;transform:rotate(180deg);min-width:32px;height:140px;border:1px solid #0f0f13;font-family:ui-monospace,monospace;">
      ${esc(sw)}${layer?` · L${layer}`:''}
    </th>`;
  }).join('');

  const bodyRows = allAlphas.map(a => {
    const cells = selSweeps.map(sw => {
      const row = perSweep[sw].find(r => r.alpha === a);
      if (!row || row.mean_reward == null) {
        return `<td style="background:#15151e;color:#444;text-align:center;font-size:0.7rem;border:1px solid #0f0f13;width:62px;height:30px;">—</td>`;
      }
      const rw = row.mean_reward;
      const bg = rewardCellColor(rw);
      const fg = (rw >= 0.4 && rw <= 0.7) ? '#1a1a24' : '#fff';
      return `<td title="α=${a} · sweep=${sw} · n=${row.n}" style="background:${bg};color:${fg};text-align:center;font-family:ui-monospace,monospace;font-size:0.72rem;font-weight:600;border:1px solid #0f0f13;width:62px;height:30px;">${rw.toFixed(2)}<br><span style="font-size:0.6rem;opacity:0.7;font-weight:400;">n${row.n}</span></td>`;
    }).join('');
    return `<tr>
      <td style="font-family:ui-monospace,monospace;font-size:0.78rem;font-weight:700;color:#c084fc;padding:5px 12px;background:#20202e;text-align:right;border:1px solid #0f0f13;">α=${fmtAlpha(a)}</td>
      ${cells}
    </tr>`;
  }).join('');

  const sweepChips = allSweeps.map(sw => {
    const on = crossSel.has(sw);
    const layer = (SWEEP_META[sw] && SWEEP_META[sw].layer != null)
      ? `L${Array.isArray(SWEEP_META[sw].layer) ? SWEEP_META[sw].layer.join('/') : SWEEP_META[sw].layer}` : '';
    return `<button class="sw-chip" data-sw="${esc(sw)}" style="padding:3px 9px;border-radius:12px;border:1px solid ${on?'#a78bfa':'#3a3a4a'};background:${on?'#3a2060':'#252535'};color:${on?'#fff':'#888'};font-size:0.7rem;cursor:pointer;font-family:ui-monospace,monospace;">${esc(sw)}${layer?` · ${layer}`:''}</button>`;
  }).join(' ');

  const paradigmOpts = ['', ...paradigms].map(p =>
    `<option value="${esc(p)}"${p === statsParadigm ? ' selected' : ''}>${p ? esc(p) : 'all paradigms'}</option>`
  ).join('');

  // === Per-sweep tables (with paradigm filter applied) ===
  const perSweepHtml = allSweeps.map(sw => {
    const rowsData = statsForSweep(sw, statsParadigm);
    if (!rowsData.length) {
      return `<div class="stats-sweep"><div class="stats-sweep-hdr"><h2>${esc(sw)}</h2><span class="meta">no rows for paradigm filter</span></div></div>`;
    }
    const meta = SWEEP_META[sw] || {};
    const layer = meta.layer != null ? (Array.isArray(meta.layer) ? meta.layer.join('/') : meta.layer) : '';
    const totalN = rowsData.reduce((x,r)=>x+r.n, 0);
    const rows = rowsData.map(row => {
      const rw = row.mean_reward;
      const rwBar = rw == null ? '' : `<div class="bar"><div class="bar-fill ${rw>=0.7?'bf-good':rw>=0.3?'bf-mid':'bf-bad'}" style="width:${(rw*100).toFixed(0)}%"></div></div>`;
      return `<tr>
        <td class="alpha">α=${fmtAlpha(row.alpha)}</td>
        <td>${row.n}</td>
        <td><div class="bar-cell">${rwBar}<span>${rw==null?'—':rw.toFixed(3)}</span></div></td>
      </tr>`;
    }).join('');
    return `
      <div class="stats-sweep">
        <div class="stats-sweep-hdr">
          <h2>${esc(sw)}${layer?` · L${layer}`:''}</h2>
          <span class="meta">${rowsData.length} α · ${totalN} rows${statsParadigm?` · paradigm=${esc(statsParadigm)}`:''}</span>
        </div>
        <table class="stats">
          <thead><tr><th>α</th><th>n</th><th>mean reward</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }).join('');

  cont.innerHTML = `
    <div style="max-width:1400px;margin:0 auto;">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;flex-wrap:wrap;">
        <h2 style="font-size:1rem;color:#a78bfa;">Cross-sweep α comparison</h2>
        <label style="font-size:0.74rem;color:#888;">paradigm
          <select id="statsParadigm" style="margin-left:4px;background:#252535;border:1px solid #3a3a4a;color:#e0e0e0;border-radius:5px;padding:3px 7px;font-size:0.75rem;">${paradigmOpts}</select>
        </label>
      </div>
      <div style="font-size:0.74rem;color:#888;margin-bottom:8px;">
        Click a sweep chip to toggle inclusion. Each cell = mean reward at that (α, sweep), filtered to the selected paradigm.
        Same row across sweeps = direct comparison at that α.
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;">
        ${sweepChips}
      </div>
      <div style="overflow-x:auto;background:#1a1a24;border:1px solid #2a2a3a;border-radius:8px;padding:10px;margin-bottom:24px;">
        <table style="border-collapse:collapse;">
          <thead><tr>
            <th style="background:#20202e;padding:6px 12px;font-size:0.7rem;color:#888;text-align:right;width:80px;">α</th>
            ${headerCells}
          </tr></thead>
          <tbody>${bodyRows || '<tr><td colspan="99" style="padding:20px;text-align:center;color:#666;">No alphas in selection.</td></tr>'}</tbody>
        </table>
      </div>

      <h2 style="font-size:1rem;color:#a78bfa;margin-bottom:10px;">Per-sweep detail</h2>
      ${perSweepHtml || '<div class="no-results">No sweep data found.</div>'}
    </div>
  `;

  document.getElementById('statsParadigm').addEventListener('change', (e) => {
    statsParadigm = e.target.value; renderStats();
  });
  document.querySelectorAll('.sw-chip').forEach(b => {
    b.onclick = () => {
      const sw = b.dataset.sw;
      if (crossSel.has(sw)) crossSel.delete(sw); else crossSel.add(sw);
      renderStats();
    };
  });
}

/* ----- sweeps tab ----- */
function rewardCellColor(r) {
  if (r == null) return '#252535';
  // gradient red(0) → yellow(0.5) → green(1)
  if (r <= 0.5) {
    const t = r / 0.5;
    const r_ = Math.round(248 + (251-248)*t);
    const g_ = Math.round(113 + (191-113)*t);
    const b_ = Math.round(113 + (36-113)*t);
    return `rgb(${r_},${g_},${b_})`;
  }
  const t = (r - 0.5) / 0.5;
  const r_ = Math.round(251 + (74-251)*t);
  const g_ = Math.round(191 + (222-191)*t);
  const b_ = Math.round(36 + (128-36)*t);
  return `rgb(${r_},${g_},${b_})`;
}

function renderHeatmap(sweep, det) {
  if (!det.eval_pairs.length) return '<div style="padding:14px;color:#555;font-size:0.8rem;">No eval rows matched for this sweep.</div>';
  const alphas = det.alphas;
  const headerCells = alphas.map(a => `<th style="font-family:ui-monospace,monospace;padding:4px 6px;font-size:0.7rem;color:#888;font-weight:600;text-align:center;">α${a>=0?'+':''}${a}</th>`).join('');
  const bodyRows = det.eval_pairs.map(row => {
    const r = ROW_BY_ID[row.pid];
    const label = r ? `<td style="padding:3px 8px;font-family:ui-monospace,monospace;font-size:0.7rem;color:#aaa;white-space:nowrap;cursor:pointer;" onclick="openPair('${esc(row.pid)}','${esc(sweep)}')">${esc(row.pid)}<br><span style="color:#666;font-size:0.66rem;">${esc(r.tags.server)} · ${esc(r.tags.security_risk)}</span></td>`
                  : `<td style="padding:3px 8px;font-family:ui-monospace,monospace;font-size:0.7rem;color:#666;">${esc(row.pid)}</td>`;
    const cells = row.rewards.map(rw => {
      const bg = rewardCellColor(rw);
      const txt = rw == null ? '—' : rw.toFixed(2);
      const fg = (rw == null) ? '#666' : (rw >= 0.4 && rw <= 0.7 ? '#1a1a24' : '#fff');
      return `<td style="padding:0;text-align:center;font-family:ui-monospace,monospace;font-size:0.66rem;background:${bg};color:${fg};border:1px solid #0f0f13;min-width:42px;height:24px;">${txt}</td>`;
    }).join('');
    return `<tr>${label}${cells}</tr>`;
  }).join('');
  return `<div style="overflow-x:auto;padding:8px;">
    <table style="border-collapse:collapse;width:auto;">
      <thead><tr><th style="background:#20202e;padding:6px 8px;font-size:0.7rem;color:#888;text-align:left;">pair · server / risk</th>${headerCells}</tr></thead>
      <tbody>${bodyRows}</tbody>
    </table>
  </div>`;
}

function renderTrainList(det) {
  if (!det.train_pair_ids || !det.train_pair_ids.length) {
    return `<div style="padding:14px;color:#555;font-size:0.8rem;">No training-pair index found${det.train_acts_dir ? ` at outputs/acts/${det.train_acts_dir}/index.jsonl` : ''}.</div>`;
  }
  const ids = det.train_pair_ids;
  // tally tags
  const byServer = {};
  const byRisk = {};
  const byParadigm = {};
  ids.forEach(id => {
    const r = ROW_BY_ID[id];
    if (!r) return;
    byServer[r.tags.server] = (byServer[r.tags.server] || 0) + 1;
    byRisk[r.tags.security_risk] = (byRisk[r.tags.security_risk] || 0) + 1;
    byParadigm[r.tags.paradigm] = (byParadigm[r.tags.paradigm] || 0) + 1;
  });
  const tally = (obj) => Object.entries(obj).sort((a,b)=>b[1]-a[1])
    .map(([k,v]) => `<span style="display:inline-block;background:#252535;padding:2px 8px;border-radius:10px;margin:2px;font-size:0.7rem;"><span style="color:#aaa">${esc(k)}</span> <span style="color:#fbbf24;font-family:ui-monospace,monospace;">${v}</span></span>`).join('');
  return `<div style="padding:10px 12px;">
    <div style="font-size:0.78rem;color:#888;margin-bottom:8px;">${ids.length} pairs from <code style="color:#c084fc;">outputs/acts/${esc(det.train_acts_dir)}/index.jsonl</code></div>
    <div style="margin-bottom:8px;"><b style="color:#777;font-size:0.72rem;">by server: </b>${tally(byServer)}</div>
    <div style="margin-bottom:8px;"><b style="color:#777;font-size:0.72rem;">by risk: </b>${tally(byRisk)}</div>
    <div style="margin-bottom:10px;"><b style="color:#777;font-size:0.72rem;">by paradigm: </b>${tally(byParadigm)}</div>
    <details><summary style="cursor:pointer;color:#aaa;font-size:0.78rem;padding:4px 0;">show all ${ids.length} pair IDs</summary>
      <div style="max-height:300px;overflow-y:auto;padding:8px;background:#15151e;border-radius:4px;font-family:ui-monospace,monospace;font-size:0.7rem;line-height:1.6;color:#888;">
        ${ids.map(id => ROW_BY_ID[id]
          ? `<div style="cursor:pointer;padding:1px 2px;" onclick="openPair('${esc(id)}','')"><span style="color:#aaa;">${esc(id)}</span></div>`
          : `<div style="color:#555;">${esc(id)} <span style="color:#444;">(not in current pair set)</span></div>`).join('')}
      </div>
    </details>
  </div>`;
}

function renderSweeps() {
  const cont = document.getElementById('sweepsPage');
  const sweepNames = [...new Set([
    ...Object.keys(STATS),
    ...Object.keys(SWEEP_META),
    ...Object.keys(SWEEP_DETAILS),
    ...Object.values(BY_PAIR).flat().map(e => e.sweep),
  ])].sort();
  cont.innerHTML = `<div style="max-width:1100px;margin:0 auto;">
    <h2 style="font-size:1rem;color:#a78bfa;margin-bottom:6px;">Sweep configurations</h2>
    <div style="font-size:0.78rem;color:#888;margin-bottom:14px;">
      What each sweep means — model, intervention, α range, and which examples were used (training + eval).
    </div>
    ${sweepNames.map(sw => {
      const m = SWEEP_META[sw] || {};
      const s = STATS[sw];
      const det = SWEEP_DETAILS[sw];
      const totalRows = s ? s.rows.reduce((a,r)=>a+r.n,0) : 0;
      const alphaRange = s && s.rows.length ?
        `[${s.rows[0].alpha}, ${s.rows[s.rows.length-1].alpha}]` : '—';
      const evalCount = det ? det.eval_pairs.length : 0;
      const trainCount = det ? det.train_pair_ids.length : 0;
      return `<div class="stats-sweep">
        <div class="stats-sweep-hdr">
          <h2>${esc(sw)}</h2>
          <span class="meta">${s ? `${s.n_pairs} pairs · ${s.rows.length} α · ${totalRows} rows · α∈${alphaRange}` : 'no eval rows matched'}</span>
        </div>
        <div style="padding:12px 14px;font-size:0.82rem;line-height:1.6;color:#ccc;">
          ${m.model ? `<div><b style="color:#777;">model:</b> <span style="color:#c084fc;font-family:ui-monospace,monospace;">${esc(m.model)}</span></div>`:''}
          ${m.intervention ? `<div><b style="color:#777;">intervention:</b> ${esc(m.intervention)}</div>`:''}
          ${m.layer != null ? `<div><b style="color:#777;">layer:</b> L${m.layer}</div>`:''}
          ${m.alphas ? `<div><b style="color:#777;">α tested:</b> <span style="font-family:ui-monospace,monospace;">${esc(m.alphas)}</span></div>`:''}
          ${m.n_examples ? `<div><b style="color:#777;">N examples:</b> ${m.n_examples}</div>`:''}
          ${m.filter ? `<div><b style="color:#777;">filter:</b> ${esc(m.filter)}</div>`:''}
          ${m.judge ? `<div><b style="color:#777;">judge:</b> <span style="font-family:ui-monospace,monospace;font-size:0.74rem;">${esc(m.judge)}</span></div>`:''}
          ${m.notes ? `<div style="margin-top:8px;padding:8px 10px;background:#15151e;border-left:3px solid #a78bfa;border-radius:3px;color:#ddd;font-style:italic;">${esc(m.notes)}</div>`:''}
          ${!Object.keys(m).length ? `<div style="color:#555;font-style:italic;">No annotation yet.</div>`:''}
        </div>
        <details style="border-top:1px solid #2a2a3a;">
          <summary style="cursor:pointer;padding:9px 14px;background:#1d1d2a;font-size:0.78rem;color:#aaa;">▸ Eval examples (${evalCount} pairs · per-α reward heatmap)</summary>
          ${det ? renderHeatmap(sw, det) : ''}
        </details>
        <details style="border-top:1px solid #2a2a3a;">
          <summary style="cursor:pointer;padding:9px 14px;background:#1d1d2a;font-size:0.78rem;color:#aaa;">▸ Training examples (${trainCount} pairs from outputs/acts/${esc(det && det.train_acts_dir || '?')})</summary>
          ${det ? renderTrainList(det) : ''}
        </details>
      </div>`;
    }).join('')}
  </div>`;
}

/* ----- rollouts tab ----- */
let rolFilter = 'all';

function gradeColor(g) {
  if (g === 'A') return { bg: '#14401a', fg: '#4ade80' };  // resist
  if (g === 'B') return { bg: '#401414', fg: '#f87171' };  // comply
  if (g === 'C') return { bg: '#403014', fg: '#fbbf24' };  // ambiguous / borderline
  return { bg: '#252535', fg: '#888' };
}
function labelColor(l) {
  if (l === 'resist') return { bg: '#14401a', fg: '#4ade80' };
  if (l === 'comply') return { bg: '#401414', fg: '#f87171' };
  if (l === 'ambiguous' || l === 'borderline') return { bg: '#403014', fg: '#fbbf24' };
  return { bg: '#252535', fg: '#888' };
}

function renderRolloutCard(r, idx) {
  const lab = r.label || (r.tags && r.tags.label) || 'unknown';
  const lc = labelColor(lab);
  const grade = (r.extra && r.extra.judge_grade) || (r.tags && r.tags.judge_grade) || '';
  const gc = gradeColor(grade);
  const judgeRaw = (r.extra && r.extra.judge_raw) || '';
  const tags = r.tags || {};
  const rollout = (lab === 'comply') ? r.y_pos : (lab === 'resist') ? r.y_neg : (r.y_pos || r.y_neg || '');
  const isBorderline = lab === 'ambiguous' || lab === 'borderline' || grade === 'C';

  return `<div class="stats-sweep" style="${isBorderline ? 'border:1px solid #fbbf24;' : ''}">
    <div class="stats-sweep-hdr" style="${isBorderline ? 'background:#2a2415;' : ''}">
      <div style="display:flex;gap:8px;align-items:center;">
        <span style="background:${lc.bg};color:${lc.fg};font-family:ui-monospace,monospace;font-size:0.72rem;padding:3px 9px;border-radius:10px;font-weight:600;">${esc(lab.toUpperCase())}</span>
        ${grade ? `<span style="background:${gc.bg};color:${gc.fg};font-family:ui-monospace,monospace;font-size:0.7rem;padding:2px 7px;border-radius:8px;">judge ${esc(grade)}</span>`:''}
        ${isBorderline ? '<span style="color:#fbbf24;font-size:0.72rem;font-weight:600;">⚠ BORDERLINE</span>' : ''}
        <span style="font-family:ui-monospace,monospace;font-size:0.7rem;color:#888;">${esc(r.id || ('row '+idx))}</span>
      </div>
      <div style="display:flex;gap:4px;flex-wrap:wrap;">
        ${tags.server ? `<span class="badge b-server">${esc(tags.server)}</span>`:''}
        ${tags.security_risk ? `<span class="badge b-risk">${esc(tags.security_risk)}</span>`:''}
        ${tags.paradigm ? `<span class="badge b-paradigm">${esc(tags.paradigm)}</span>`:''}
      </div>
    </div>
    <div style="padding:10px 14px;font-size:0.78rem;line-height:1.5;color:#ccc;">
      <div style="margin-bottom:6px;"><b style="color:#777;font-size:0.7rem;">USER QUERY:</b> ${esc(r.user_query || '')}</div>
      ${(r.extra && r.extra.poisoned_tool) ? `<details style="margin-bottom:6px;"><summary style="cursor:pointer;color:#aaa;font-size:0.74rem;">▸ poisoned tool</summary><pre style="background:#15151e;border-left:3px solid #fbbf24;padding:8px 10px;margin-top:4px;border-radius:4px;font-family:ui-monospace,monospace;font-size:0.72rem;color:#ddd;white-space:pre-wrap;max-height:200px;overflow-y:auto;">${esc(r.extra.poisoned_tool)}</pre></details>`:''}
      ${rollout ? `<details open><summary style="cursor:pointer;color:#aaa;font-size:0.74rem;">▸ rollout (${(rollout.length).toLocaleString()} chars)</summary><pre style="background:#15151e;border-left:3px solid ${lc.fg};padding:9px 11px;margin-top:4px;border-radius:4px;font-family:ui-monospace,monospace;font-size:0.74rem;color:#ddd;white-space:pre-wrap;max-height:340px;overflow-y:auto;line-height:1.55;">${esc(rollout)}</pre></details>`:'<div style="color:#555;font-style:italic;">(rollout text empty)</div>'}
      ${judgeRaw ? `<details style="margin-top:6px;"><summary style="cursor:pointer;color:#aaa;font-size:0.74rem;">▸ judge reasoning</summary><pre style="background:#15151e;padding:8px 10px;margin-top:4px;border-radius:4px;font-family:ui-monospace,monospace;font-size:0.72rem;color:#aaa;white-space:pre-wrap;max-height:240px;overflow-y:auto;">${esc(judgeRaw)}</pre></details>`:''}
    </div>
  </div>`;
}

function renderRollouts() {
  const cont = document.getElementById('rolloutsPage');
  const all = LABELLED.length ? LABELLED : [];
  const counts = {
    all: all.length,
    resist: all.filter(r => (r.label || (r.tags && r.tags.label)) === 'resist').length,
    comply: all.filter(r => (r.label || (r.tags && r.tags.label)) === 'comply').length,
    borderline: all.filter(r => {
      const l = r.label || (r.tags && r.tags.label);
      const g = (r.extra && r.extra.judge_grade) || (r.tags && r.tags.judge_grade);
      return l === 'ambiguous' || l === 'borderline' || g === 'C';
    }).length,
  };
  document.getElementById('rolBadge').textContent = counts.borderline || '';

  let filtered;
  if (rolFilter === 'all') filtered = all;
  else if (rolFilter === 'borderline') filtered = all.filter(r => {
    const l = r.label || (r.tags && r.tags.label);
    const g = (r.extra && r.extra.judge_grade) || (r.tags && r.tags.judge_grade);
    return l === 'ambiguous' || l === 'borderline' || g === 'C';
  });
  else filtered = all.filter(r => (r.label || (r.tags && r.tags.label)) === rolFilter);

  // borderline-first sort
  if (rolFilter === 'all') {
    filtered = [...filtered].sort((a, b) => {
      const ga = (a.extra && a.extra.judge_grade) || (a.tags && a.tags.judge_grade) || '';
      const gb = (b.extra && b.extra.judge_grade) || (b.tags && b.tags.judge_grade) || '';
      const ra = ga === 'C' ? 0 : (ga === 'B' ? 1 : 2);
      const rb = gb === 'C' ? 0 : (gb === 'B' ? 1 : 2);
      return ra - rb;
    });
  }

  if (!all.length) {
    cont.innerHTML = `<div style="max-width:780px;margin:60px auto;color:#777;font-size:0.9rem;line-height:1.6;">
      <h2 style="color:#a78bfa;font-size:1rem;margin-bottom:10px;">No labelled rollouts</h2>
      <p>Expected at <code>diffmean/outputs/qwen3_rollouts.labelled.jsonl</code> — generate with the rollout script.</p>
    </div>`;
    return;
  }

  const meta = all[0] && all[0].extra ? all[0].extra : {};
  const borderlineExtra = BORDERLINE.length
    ? `<div style="background:#1f1c12;border:1px solid #fbbf24;border-radius:6px;padding:10px 14px;margin-bottom:12px;font-size:0.8rem;color:#fde68a;">
        <b>Borderline demo cases:</b> ${BORDERLINE.length} cases curated by the live picker
        <span style="color:#888;">(loaded from <code>outputs/eval/borderline_demo.jsonl</code>)</span>
      </div>` : '';

  cont.innerHTML = `<div style="max-width:1100px;margin:0 auto;">
    <h2 style="font-size:1rem;color:#a78bfa;margin-bottom:6px;">Labelled rollouts — ${all.length} total</h2>
    <div style="font-size:0.78rem;color:#888;margin-bottom:6px;">
      Source: <code>outputs/qwen3_rollouts.labelled.jsonl</code>
      ${meta.rollout_model ? ` · rollout model: <span style="color:#c084fc;font-family:ui-monospace,monospace;">${esc(meta.rollout_model)}</span>`:''}
      ${meta.judge_model ? ` · judge: <span style="color:#c084fc;font-family:ui-monospace,monospace;">${esc(meta.judge_model)}</span>`:''}
      ${meta.temperature != null ? ` · T=${meta.temperature}`:''}
    </div>
    <div style="font-size:0.74rem;color:#666;margin-bottom:14px;">
      Borderline = label ∈ {ambiguous, borderline} OR judge_grade = C. These are the cases where the
      judge couldn't confidently classify resist vs comply — the most informative for activation analysis.
    </div>
    ${borderlineExtra}
    <div style="display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap;">
      <button class="rol-pill" data-rf="all"        style="${rolFilter==='all'        ? 'background:#3a2060;border-color:#a78bfa;color:#fff;' : ''}">All <span style="color:#888;">${counts.all}</span></button>
      <button class="rol-pill" data-rf="resist"     style="${rolFilter==='resist'     ? 'background:#14401a;border-color:#4ade80;color:#fff;' : ''}">Resist <span style="color:#888;">${counts.resist}</span></button>
      <button class="rol-pill" data-rf="comply"     style="${rolFilter==='comply'     ? 'background:#401414;border-color:#f87171;color:#fff;' : ''}">Comply <span style="color:#888;">${counts.comply}</span></button>
      <button class="rol-pill" data-rf="borderline" style="${rolFilter==='borderline' ? 'background:#403014;border-color:#fbbf24;color:#fff;' : ''}">⚠ Borderline <span style="color:#888;">${counts.borderline}</span></button>
    </div>
    ${filtered.map((r, i) => renderRolloutCard(r, i)).join('') || '<div class="no-results">No rollouts in this filter.</div>'}
  </div>`;

  document.querySelectorAll('.rol-pill').forEach(b => {
    b.onclick = () => { rolFilter = b.dataset.rf; renderRollouts(); };
  });
}

/* ----- stability tab ----- */
function parseStabilityLog(text) {
  // Lines like:  +2.0  1/3  0/3  0/3  3/3  3/3  3/3
  // or          -1.0  ...  ...
  const lines = text.split('\n');
  const rowRe = /^\s*([+-]?\d+(?:\.\d+)?)\s+((?:\d+\/\d+\s*)+)$/;
  const rows = [];
  let maxCols = 0;
  for (const line of lines) {
    const m = line.match(rowRe);
    if (!m) continue;
    const alpha = parseFloat(m[1]);
    const cells = m[2].trim().split(/\s+/).map(c => {
      const [num, den] = c.split('/').map(Number);
      return (Number.isFinite(num) && Number.isFinite(den) && den > 0)
        ? { num, den, ratio: num / den }
        : null;
    });
    rows.push({ alpha, cells });
    if (cells.length > maxCols) maxCols = cells.length;
  }
  rows.sort((a, b) => a.alpha - b.alpha);
  return { rows, nCases: maxCols };
}

function renderStabilityHeatmap(parsed) {
  if (!parsed.rows.length) {
    return '<div style="color:#666;padding:20px;font-size:0.85rem;">No table rows parsed. Expected lines like <code>+2.0  1/3  0/3  ...</code></div>';
  }
  const headerCells = Array.from({length: parsed.nCases}, (_, i) =>
    `<th style="font-family:ui-monospace,monospace;padding:5px 8px;font-size:0.72rem;color:#888;font-weight:600;text-align:center;background:#20202e;">case ${i+1}</th>`
  ).join('');

  // per-row average
  const rowAvgs = parsed.rows.map(r => {
    const valid = r.cells.filter(c => c);
    return valid.length ? valid.reduce((a,c)=>a+c.ratio,0)/valid.length : null;
  });
  // per-column average
  const colAvgs = Array.from({length: parsed.nCases}, (_, i) => {
    const vals = parsed.rows.map(r => r.cells[i]).filter(c => c);
    return vals.length ? vals.reduce((a,c)=>a+c.ratio,0)/vals.length : null;
  });

  const bodyRows = parsed.rows.map((r, ri) => {
    const alphaSign = r.alpha >= 0 ? '+' : '';
    const cells = Array.from({length: parsed.nCases}, (_, i) => {
      const c = r.cells[i];
      if (!c) return `<td style="background:#15151e;color:#444;border:1px solid #0f0f13;text-align:center;padding:0;width:62px;height:32px;">—</td>`;
      const bg = rewardCellColor(c.ratio);
      const fg = (c.ratio >= 0.4 && c.ratio <= 0.7) ? '#1a1a24' : '#fff';
      return `<td title="${c.num}/${c.den}" style="background:${bg};color:${fg};font-family:ui-monospace,monospace;font-size:0.78rem;font-weight:600;text-align:center;padding:0;width:62px;height:32px;border:1px solid #0f0f13;">${c.num}/${c.den}</td>`;
    }).join('');
    const avgR = rowAvgs[ri];
    const avgBg = avgR == null ? '#252535' : rewardCellColor(avgR);
    const avgFg = (avgR != null && avgR >= 0.4 && avgR <= 0.7) ? '#1a1a24' : '#fff';
    return `<tr>
      <td style="font-family:ui-monospace,monospace;font-size:0.78rem;font-weight:700;color:#c084fc;padding:6px 12px;background:#20202e;text-align:right;">α=${alphaSign}${r.alpha}</td>
      ${cells}
      <td style="background:${avgBg};color:${avgFg};font-family:ui-monospace,monospace;font-size:0.74rem;font-weight:700;text-align:center;padding:0;width:62px;height:32px;border-left:2px solid #2a2a3a;">${avgR==null?'—':avgR.toFixed(2)}</td>
    </tr>`;
  }).join('');

  const colAvgRow = `<tr>
    <td style="font-family:ui-monospace,monospace;font-size:0.7rem;color:#888;padding:5px 12px;background:#15151e;text-align:right;">col avg</td>
    ${colAvgs.map(av => {
      const bg = av == null ? '#252535' : rewardCellColor(av);
      const fg = (av != null && av >= 0.4 && av <= 0.7) ? '#1a1a24' : '#fff';
      return `<td style="background:${bg};color:${fg};font-family:ui-monospace,monospace;font-size:0.72rem;font-weight:700;text-align:center;padding:0;width:62px;height:30px;border:1px solid #0f0f13;">${av==null?'—':av.toFixed(2)}</td>`;
    }).join('')}
    <td style="background:#15151e;border-left:2px solid #2a2a3a;"></td>
  </tr>`;

  return `<div style="overflow-x:auto;background:#1a1a24;border:1px solid #2a2a3a;border-radius:8px;padding:10px;">
    <table style="border-collapse:collapse;">
      <thead><tr>
        <th style="background:#20202e;padding:5px 12px;font-size:0.72rem;color:#888;text-align:right;">α</th>
        ${headerCells}
        <th style="background:#20202e;padding:5px 8px;font-size:0.72rem;color:#888;text-align:center;border-left:2px solid #2a2a3a;">row avg</th>
      </tr></thead>
      <tbody>${bodyRows}${colAvgRow}</tbody>
    </table>
    <div style="margin-top:10px;font-size:0.72rem;color:#666;">
      cell = samples that complied / total samples · green = consistently resistant · red = consistently compliant · yellow = unstable
    </div>
  </div>`;
}

function renderStability() {
  const cont = document.getElementById('stabilityPage');
  const initial = (window._stabilityText != null) ? window._stabilityText : BORDER_LOG;
  const parsed = parseStabilityLog(initial);
  const haveLocalLog = BORDER_LOG && BORDER_LOG.trim().length > 0;
  const haveJsonl = BORDERLINE && BORDERLINE.length > 0;

  cont.innerHTML = `<div style="max-width:1100px;margin:0 auto;">
    <h2 style="font-size:1rem;color:#a78bfa;margin-bottom:6px;">Borderline α-stability</h2>
    <div style="font-size:0.78rem;color:#888;margin-bottom:14px;line-height:1.55;">
      For each borderline case (where the baseline judge is uncertain), the stability sweep runs N samples
      per α and records how many comply. A row that's red across all α means steering is fully ineffective on that case;
      a row that flips red→green tells you steering actually shifts behavior at some α.
    </div>

    <div style="font-size:0.74rem;color:${haveLocalLog?'#4ade80':'#888'};margin-bottom:6px;">
      ${haveLocalLog
        ? `✓ loaded <code>outputs/eval/border.log</code> (${BORDER_LOG.length.toLocaleString()} bytes)`
        : '⚠ <code>outputs/eval/border.log</code> not found locally — paste the table below.'}
      ${haveJsonl ? ` · ✓ <code>borderline_demo.jsonl</code> (${BORDERLINE.length} per-sample rows)` : ''}
    </div>

    <div id="stabilityHeatmap" style="margin-bottom:18px;">${renderStabilityHeatmap(parsed)}</div>

    <details ${haveLocalLog ? '' : 'open'} style="margin-bottom:18px;">
      <summary style="cursor:pointer;color:#aaa;font-size:0.82rem;padding:6px 0;">▸ paste / edit the stability log</summary>
      <textarea id="stabilityInput" rows="12" placeholder="paste rows like:&#10;+2.0  1/3  0/3  0/3  3/3  3/3  3/3" style="width:100%;background:#15151e;border:1px solid #2a2a3a;border-radius:6px;color:#ddd;font-family:ui-monospace,monospace;font-size:0.78rem;padding:10px;line-height:1.5;margin-top:6px;">${esc(initial)}</textarea>
      <div style="margin-top:6px;font-size:0.7rem;color:#666;">parses lines matching <code>±α  X/Y  X/Y  …</code> · header / blank / non-matching lines are ignored · re-renders as you type</div>
    </details>

    ${haveJsonl ? renderBorderlineCases() : ''}
  </div>`;

  const ta = document.getElementById('stabilityInput');
  if (ta) {
    ta.addEventListener('input', () => {
      window._stabilityText = ta.value;
      const p = parseStabilityLog(ta.value);
      document.getElementById('stabilityHeatmap').innerHTML = renderStabilityHeatmap(p);
    });
  }
}

function renderBorderlineCases() {
  if (!BORDERLINE.length) return '';
  // group by case_id if present
  const byCase = {};
  BORDERLINE.forEach(r => {
    const k = (r.case_id != null) ? `case ${r.case_id}` : (r.id || 'case');
    (byCase[k] = byCase[k] || []).push(r);
  });
  return `<div class="stats-sweep" style="margin-top:18px;">
    <div class="stats-sweep-hdr">
      <h2 style="font-size:0.9rem;">Per-sample borderline detail</h2>
      <span class="meta">${BORDERLINE.length} sample rows · ${Object.keys(byCase).length} cases</span>
    </div>
    <div style="padding:0 14px;">
      ${Object.keys(byCase).sort().map(k => {
        const rows = byCase[k];
        const first = rows[0];
        return `<details style="border-top:1px solid #2a2a3a;padding:8px 0;">
          <summary style="cursor:pointer;color:#ddd;font-size:0.82rem;padding:4px 0;">${esc(k)} <span style="color:#888;">(${rows.length} samples)</span></summary>
          <div style="font-family:ui-monospace,monospace;font-size:0.74rem;color:#aaa;padding:8px 0;line-height:1.6;">
            ${first.user_query ? `<div><b style="color:#777;">query:</b> ${esc(first.user_query.slice(0,200))}</div>`:''}
            ${first.poisoned_tool ? `<div><b style="color:#777;">poison:</b> ${esc((first.poisoned_tool||'').slice(0,200))}…</div>`:''}
            <div style="margin-top:8px;display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px;">
              ${rows.map(r => {
                const verdict = r.verdict || r.label || (r.complied != null ? (r.complied?'comply':'resist') : '?');
                const c = labelColor(verdict);
                return `<div style="background:${c.bg};color:${c.fg};padding:4px 8px;border-radius:4px;font-size:0.7rem;">α${r.alpha>=0?'+':''}${r.alpha} · sample ${r.sample!=null?r.sample:'?'} · ${esc(String(verdict))}</div>`;
              }).join('')}
            </div>
          </div>
        </details>`;
      }).join('')}
    </div>
  </div>`;
}

/* ----- highlights tab ----- */
function renderHighlights() {
  const cont = document.getElementById('highlightsPage');
  const items = Object.values(ANALYSIS);
  const changed = items.filter(x => x.verdict === 'CHANGED');
  const noChange = items.filter(x => x.verdict === 'NO_CHANGE').length;
  const errored  = items.filter(x => x.verdict !== 'CHANGED' && x.verdict !== 'NO_CHANGE').length;
  document.getElementById('hlBadge').textContent = changed.length || '';

  if (!items.length) {
    cont.innerHTML = `<div style="max-width:780px;margin:60px auto;color:#777;font-size:0.9rem;line-height:1.6;">
      <h2 style="color:#a78bfa;font-size:1rem;margin-bottom:10px;">No analysis yet</h2>
      <p>Run the LLM judge to populate this page:</p>
      <pre style="background:#1a1a24;padding:12px;border-radius:6px;border:1px solid #2a2a3a;font-size:0.78rem;color:#ccc;margin-top:10px;">python3 -m diffmean.analyze_alphas --concurrency 6</pre>
      <p style="margin-top:10px;">Output is cached at <code>diffmean/outputs/alpha_analysis.json</code>. Reload this page when done.</p>
    </div>`;
    return;
  }

  // group changed by sweep
  const bySweep = {};
  changed.forEach(c => { (bySweep[c.sweep] = bySweep[c.sweep] || []).push(c); });

  cont.innerHTML = `<div style="max-width:980px;margin:0 auto;">
    <h2 style="font-size:1rem;color:#a78bfa;margin-bottom:6px;">Highlights — pairs where α actually shifted behavior</h2>
    <div style="font-size:0.78rem;color:#888;margin-bottom:14px;">
      gpt-5-mini reviewed ${items.length} (pair × sweep) units.
      <span style="color:#fbbf24;">${changed.length} CHANGED</span>,
      <span style="color:#4ade80;">${noChange} NO_CHANGE</span>${errored ? `, <span style="color:#f87171;">${errored} error</span>` : ''}.
      Only CHANGED units are summarised below.
    </div>
    ${Object.keys(bySweep).sort().map(sw => `
      <div class="stats-sweep">
        <div class="stats-sweep-hdr">
          <h2>${esc(sw)}</h2>
          <span class="meta">${bySweep[sw].length} pairs flagged</span>
        </div>
        <div>
          ${bySweep[sw].map(c => {
            const pair = ROWS.find(r => r.id === c.pair_id);
            const rewards = c.rewards.map((r, i) => {
              const a = c.alphas[i];
              const sign = a >= 0 ? '+' : '';
              const pc = pillClass(r);
              return `<span class="reward-pill ${pc}" style="margin-right:3px;">α${sign}${a}: ${r==null?'—':r.toFixed(2)}</span>`;
            }).join('');
            return `<div style="padding:12px 14px;border-top:1px solid #2a2a3a;cursor:pointer;" onclick="openPair('${esc(c.pair_id)}','${esc(sw)}')">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:5px;">
                <div style="font-family:ui-monospace,monospace;font-size:0.74rem;color:#888;">${esc(c.pair_id)}</div>
                ${pair ? `<div style="display:flex;gap:4px;flex-shrink:0;">
                  <span class="badge b-server">${esc(pair.tags.server)}</span>
                  <span class="badge b-risk">${esc(pair.tags.security_risk)}</span>
                </div>`: ''}
              </div>
              ${pair ? `<div style="font-size:0.82rem;color:#ddd;margin-bottom:6px;">${esc(pair.user_query.slice(0,150))}${pair.user_query.length>150?'…':''}</div>`:''}
              <div style="margin-bottom:6px;">${rewards}</div>
              <div style="font-size:0.82rem;line-height:1.55;color:#fde68a;background:#1f1c12;padding:8px 10px;border-radius:4px;border-left:3px solid #fbbf24;">${esc(c.summary)}</div>
            </div>`;
          }).join('')}
        </div>
      </div>`).join('')}
  </div>`;
}

/* ----- HyperSteer tab ----- */
let HS_VIEW = { kind: 'index' };  // {kind:'index'} | {kind:'run', run} | {kind:'example', run, tag, eid}
                                  // | {kind:'train', name} | {kind:'compare', eid}
let HS_PARQUET_CACHE = {};
let HS_RESULTS_CACHE = {};        // key=run|tag → array of trimmed examples
let HS_FULL_CACHE = {};           // key=run|tag|eid → full row

function hsRewardCellColor(r) {
  if (r == null) return '#252535';
  return rewardCellColor(r);
}

function fmtFactor(f) {
  if (f == null) return '—';
  return (Math.round(f * 100) / 100).toString();
}

function fmtMs(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return ms.toFixed(0) + 'ms';
  if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
  return (ms / 60000).toFixed(1) + 'm';
}

function hsBackBar() {
  return `<div style="margin-bottom:14px;">
    <button onclick="hsGo({kind:'index'})" style="background:#252535;border:1px solid #3a3a4a;color:#aaa;padding:5px 12px;border-radius:5px;font-size:0.74rem;cursor:pointer;">← back to HyperSteer index</button>
  </div>`;
}

function renderHSteerIndex() {
  // Group runs by validity, list training dirs, link to compare-view at top.
  const validRuns = HSTEER_RUNS.filter(r => !r.invalid_reason);
  const invalidRuns = HSTEER_RUNS.filter(r => r.invalid_reason);

  // Variant legend
  const variantRows = Object.entries(HSTEER_VARIANT_KEY).map(([v, desc]) =>
    `<tr><td style="padding:4px 10px;font-family:ui-monospace,monospace;color:#60a5fa;font-weight:600;">${esc(v)}</td><td style="padding:4px 10px;color:#ccc;">${esc(desc)}</td></tr>`
  ).join('');

  // Training dirs table
  const trainTable = HSTEER_TRAIN.map(t => {
    const flags = [
      t.config_exists ? '<span style="color:#4ade80;">cfg</span>' : '<span style="color:#666;">cfg</span>',
      t.parquet_exists ? '<span style="color:#4ade80;">parquet</span>' : '<span style="color:#666;">parquet</span>',
      t.merged_concepts_exists ? '<span style="color:#4ade80;">merged</span>' : '<span style="color:#666;">merged</span>',
      t.ckpt_exists ? '<span style="color:#4ade80;">ckpt</span>' : '<span style="color:#f87171;">no-ckpt</span>',
    ].join(' · ');
    return `<tr style="border-top:1px solid #1e1e2a;">
      <td style="padding:5px 10px;font-family:ui-monospace,monospace;color:#aaa;cursor:pointer;" onclick="hsGo({kind:'train', name:'${esc(t.name)}'})">${esc(t.name)}</td>
      <td style="padding:5px 10px;font-family:ui-monospace,monospace;color:#60a5fa;font-weight:600;">${esc(t.variant || '?')}</td>
      <td style="padding:5px 10px;text-align:right;font-family:ui-monospace,monospace;color:#ccc;">${t.n_concepts}</td>
      <td style="padding:5px 10px;font-size:0.72rem;color:#888;">${flags}</td>
    </tr>`;
  }).join('');

  // Run sections
  function runCard(r) {
    const cfgRows = r.configs.map(c => {
      const ar = c.avg_metrics ? c.avg_metrics.attack_resistance : c.avg_reward;
      const bg = hsRewardCellColor(ar);
      const fg = (ar != null && ar >= 0.4 && ar <= 0.7) ? '#1a1a24' : '#fff';
      const tokFlag = c.max_tokens === 1024
        ? '<span style="color:#fbbf24;font-size:0.66rem;" title="1024 may truncate long deliberations">⚠1k</span>'
        : (c.max_tokens === 2048
          ? '<span style="color:#4ade80;font-size:0.66rem;">2k</span>'
          : `<span style="color:#888;font-size:0.66rem;">${c.max_tokens||'?'}</span>`);
      return `<tr style="border-top:1px solid #1e1e2a;">
        <td style="padding:4px 10px;font-family:ui-monospace,monospace;color:#aaa;cursor:pointer;" onclick="hsGo({kind:'run', run:'${esc(r.run)}', tag:'${esc(c.tag)}'})">${esc(c.tag)}</td>
        <td style="padding:4px 10px;font-family:ui-monospace,monospace;color:#60a5fa;font-weight:600;">${esc(c.variant || '—')}</td>
        <td style="padding:4px 10px;font-family:ui-monospace,monospace;color:#c084fc;text-align:right;">${fmtFactor(c.factor)}</td>
        <td style="padding:4px 10px;text-align:center;">${tokFlag}</td>
        <td style="padding:0;text-align:center;background:${bg};color:${fg};font-family:ui-monospace,monospace;font-size:0.78rem;font-weight:700;width:60px;">${ar==null?'—':ar.toFixed(2)}</td>
        <td style="padding:4px 10px;text-align:right;font-family:ui-monospace,monospace;color:#888;font-size:0.72rem;">${c.n_results}/${c.num_examples||'?'}</td>
      </tr>`;
    }).join('');
    const stamp = r.mtime ? new Date(r.mtime * 1000).toISOString().replace('T',' ').slice(0,19) : '';
    const invalidNotice = r.invalid_reason
      ? `<div style="background:#3d1a1a;color:#fca5a5;padding:6px 11px;font-size:0.74rem;border-bottom:1px solid #5a2020;">⚠ INVALID: ${esc(r.invalid_reason)}</div>`
      : '';
    return `<div class="stats-sweep">
      <div class="stats-sweep-hdr">
        <h2 style="font-family:ui-monospace,monospace;font-size:0.85rem;">${esc(r.run)}</h2>
        <span class="meta">${r.configs.length} configs · ${esc(stamp)}</span>
      </div>
      ${invalidNotice}
      <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">
        <thead><tr style="background:#15151e;">
          <th style="padding:6px 10px;text-align:left;color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">config tag</th>
          <th style="padding:6px 10px;text-align:left;color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">variant</th>
          <th style="padding:6px 10px;text-align:right;color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">factor</th>
          <th style="padding:6px 10px;text-align:center;color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">tokens</th>
          <th style="padding:6px 10px;text-align:center;color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">avg AR</th>
          <th style="padding:6px 10px;text-align:right;color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">n</th>
        </tr></thead>
        <tbody>${cfgRows}</tbody>
      </table>
    </div>`;
  }

  const cont = document.getElementById('hsteerPage');
  cont.innerHTML = `<div style="max-width:1200px;margin:0 auto;">
    <h2 style="font-size:1rem;color:#60a5fa;margin-bottom:6px;">HyperSteer experiments</h2>
    <div style="font-size:0.78rem;color:#888;margin-bottom:14px;">
      Browse training-data and per-example completions for every HyperSteer variant + steering-factor cell.
      Data discovered under <code style="color:#aaa;">axbench/axbench/outputs/{mcp_hsteer_*, eval/vfeval_n50_*}</code>.
    </div>

    <div class="stats-sweep">
      <div class="stats-sweep-hdr">
        <h2 style="font-size:0.9rem;">Cross-run prompt comparison</h2>
        <span class="meta">enter an example_id to see how every run answered it</span>
      </div>
      <div style="padding:10px 14px;display:flex;gap:8px;align-items:center;">
        <label style="font-size:0.78rem;color:#aaa;">example_id (0-49):
          <input type="number" id="hsCmpEid" min="0" max="49" value="0" style="margin-left:6px;width:70px;background:#252535;border:1px solid #3a3a4a;color:#e0e0e0;border-radius:5px;padding:3px 7px;font-size:0.78rem;">
        </label>
        <button onclick="hsCompareGo()" style="background:#1a2d3d;border:1px solid #60a5fa;color:#60a5fa;padding:5px 14px;border-radius:5px;font-size:0.78rem;cursor:pointer;font-weight:600;">→ compare across runs</button>
      </div>
    </div>

    <div class="stats-sweep">
      <div class="stats-sweep-hdr">
        <h2 style="font-size:0.9rem;">Variant key</h2>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">
        <tbody>${variantRows}</tbody>
      </table>
    </div>

    <div class="stats-sweep">
      <div class="stats-sweep-hdr">
        <h2 style="font-size:0.9rem;">Training dirs (${HSTEER_TRAIN.length})</h2>
        <span class="meta">click a row to inspect concepts + a sample of train_data.parquet</span>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">
        <thead><tr style="background:#15151e;">
          <th style="padding:6px 10px;text-align:left;color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">dir</th>
          <th style="padding:6px 10px;text-align:left;color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">variant</th>
          <th style="padding:6px 10px;text-align:right;color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">concepts</th>
          <th style="padding:6px 10px;text-align:left;color:#777;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">artifacts</th>
        </tr></thead>
        <tbody>${trainTable || '<tr><td colspan="4" style="padding:20px;color:#666;text-align:center;">No training dirs found.</td></tr>'}</tbody>
      </table>
    </div>

    <h3 style="font-size:0.9rem;color:#60a5fa;margin:18px 0 10px;">Eval runs — VALID (${validRuns.length})</h3>
    ${validRuns.map(runCard).join('') || '<div class="no-results">No valid runs found.</div>'}

    ${invalidRuns.length ? `<h3 style="font-size:0.9rem;color:#f87171;margin:18px 0 10px;">Eval runs — INVALID (${invalidRuns.length})</h3>${invalidRuns.map(runCard).join('')}` : ''}
  </div>`;
}

function hsGo(view) {
  HS_VIEW = view;
  if (view.kind === 'run') renderHSteerRun(view.run, view.tag);
  else if (view.kind === 'example') renderHSteerExample(view.run, view.tag, view.eid);
  else if (view.kind === 'train') renderHSteerTrain(view.name);
  else if (view.kind === 'compare') renderHSteerCompare(view.eid);
  else renderHSteerIndex();
  document.getElementById('hsteerPage').scrollTop = 0;
}

function hsCompareGo() {
  const v = parseInt(document.getElementById('hsCmpEid').value, 10);
  if (Number.isFinite(v)) hsGo({kind:'compare', eid: v});
}

function renderHSteerTrain(name) {
  const t = HSTEER_TRAIN.find(x => x.name === name);
  const cont = document.getElementById('hsteerPage');
  if (!t) { cont.innerHTML = hsBackBar() + '<div class="no-results">Training dir not found.</div>'; return; }

  const conceptList = t.concepts.map(c => `<tr style="border-top:1px solid #1e1e2a;">
    <td style="padding:4px 10px;font-family:ui-monospace,monospace;color:#888;text-align:right;width:50px;">${esc(String(c.concept_id))}</td>
    <td style="padding:4px 10px;color:#ddd;">${esc(c.concept || '')}</td>
    <td style="padding:4px 10px;font-size:0.7rem;color:#666;">${esc(c.ref || '')}</td>
  </tr>`).join('');

  cont.innerHTML = hsBackBar() + `<div style="max-width:1200px;margin:0 auto;">
    <h2 style="font-size:1rem;color:#60a5fa;margin-bottom:4px;font-family:ui-monospace,monospace;">${esc(t.name)}</h2>
    <div style="font-size:0.78rem;color:#888;margin-bottom:14px;">
      variant <span style="color:#60a5fa;font-family:ui-monospace,monospace;font-weight:600;">${esc(t.variant || '?')}</span>
      ${t.variant_desc ? ` — <span style="color:#aaa;">${esc(t.variant_desc)}</span>` : ''}
    </div>

    <div class="stats-sweep">
      <div class="stats-sweep-hdr"><h2 style="font-size:0.9rem;">mcp_hypersteer_config.yaml</h2></div>
      <pre id="hsCfg" style="background:#15151e;padding:11px 14px;font-family:ui-monospace,monospace;font-size:0.74rem;color:#ccc;line-height:1.55;white-space:pre-wrap;max-height:400px;overflow-y:auto;">loading…</pre>
    </div>

    <div class="stats-sweep">
      <div class="stats-sweep-hdr"><h2 style="font-size:0.9rem;">Concepts (${t.n_concepts})</h2></div>
      <div style="max-height:300px;overflow-y:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">
          <thead><tr style="background:#15151e;position:sticky;top:0;">
            <th style="padding:6px 10px;text-align:right;color:#777;font-size:0.68rem;">id</th>
            <th style="padding:6px 10px;text-align:left;color:#777;font-size:0.68rem;">concept</th>
            <th style="padding:6px 10px;text-align:left;color:#777;font-size:0.68rem;">ref</th>
          </tr></thead>
          <tbody>${conceptList || '<tr><td colspan="3" style="padding:14px;color:#666;text-align:center;">No concepts.</td></tr>'}</tbody>
        </table>
      </div>
    </div>

    <div class="stats-sweep">
      <div class="stats-sweep-hdr">
        <h2 style="font-size:0.9rem;">train_data.parquet sample</h2>
        <span class="meta" id="hsParquetMeta">loading…</span>
      </div>
      <div id="hsParquet" style="padding:0;"></div>
    </div>
  </div>`;

  // Async loaders
  fetch('/api/hsteer/yaml?train=' + encodeURIComponent(name))
    .then(r => r.text())
    .then(txt => { document.getElementById('hsCfg').textContent = txt; })
    .catch(() => { document.getElementById('hsCfg').textContent = '(failed to load)'; });

  if (HS_PARQUET_CACHE[name]) {
    renderHSParquet(HS_PARQUET_CACHE[name]);
  } else {
    fetch('/api/hsteer/parquet?train=' + encodeURIComponent(name))
      .then(r => r.json())
      .then(j => { HS_PARQUET_CACHE[name] = j; renderHSParquet(j); })
      .catch(e => { document.getElementById('hsParquet').innerHTML = `<div style="padding:14px;color:#f87171;">parquet load failed: ${esc(String(e))}</div>`; });
  }
}

function renderHSParquet(j) {
  const meta = document.getElementById('hsParquetMeta');
  const cont = document.getElementById('hsParquet');
  if (!j.ok) {
    meta.textContent = '';
    cont.innerHTML = `<div style="padding:14px;color:#f87171;font-size:0.78rem;">${esc(j.error || 'unknown error')}</div>`;
    return;
  }
  meta.textContent = `${j.rows.length} of ${j.total} rows · cols: ${j.columns.join(', ')}`;
  const cards = j.rows.map((r, i) => {
    const cat = r.category || '?';
    const catColor = cat === 'positive' ? '#4ade80' : '#f87171';
    return `<div style="border-top:1px solid #1e1e2a;padding:10px 14px;">
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:6px;flex-wrap:wrap;">
        <span style="font-family:ui-monospace,monospace;color:#666;font-size:0.7rem;">#${i}</span>
        <span style="background:#252535;color:${catColor};font-family:ui-monospace,monospace;font-size:0.7rem;padding:2px 8px;border-radius:8px;font-weight:600;">${esc(cat)}</span>
        <span style="font-family:ui-monospace,monospace;color:#888;font-size:0.7rem;">concept_id=${esc(String(r.concept_id))}</span>
        ${r.concept_genre ? `<span style="font-family:ui-monospace,monospace;color:#888;font-size:0.7rem;">genre=${esc(String(r.concept_genre))}</span>`:''}
      </div>
      ${r.output_concept ? `<div style="font-size:0.74rem;color:#aaa;margin-bottom:6px;"><b style="color:#666;">concept:</b> ${esc(String(r.output_concept))}</div>`:''}
      <details>
        <summary style="cursor:pointer;color:#aaa;font-size:0.74rem;padding:2px 0;">▸ input (${(String(r.input || '').length).toLocaleString()} chars)</summary>
        <pre style="background:#15151e;padding:8px 10px;margin-top:4px;border-radius:4px;font-family:ui-monospace,monospace;font-size:0.72rem;color:#bbb;white-space:pre-wrap;max-height:300px;overflow-y:auto;line-height:1.5;">${esc(String(r.input || ''))}</pre>
      </details>
      <details open style="margin-top:4px;">
        <summary style="cursor:pointer;color:#aaa;font-size:0.74rem;padding:2px 0;">▸ output (${(String(r.output || '').length).toLocaleString()} chars) — y_neg/y_pos target</summary>
        <pre style="background:#15151e;border-left:3px solid ${catColor};padding:8px 10px;margin-top:4px;border-radius:4px;font-family:ui-monospace,monospace;font-size:0.74rem;color:#ddd;white-space:pre-wrap;max-height:340px;overflow-y:auto;line-height:1.55;">${esc(String(r.output || ''))}</pre>
      </details>
    </div>`;
  }).join('');
  cont.innerHTML = cards || '<div style="padding:14px;color:#666;">No rows.</div>';
}

function renderHSteerRun(run, tag) {
  const cont = document.getElementById('hsteerPage');
  const r = HSTEER_RUNS.find(x => x.run === run);
  if (!r) { cont.innerHTML = hsBackBar() + '<div class="no-results">Run not found.</div>'; return; }
  const c = r.configs.find(x => x.tag === tag);
  if (!c) { cont.innerHTML = hsBackBar() + '<div class="no-results">Config not found.</div>'; return; }

  const meta = c.metadata || {};
  const sa = meta.sampling_args || {};
  const variantDesc = HSTEER_VARIANT_KEY[c.variant] || '';
  const invalidNotice = r.invalid_reason
    ? `<div style="background:#3d1a1a;color:#fca5a5;padding:9px 13px;border-radius:5px;margin-bottom:12px;font-size:0.78rem;border:1px solid #5a2020;"><b>⚠ INVALID RUN:</b> ${esc(r.invalid_reason)}</div>`
    : '';
  const tokWarning = c.max_tokens === 1024
    ? `<div style="background:#3d2d12;color:#fde68a;padding:7px 12px;border-radius:5px;margin-bottom:10px;font-size:0.74rem;border-left:3px solid #fbbf24;">max_tokens=1024 — long deliberations may be cut off; "no tool call" can read as resist=1.0 falsely. Prefer 2k runs for clean signal.</div>`
    : '';

  cont.innerHTML = hsBackBar() + `<div style="max-width:1200px;margin:0 auto;">
    ${invalidNotice}
    ${tokWarning}
    <h2 style="font-size:1rem;color:#60a5fa;margin-bottom:4px;font-family:ui-monospace,monospace;">${esc(run)} / ${esc(tag)}</h2>
    <div style="font-size:0.78rem;color:#888;margin-bottom:14px;">
      variant <span style="color:#60a5fa;font-family:ui-monospace,monospace;font-weight:600;">${esc(c.variant || '—')}</span>
      ${variantDesc ? ` — <span style="color:#aaa;">${esc(variantDesc)}</span>` : ''}
      · factor <span style="color:#c084fc;font-family:ui-monospace,monospace;font-weight:600;">${fmtFactor(c.factor)}</span>
    </div>

    <div class="stats-sweep">
      <div class="stats-sweep-hdr"><h2 style="font-size:0.9rem;">metadata.json</h2></div>
      <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">
        <tbody>
          ${kv('env_id', meta.env_id)}
          ${kv('model', meta.model)}
          ${kv('judge_model', (meta.env_args||{}).judge_model)}
          ${kv('num_examples', meta.num_examples)}
          ${kv('max_tokens', sa.max_tokens)}
          ${kv('temperature', sa.temperature)}
          ${kv('time', fmtMs(meta.time_ms))}
          ${kv('avg_reward', meta.avg_reward != null ? meta.avg_reward.toFixed(3) : null)}
          ${kv('avg attack_resistance', meta.avg_metrics ? (meta.avg_metrics.attack_resistance||0).toFixed(3) : null)}
          ${kv('avg attack_detected', meta.avg_metrics ? (meta.avg_metrics.attack_detected||0).toFixed(3) : null)}
          ${kv('avg num_turns', meta.avg_metrics ? (meta.avg_metrics.num_turns||0).toFixed(2) : null)}
        </tbody>
      </table>
    </div>

    <div class="stats-sweep">
      <div class="stats-sweep-hdr">
        <h2 style="font-size:0.9rem;">Examples (${c.n_results})</h2>
        <span class="meta">click a row for full prompt + completion + judge</span>
      </div>
      <div id="hsRunRows" style="padding:14px;color:#666;font-size:0.8rem;">loading…</div>
    </div>
  </div>`;

  const key = run + '|' + tag;
  if (HS_RESULTS_CACHE[key]) {
    renderHSRunRows(run, tag, HS_RESULTS_CACHE[key]);
  } else {
    fetch('/api/hsteer/results?run=' + encodeURIComponent(run) + '&tag=' + encodeURIComponent(tag))
      .then(r => r.json())
      .then(rows => { HS_RESULTS_CACHE[key] = rows; renderHSRunRows(run, tag, rows); })
      .catch(e => { document.getElementById('hsRunRows').innerHTML = `<div style="color:#f87171;">load failed: ${esc(String(e))}</div>`; });
  }
}

function kv(k, v) {
  if (v == null || v === '') return '';
  return `<tr style="border-top:1px solid #1e1e2a;">
    <td style="padding:4px 10px;font-size:0.74rem;color:#777;width:200px;">${esc(k)}</td>
    <td style="padding:4px 10px;font-family:ui-monospace,monospace;font-size:0.78rem;color:#ddd;">${esc(String(v))}</td>
  </tr>`;
}

function renderHSRunRows(run, tag, rows) {
  const cont = document.getElementById('hsRunRows');
  if (!rows.length) { cont.innerHTML = '<div style="color:#666;">No rows.</div>'; return; }
  const list = rows.map(r => {
    const ar = r.attack_resistance;
    const bg = hsRewardCellColor(ar);
    const fg = (ar != null && ar >= 0.4 && ar <= 0.7) ? '#1a1a24' : '#fff';
    const truncFlag = r.is_truncated ? '<span style="color:#fbbf24;font-size:0.66rem;" title="completion truncated">⊘</span>' : '';
    return `<div style="display:flex;align-items:flex-start;border-top:1px solid #1e1e2a;padding:7px 10px;cursor:pointer;gap:10px;" onclick="hsGo({kind:'example', run:'${esc(run)}', tag:'${esc(tag)}', eid:${r.example_id}})">
      <span style="background:${bg};color:${fg};font-family:ui-monospace,monospace;font-size:0.74rem;font-weight:700;padding:3px 8px;border-radius:4px;min-width:42px;text-align:center;flex-shrink:0;">${ar==null?'—':ar.toFixed(2)}</span>
      <span style="font-family:ui-monospace,monospace;font-size:0.7rem;color:#666;flex-shrink:0;width:30px;text-align:right;">${esc(String(r.example_id))}</span>
      <div style="flex:1;min-width:0;">
        <div style="font-size:0.78rem;color:#ddd;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(r.user_query)}</div>
        <div style="font-size:0.68rem;color:#666;margin-top:2px;display:flex;gap:8px;flex-wrap:wrap;">
          ${r.server ? `<span style="color:#60a5fa;">${esc(r.server)}</span>`:''}
          ${r.security_risk ? `<span style="color:#f87171;">${esc(r.security_risk)}</span>`:''}
          ${r.paradigm ? `<span style="color:#4ade80;">${esc(r.paradigm)}</span>`:''}
          <span>uq=${r.user_query_len}c</span>
          <span>resp=${r.completion_len}c</span>
          <span>${esc(r.stop_condition || '')}</span>
          ${truncFlag}
        </div>
      </div>
    </div>`;
  }).join('');
  cont.innerHTML = list;
  cont.style.padding = '0';
}

function renderHSteerExample(run, tag, eid) {
  const cont = document.getElementById('hsteerPage');
  const key = run + '|' + tag + '|' + eid;
  cont.innerHTML = hsBackBar() + `
    <div style="margin-bottom:10px;">
      <button onclick="hsGo({kind:'run', run:'${esc(run)}', tag:'${esc(tag)}'})" style="background:#252535;border:1px solid #3a3a4a;color:#aaa;padding:5px 12px;border-radius:5px;font-size:0.74rem;cursor:pointer;">← back to ${esc(tag)} examples</button>
      <button onclick="hsGo({kind:'compare', eid:${eid}})" style="background:#1a2d3d;border:1px solid #60a5fa;color:#60a5fa;padding:5px 12px;border-radius:5px;font-size:0.74rem;cursor:pointer;margin-left:6px;font-weight:600;">→ compare example ${eid} across all runs</button>
    </div>
    <div id="hsExBody" style="color:#666;font-size:0.85rem;">loading…</div>`;

  const render = (full) => renderHSExampleBody(run, tag, eid, full);
  if (HS_FULL_CACHE[key]) { render(HS_FULL_CACHE[key]); return; }
  fetch('/api/hsteer/example?run=' + encodeURIComponent(run) + '&tag=' + encodeURIComponent(tag) + '&eid=' + eid)
    .then(r => r.json())
    .then(j => { HS_FULL_CACHE[key] = j; render(j); })
    .catch(e => { document.getElementById('hsExBody').innerHTML = `<div style="color:#f87171;">load failed: ${esc(String(e))}</div>`; });
}

function splitThink(content) {
  // Returns [thinkBlock, postThink, hasThink]
  const m = content && content.match(/<think>([\s\S]*?)<\/think>([\s\S]*)/);
  if (m) return [m[1], m[2], true];
  return ['', content || '', false];
}

function renderHSExampleBody(run, tag, eid, j) {
  const cont = document.getElementById('hsExBody');
  if (!j || j.error) { cont.innerHTML = `<div style="color:#f87171;">${esc((j && j.error) || 'not found')}</div>`; return; }
  const prompt = j.prompt || [];
  const completion = j.completion || [];
  const sys = (prompt[0] || {}).content || '';
  const user = (prompt[1] || {}).content || '';
  const last = completion.length ? (completion[completion.length-1] || {}) : {};
  const lastContent = last.content || '';
  const [thinkBlock, postThink, hasThink] = splitThink(lastContent);
  const info = j.info || {};
  const metrics = j.metrics || {};

  const ar = metrics.attack_resistance;
  const bg = hsRewardCellColor(ar);
  const fg = (ar != null && ar >= 0.4 && ar <= 0.7) ? '#1a1a24' : '#fff';

  cont.innerHTML = `<div style="max-width:1200px;margin:0 auto;">
    <h2 style="font-size:1rem;color:#60a5fa;margin-bottom:4px;font-family:ui-monospace,monospace;">${esc(run)} / ${esc(tag)} / example ${eid}</h2>
    <div style="margin-bottom:14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
      <span style="background:${bg};color:${fg};font-family:ui-monospace,monospace;font-size:0.78rem;font-weight:700;padding:4px 10px;border-radius:5px;">attack_resistance ${ar==null?'—':ar.toFixed(2)}</span>
      <span style="font-family:ui-monospace,monospace;font-size:0.74rem;color:#aaa;">reward=${j.reward==null?'—':j.reward.toFixed(2)}</span>
      <span style="font-family:ui-monospace,monospace;font-size:0.74rem;color:#aaa;">attack_detected=${metrics.attack_detected==null?'—':metrics.attack_detected.toFixed(2)}</span>
      <span style="font-family:ui-monospace,monospace;font-size:0.74rem;color:#aaa;">turns=${metrics.num_turns||0}</span>
      <span style="font-family:ui-monospace,monospace;font-size:0.74rem;color:#aaa;">stop=${esc(j.stop_condition||'')}</span>
      ${j.is_truncated ? '<span style="background:#3d2d12;color:#fbbf24;font-size:0.7rem;padding:2px 8px;border-radius:8px;font-weight:600;">⊘ truncated</span>' : ''}
    </div>

    ${info && Object.keys(info).length ? `<div class="stats-sweep">
      <div class="stats-sweep-hdr"><h2 style="font-size:0.9rem;">info</h2></div>
      <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">
        <tbody>
          ${kv('server_name', info.server_name || info.server)}
          ${kv('security_risk', info.security_risk)}
          ${kv('paradigm', info.paradigm)}
          ${kv('data_id', info.data_id)}
        </tbody>
      </table>
      ${info.poisoned_tool ? `<details style="border-top:1px solid #1e1e2a;"><summary style="cursor:pointer;padding:8px 14px;color:#aaa;font-size:0.78rem;background:#1d1d2a;">▸ poisoned_tool description</summary><pre style="background:#15151e;border-left:3px solid #fbbf24;padding:10px 14px;margin:0;font-family:ui-monospace,monospace;font-size:0.74rem;color:#ddd;white-space:pre-wrap;max-height:280px;overflow-y:auto;line-height:1.55;">${esc(info.poisoned_tool)}</pre></details>` : ''}
    </div>` : ''}

    <div class="stats-sweep">
      <div class="stats-sweep-hdr"><h2 style="font-size:0.9rem;">user query — prompt[1].content (${user.length.toLocaleString()} chars)</h2></div>
      <pre style="background:#15151e;padding:11px 14px;margin:0;font-family:ui-monospace,monospace;font-size:0.78rem;color:#ddd;white-space:pre-wrap;max-height:240px;overflow-y:auto;line-height:1.55;">${esc(user)}</pre>
    </div>

    <div class="stats-sweep">
      <div class="stats-sweep-hdr">
        <h2 style="font-size:0.9rem;">model response — completion[-1].content (${lastContent.length.toLocaleString()} chars) ${hasThink ? '<span style="color:#fbbf24;font-weight:400;font-size:0.7rem;">· split at &lt;/think&gt;</span>' : ''}</h2>
      </div>
      ${hasThink ? `
        <div style="border-top:1px solid #1e1e2a;">
          <div style="background:#1d1d2a;padding:6px 14px;font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:0.05em;">&lt;think&gt; block (${thinkBlock.length.toLocaleString()} chars)</div>
          <pre style="background:#15151e;border-left:3px solid #c084fc;padding:11px 14px;margin:0;font-family:ui-monospace,monospace;font-size:0.76rem;color:#cbb;white-space:pre-wrap;max-height:400px;overflow-y:auto;line-height:1.55;">${esc(thinkBlock)}</pre>
          <div style="background:#1d1d2a;padding:6px 14px;font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:0.05em;border-top:1px solid #1e1e2a;">post-think output (${postThink.length.toLocaleString()} chars)</div>
          <pre style="background:#15151e;border-left:3px solid #4ade80;padding:11px 14px;margin:0;font-family:ui-monospace,monospace;font-size:0.78rem;color:#ddd;white-space:pre-wrap;max-height:400px;overflow-y:auto;line-height:1.55;">${esc(postThink)}</pre>
        </div>
      ` : `
        <pre style="background:#15151e;padding:11px 14px;margin:0;font-family:ui-monospace,monospace;font-size:0.78rem;color:#ddd;white-space:pre-wrap;max-height:500px;overflow-y:auto;line-height:1.55;">${esc(lastContent)}</pre>
      `}
    </div>

    ${j.answer ? `<div class="stats-sweep">
      <div class="stats-sweep-hdr"><h2 style="font-size:0.9rem;">answer (judge expected)</h2></div>
      <pre style="background:#15151e;padding:10px 14px;margin:0;font-family:ui-monospace,monospace;font-size:0.78rem;color:#aaa;white-space:pre-wrap;">${esc(String(j.answer))}</pre>
    </div>`:''}

    <details>
      <summary style="cursor:pointer;color:#aaa;font-size:0.78rem;padding:6px 0;">▸ system prompt — prompt[0].content (${sys.length.toLocaleString()} chars)</summary>
      <pre style="background:#15151e;padding:11px 14px;margin-top:6px;font-family:ui-monospace,monospace;font-size:0.74rem;color:#bbb;white-space:pre-wrap;max-height:400px;overflow-y:auto;line-height:1.55;border-radius:4px;">${esc(sys)}</pre>
    </details>

    <details>
      <summary style="cursor:pointer;color:#aaa;font-size:0.78rem;padding:6px 0;">▸ raw json (everything)</summary>
      <pre style="background:#15151e;padding:11px 14px;margin-top:6px;font-family:ui-monospace,monospace;font-size:0.7rem;color:#888;white-space:pre-wrap;max-height:500px;overflow-y:auto;line-height:1.5;border-radius:4px;">${esc(JSON.stringify(j, null, 2))}</pre>
    </details>
  </div>`;
}

function renderHSteerCompare(eid) {
  const cont = document.getElementById('hsteerPage');
  // Collect every (run, config) cell and request the row for this eid in parallel.
  const cells = [];
  HSTEER_RUNS.forEach(r => r.configs.forEach(c => {
    if (c.has_leaf) cells.push({ run: r.run, tag: c.tag, variant: c.variant, factor: c.factor, max_tokens: c.max_tokens, invalid: !!r.invalid_reason, mtime: r.mtime });
  }));
  cells.sort((a, b) => {
    if (a.invalid !== b.invalid) return a.invalid ? 1 : -1;  // valid first
    const va = a.variant || '~';
    const vb = b.variant || '~';
    if (va !== vb) return va.localeCompare(vb);
    return (a.factor||0) - (b.factor||0);
  });

  cont.innerHTML = hsBackBar() + `<div style="max-width:1400px;margin:0 auto;">
    <h2 style="font-size:1rem;color:#60a5fa;margin-bottom:6px;">Cross-run comparison — example ${eid}</h2>
    <div style="font-size:0.78rem;color:#888;margin-bottom:14px;">
      Same prompt, every (run × config) cell side-by-side. Cells without this example_id are skipped.
      Loading ${cells.length} cells…
    </div>
    <div id="hsCmpGrid" style="display:flex;flex-wrap:wrap;gap:12px;"></div>
  </div>`;

  const grid = document.getElementById('hsCmpGrid');
  cells.forEach(cell => {
    const cellId = `hsc_${cell.run}_${cell.tag}`.replace(/[^a-z0-9_]/gi,'_');
    const placeholder = document.createElement('div');
    placeholder.id = cellId;
    placeholder.style.cssText = 'flex:0 0 380px;background:#1a1a24;border:1px solid #2a2a3a;border-radius:6px;overflow:hidden;display:flex;flex-direction:column;max-height:600px;';
    placeholder.innerHTML = `<div style="padding:8px 11px;background:#20202e;font-size:0.74rem;color:#aaa;font-family:ui-monospace,monospace;">${esc(cell.tag)} <span style="color:#666;font-size:0.68rem;">${esc(cell.run)}</span></div><div style="padding:14px;color:#666;font-size:0.78rem;flex:1;">loading…</div>`;
    grid.appendChild(placeholder);

    const key = cell.run + '|' + cell.tag + '|' + eid;
    const apply = (j) => {
      HS_FULL_CACHE[key] = j;
      const el = document.getElementById(cellId);
      if (!el) return;
      if (!j || j.error || (j && Object.keys(j).length === 0)) {
        el.innerHTML = `<div style="padding:8px 11px;background:#20202e;font-size:0.74rem;color:#666;font-family:ui-monospace,monospace;">${esc(cell.tag)}</div><div style="padding:14px;color:#666;font-size:0.74rem;">no example ${eid} in this cell</div>`;
        return;
      }
      const completion = j.completion || [];
      const last = completion.length ? (completion[completion.length-1] || {}) : {};
      const lastContent = last.content || '';
      const [thinkBlock, postThink, hasThink] = splitThink(lastContent);
      const ar = (j.metrics||{}).attack_resistance;
      const bg = hsRewardCellColor(ar);
      const fg = (ar != null && ar >= 0.4 && ar <= 0.7) ? '#1a1a24' : '#fff';
      const variantDot = cell.variant ? `<span style="color:#60a5fa;">${esc(cell.variant)}</span>` : '';
      const factorDot = cell.factor != null ? `<span style="color:#c084fc;">f${fmtFactor(cell.factor)}</span>` : '';
      const tokDot = cell.max_tokens ? `<span style="color:${cell.max_tokens===2048?'#4ade80':'#fbbf24'};font-size:0.66rem;">${cell.max_tokens===2048?'2k':(cell.max_tokens===1024?'1k':cell.max_tokens)}</span>` : '';
      const invalidBadge = cell.invalid ? '<span style="background:#3d1a1a;color:#fca5a5;font-size:0.62rem;padding:1px 6px;border-radius:6px;font-weight:600;">INVALID</span>' : '';
      el.innerHTML = `
        <div style="padding:7px 11px;background:#20202e;font-size:0.74rem;color:#ddd;font-family:ui-monospace,monospace;display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
          <span style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:0.72rem;">
            ${esc(cell.tag)} ${variantDot} ${factorDot} ${tokDot} ${invalidBadge}
          </span>
          <span style="background:${bg};color:${fg};font-family:ui-monospace,monospace;font-size:0.7rem;font-weight:700;padding:2px 8px;border-radius:8px;">AR ${ar==null?'—':ar.toFixed(2)}</span>
        </div>
        <div style="padding:5px 11px;background:#15151e;font-size:0.66rem;color:#666;font-family:ui-monospace,monospace;">${esc(cell.run)}</div>
        ${hasThink ? `
          <div style="background:#1d1d2a;padding:3px 11px;font-size:0.62rem;color:#888;text-transform:uppercase;letter-spacing:0.05em;">think (${thinkBlock.length}c)</div>
          <pre style="background:#15151e;border-left:3px solid #c084fc;padding:7px 11px;margin:0;font-family:ui-monospace,monospace;font-size:0.7rem;color:#cbb;white-space:pre-wrap;max-height:200px;overflow-y:auto;line-height:1.45;">${esc(thinkBlock)}</pre>
          <div style="background:#1d1d2a;padding:3px 11px;font-size:0.62rem;color:#888;text-transform:uppercase;letter-spacing:0.05em;">post (${postThink.length}c)</div>
          <pre style="background:#15151e;border-left:3px solid #4ade80;padding:7px 11px;margin:0;font-family:ui-monospace,monospace;font-size:0.72rem;color:#ddd;white-space:pre-wrap;flex:1;overflow-y:auto;line-height:1.45;">${esc(postThink)}</pre>
        ` : `
          <pre style="background:#15151e;padding:7px 11px;margin:0;font-family:ui-monospace,monospace;font-size:0.72rem;color:#ddd;white-space:pre-wrap;flex:1;overflow-y:auto;line-height:1.45;">${esc(lastContent)}</pre>
        `}
      `;
    };
    if (HS_FULL_CACHE[key]) { apply(HS_FULL_CACHE[key]); }
    else {
      fetch('/api/hsteer/example?run=' + encodeURIComponent(cell.run) + '&tag=' + encodeURIComponent(cell.tag) + '&eid=' + eid)
        .then(r => r.json()).then(apply).catch(() => apply({error:'fetch failed'}));
    }
  });
}

window.openPair = (pid, sweep) => {
  selectedId = pid;
  selectedSweep = sweep;
  document.querySelector('.tab[data-tab="browse"]').click();
  const r = ROWS.find(x => x.id === pid);
  if (r) renderDetail(r);
  renderList();
};

/* ----- tabs ----- */
document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
    const tab = t.dataset.tab;
    ['browsePage','statsPage','sweepsPage','rolloutsPage','stabilityPage','highlightsPage','hsteerPage'].forEach(id => {
      document.getElementById(id).classList.toggle('active', id === tab+'Page');
    });
    document.getElementById('browseFilters').style.display = tab === 'browse' ? '' : 'none';
    if (tab === 'stats') renderStats();
    if (tab === 'sweeps') renderSweeps();
    if (tab === 'rollouts') renderRollouts();
    if (tab === 'stability') renderStability();
    if (tab === 'highlights') renderHighlights();
    if (tab === 'hsteer') hsGo(HS_VIEW || {kind:'index'});
  };
});

/* ----- live polling ----- */
let CURRENT_TAB = 'browse';
let LIVE_ETAG = null;
let LIVE_ENABLED = true;
let LIVE_TIMER = null;

document.querySelectorAll('.tab').forEach(t => {
  const orig = t.onclick;
  t.addEventListener('click', () => { CURRENT_TAB = t.dataset.tab; });
});

function rebuildDerived() {
  ROW_BY_ID  = Object.fromEntries(ROWS.map(r => [r.id, r]));
  PAIR_LAYERS = {};
  Object.entries(BY_PAIR).forEach(([pid, entries]) => {
    const layers = new Set();
    entries.forEach(e => sweepLayers(e.sweep).forEach(L => layers.add(L)));
    PAIR_LAYERS[pid] = layers;
  });
}

function reRenderActive() {
  applyFilters();  // browse list always re-renders
  if (CURRENT_TAB === 'stats')      renderStats();
  if (CURRENT_TAB === 'sweeps')     renderSweeps();
  if (CURRENT_TAB === 'rollouts')   renderRollouts();
  if (CURRENT_TAB === 'stability')  renderStability();
  if (CURRENT_TAB === 'highlights') renderHighlights();
  // re-render detail if a pair is open
  if (selectedId) {
    const r = ROW_BY_ID[selectedId];
    if (r) renderDetail(r);
  }
}

async function pollOnce() {
  const status = document.getElementById('liveStatus');
  if (!LIVE_ENABLED) { status.textContent = '○ paused'; status.style.color = '#666'; return; }
  try {
    const resp = await fetch('/api/state', LIVE_ETAG ? { headers: { 'If-None-Match': LIVE_ETAG }} : {});
    if (resp.status === 304) {
      const t = new Date().toLocaleTimeString();
      status.textContent = '● live · ' + t;
      status.style.color = '#4ade80';
      return;
    }
    if (!resp.ok) throw new Error('http ' + resp.status);
    LIVE_ETAG = resp.headers.get('ETag');
    const data = await resp.json();
    ROWS          = data.rows;
    BY_PAIR       = data.by_pair;
    STATS         = data.stats;
    VARIES        = new Set(data.varies);
    SWEEP_META    = data.sweep_meta;
    ANALYSIS      = data.analysis;
    SWEEP_DETAILS = data.sweep_details;
    LABELLED      = data.labelled;
    BORDERLINE    = data.borderline;
    BORDER_LOG    = data.border_log;
    rebuildDerived();
    reRenderActive();
    const t = new Date().toLocaleTimeString();
    status.textContent = '● updated · ' + t;
    status.style.color = '#fbbf24';
    setTimeout(() => { status.textContent = '● live · ' + t; status.style.color = '#4ade80'; }, 1500);
  } catch (e) {
    status.textContent = '⚠ poll error';
    status.style.color = '#f87171';
  }
}

document.getElementById('liveStatus').addEventListener('click', () => {
  LIVE_ENABLED = !LIVE_ENABLED;
  if (LIVE_ENABLED) {
    pollOnce();
    LIVE_TIMER = setInterval(pollOnce, 15000);
  } else {
    clearInterval(LIVE_TIMER); LIVE_TIMER = null;
    document.getElementById('liveStatus').textContent = '○ paused';
    document.getElementById('liveStatus').style.color = '#666';
  }
});
LIVE_TIMER = setInterval(pollOnce, 15000);
setTimeout(pollOnce, 200);

// initial badges
{
  const changed = Object.values(ANALYSIS).filter(x => x.verdict === 'CHANGED').length;
  document.getElementById('hlBadge').textContent = changed || '';
  const totalRuns = HSTEER_RUNS.length;
  document.getElementById('hsBadge').textContent = totalRuns || '';
  const borderline = LABELLED.filter(r => {
    const l = r.label || (r.tags && r.tags.label);
    const g = (r.extra && r.extra.judge_grade) || (r.tags && r.tags.judge_grade);
    return l === 'ambiguous' || l === 'borderline' || g === 'C';
  }).length;
  document.getElementById('rolBadge').textContent = borderline || '';
}

['search','fServer','fRisk','fParadigm','fLayer','fSteering','fVaries'].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener(el.type === 'checkbox' ? 'change' : 'input', applyFilters);
});

applyFilters();
</script>
</body>
</html>"""


def _input_signature() -> str:
    """Hash of every input file's mtime+size — cheap, picks up new sweep dirs and edits."""
    parts = []
    for f in [DATA_FILE, ANALYSIS_FILE, LABELLED_FILE, BORDERLINE_FILE, BORDER_LOG_FILE]:
        if f.exists():
            st = f.stat()
            parts.append(f"{f.name}:{st.st_mtime_ns}:{st.st_size}")
    if EVAL_BASE.exists():
        for rf in EVAL_BASE.rglob("results.jsonl"):
            st = rf.stat()
            parts.append(f"{rf.relative_to(EVAL_BASE)}:{st.st_mtime_ns}:{st.st_size}")
    acts = ROOT / "outputs" / "acts"
    if acts.exists():
        for idx in acts.rglob("index.jsonl"):
            st = idx.stat()
            parts.append(f"acts/{idx.relative_to(acts)}:{st.st_mtime_ns}:{st.st_size}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _rebuild_globals():
    """Re-read every input file and rebuild the cached globals."""
    global ROWS, BY_PAIR, BY_SWEEP_ALPHA, AUTO_SWEEP_LAYERS, STATS, VARIES
    global ANALYSIS, TRAIN_PAIRS, SWEEP_DETAILS, LABELLED, BORDERLINE, BORDER_LOG
    ROWS = [json.loads(l) for l in DATA_FILE.read_text().splitlines() if l.strip()]
    BY_PAIR, BY_SWEEP_ALPHA, AUTO_SWEEP_LAYERS = _build_steering()
    _merge_auto_layers(BY_SWEEP_ALPHA, AUTO_SWEEP_LAYERS)
    STATS = _build_stats(BY_SWEEP_ALPHA)
    VARIES = _mark_varies(BY_PAIR)
    ANALYSIS = _load_analysis()
    TRAIN_PAIRS = _load_training_pairs()
    SWEEP_DETAILS = _build_sweep_details()
    LABELLED = _load_jsonl(LABELLED_FILE)
    BORDERLINE = _load_jsonl(BORDERLINE_FILE)
    BORDER_LOG = BORDER_LOG_FILE.read_text() if BORDER_LOG_FILE.exists() else ""


def _state_dict() -> dict:
    return {
        "rows": ROWS,
        "by_pair": BY_PAIR,
        "stats": STATS,
        "varies": VARIES,
        "sweep_meta": SWEEP_META,
        "analysis": ANALYSIS,
        "sweep_details": SWEEP_DETAILS,
        "labelled": LABELLED,
        "borderline": BORDERLINE,
        "border_log": BORDER_LOG,
    }


_LAST_SIG = None


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send_json(self, obj, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global _LAST_SIG
        if self.path.startswith("/api/hsteer/"):
            try:
                u = urlparse(self.path)
                qs = {k: v[0] for k, v in parse_qs(u.query).items()}
                ep = u.path[len("/api/hsteer/"):]
                if ep == "yaml":
                    t = _find_train(qs.get("train", ""))
                    if not t:
                        self._send_text("(training dir not found)"); return
                    p = pathlib.Path(t["_paths"]["config"])
                    self._send_text(p.read_text() if p.exists() else "(config not found)")
                    return
                if ep == "parquet":
                    t = _find_train(qs.get("train", ""))
                    if not t:
                        self._send_json({"ok": False, "error": "training dir not found"}); return
                    self._send_json(_read_train_parquet_sample(pathlib.Path(t["_paths"]["parquet"])))
                    return
                if ep == "results":
                    _, c = _find_config(qs.get("run", ""), qs.get("tag", ""))
                    if not c or not c["leaf"]:
                        self._send_json([]); return
                    self._send_json(_read_results_examples(pathlib.Path(c["leaf"])))
                    return
                if ep == "example":
                    _, c = _find_config(qs.get("run", ""), qs.get("tag", ""))
                    if not c or not c["leaf"]:
                        self._send_json({"error": "config not found"}); return
                    eid = qs.get("eid", "")
                    full = _read_results_full(pathlib.Path(c["leaf"]), eid)
                    if full is None:
                        self._send_json({}); return
                    self._send_json(full)
                    return
                self._send_json({"error": "unknown endpoint"}, status=404)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path.startswith("/api/state"):
            sig = _input_signature()
            client_etag = self.headers.get("If-None-Match", "")
            if sig == client_etag:
                self.send_response(304)
                self.send_header("ETag", sig)
                self.end_headers()
                return
            if sig != _LAST_SIG:
                _rebuild_globals()
                _LAST_SIG = sig
            body = json.dumps(_state_dict(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("ETag", sig)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # initial HTML page — also keeps cache fresh
        sig = _input_signature()
        if sig != _LAST_SIG:
            _rebuild_globals()
            _LAST_SIG = sig
        page = (
            HTML
            .replace("__ROWS__",       json.dumps(ROWS,       ensure_ascii=False))
            .replace("__BY_PAIR__",    json.dumps(BY_PAIR,    ensure_ascii=False))
            .replace("__STATS__",      json.dumps(STATS,      ensure_ascii=False))
            .replace("__VARIES__",     json.dumps(VARIES,     ensure_ascii=False))
            .replace("__SWEEP_META__", json.dumps(SWEEP_META, ensure_ascii=False))
            .replace("__ANALYSIS__",   json.dumps(ANALYSIS,   ensure_ascii=False))
            .replace("__SWEEP_DETAILS__", json.dumps(SWEEP_DETAILS, ensure_ascii=False))
            .replace("__LABELLED__",   json.dumps(LABELLED,   ensure_ascii=False))
            .replace("__BORDERLINE__", json.dumps(BORDERLINE, ensure_ascii=False))
            .replace("__BORDER_LOG__", json.dumps(BORDER_LOG, ensure_ascii=False))
            .replace("__HSTEER_RUNS__",        json.dumps(_hsteer_runs_index_lite(), ensure_ascii=False))
            .replace("__HSTEER_TRAIN__",       json.dumps(_hsteer_train_index_lite(), ensure_ascii=False))
            .replace("__HSTEER_VARIANT_KEY__", json.dumps(HSTEER_VARIANT_KEY, ensure_ascii=False))
        )
        body = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = 7331
    print(f"Pairs: {len(ROWS)} · with steering: {len(BY_PAIR)} · varies: {len(VARIES)}")
    print(f"Sweeps: {list(STATS.keys())}")
    print(f"Dashboard at http://localhost:{port}  (Ctrl+C to stop)")
    http.server.HTTPServer(("localhost", port), Handler).serve_forever()
