"""OpenAI-compatible server that wraps a HuggingFace causal LM with either a
DiffMean steering hook or a trained LoReFT intervention. Lets prime-envs (or
any OpenAI-client tool) point at us and toggle steering via env or per-request
`extra_body`.

Endpoints:
  POST /v1/chat/completions   — OpenAI chat completions (non-streaming)
  POST /v1/completions        — OpenAI text completions (non-streaming)
  GET  /v1/models             — lists the loaded model id
  GET  /healthz               — { "ok": true, "model": "...", "alpha": 0.0 }

Per-request override (works in either endpoint via OpenAI clients'
`extra_body={...}` kwarg):
  {"steering": {"alpha": -4.0, "layer": 20}}
  (alpha is ignored when REFT_DIR is set — ReFT is always-on)

Server-side defaults come from env:
  DIFFMEAN_MODEL          default: google/gemma-2-9b-it
  DIFFMEAN_VEC            path to diffmean_vec.pt (required unless REFT_DIR set)
  DIFFMEAN_LAYER          default: 20
  DIFFMEAN_ALPHA          default: 0.0
  DIFFMEAN_DEVICE         default: cuda if available else cpu
  DIFFMEAN_DTYPE          default: bfloat16
  DIFFMEAN_ALL_TOKENS     "1" to steer all positions (default: last only)
  REFT_DIR                path to train_reft.py output dir (overrides DiffMean)

Run with DiffMean (original):
  DIFFMEAN_VEC=diffmean/outputs/acts/qwen3-8b/L20/diffmean_vec.pt \\
  DIFFMEAN_LAYER=20 DIFFMEAN_ALPHA=-4 \\
    uvicorn diffmean.serve:app --host 0.0.0.0 --port 8000

Run with ReFT (new):
  DIFFMEAN_MODEL=Qwen/Qwen3-8B \\
  REFT_DIR=diffmean/outputs/reft/qwen3-8b-L20-r4 \\
    uvicorn diffmean.serve:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------- LoReFT module (matches train_reft.py) --------------------------

class _LoReftIntervention(torch.nn.Module):
    """h' = h + R^T(Rh - b)  at chosen positions."""
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.R = torch.nn.Linear(d_model, rank, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(rank))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + (self.R(h) - self.b) @ self.R.weight


# ---------- Steering hook (mirrors diffmean/steer.py) ----------------------

def _get_layer_module(model, layer: int):
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("could not locate model.model.layers")
    return layers[layer]


@contextmanager
def steering_hook(model, layer: int, v: torch.Tensor, alpha: float, all_tokens: bool):
    """Add alpha*v to the residual stream coming out of layer L.
    No-op when alpha == 0 so the same code path serves both baseline and steered."""
    if alpha == 0.0:
        yield
        return
    module = _get_layer_module(model, layer)

    def hook(_mod, _inputs, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None
        delta = (alpha * v).to(h.dtype).to(h.device)
        if all_tokens:
            h = h + delta
        else:
            h[..., -1, :] = h[..., -1, :] + delta
        return (h, *rest) if rest is not None else h

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def reft_hook(model, interventions: dict[int, _LoReftIntervention],
              positions_fn, all_tokens: bool):
    """Apply trained LoReFT interventions at the prefix positions during generation.

    When all_tokens=True the intervention fires on every generated token;
    otherwise only on the last token of the prompt prefix (position=-1 semantics
    replicated via a generate-time step hook).
    """
    handles = []
    for layer_idx, iv in interventions.items():
        module = _get_layer_module(model, layer_idx)

        def make_hook(interv, all_tok):
            def hook(_mod, _inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                rest = output[1:] if isinstance(output, tuple) else None
                if all_tok:
                    hs = interv(hs)
                else:
                    # Only edit the last position (current generation step).
                    hs = hs.clone()
                    hs[:, -1, :] = interv(hs[:, -1, :])
                return (hs, *rest) if rest is not None else hs
            return hook

        handles.append(module.register_forward_hook(make_hook(iv, all_tokens)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------- App + state ----------------------------------------------------

class _State:
    model = None
    tokenizer = None
    vec: torch.Tensor | None = None
    reft_interventions: dict[int, _LoReftIntervention] | None = None
    reft_positions_fn = None  # callable(ctx_len) -> list[int], unused at serve time
    reft_all_tokens: bool = True
    model_id: str = ""
    default_layer: int = 20
    default_alpha: float = 0.0
    device: str = "cpu"
    all_tokens: bool = False
    mode: str = "diffmean"  # "diffmean" | "reft"


S = _State()
app = FastAPI(title="diffmean-serve", version="0.1")


@app.on_event("startup")
def _load() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    S.model_id = os.environ.get("DIFFMEAN_MODEL", "google/gemma-2-9b-it")
    reft_dir = os.environ.get("REFT_DIR")
    vec_path = os.environ.get("DIFFMEAN_VEC")

    if not reft_dir and not vec_path:
        raise RuntimeError("Set REFT_DIR (LoReFT) or DIFFMEAN_VEC (DiffMean)")

    S.default_layer = int(os.environ.get("DIFFMEAN_LAYER", "20"))
    S.default_alpha = float(os.environ.get("DIFFMEAN_ALPHA", "0.0"))
    S.all_tokens = os.environ.get("DIFFMEAN_ALL_TOKENS", "0") == "1"
    dtype_name = os.environ.get("DIFFMEAN_DTYPE", "bfloat16")
    S.device = os.environ.get("DIFFMEAN_DEVICE",
                              "cuda" if torch.cuda.is_available() else "cpu")

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype_name]
    S.tokenizer = AutoTokenizer.from_pretrained(S.model_id)
    S.model = AutoModelForCausalLM.from_pretrained(
        S.model_id, torch_dtype=torch_dtype, device_map=S.device,
    )
    S.model.eval()

    if reft_dir:
        _load_reft(Path(reft_dir))
    else:
        v = torch.load(vec_path).float()
        S.vec = v / (v.norm() + 1e-9)
        S.mode = "diffmean"
        print(f"[serve] DiffMean mode: {S.model_id} on {S.device}, "
              f"vec {vec_path} (d={S.vec.numel()}), "
              f"layer={S.default_layer} alpha={S.default_alpha} "
              f"all_tokens={S.all_tokens}", flush=True)


def _load_reft(reft_dir: Path) -> None:
    cfg_path = reft_dir / "reft_config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"reft_config.json not found in {reft_dir}")
    cfg = json.loads(cfg_path.read_text())

    d_model = cfg["d_model"]
    rank = cfg["rank"]
    layers = cfg["layers"]
    S.reft_all_tokens = os.environ.get("DIFFMEAN_ALL_TOKENS", "1") == "1"

    S.reft_interventions = {}
    for L in layers:
        iv = _LoReftIntervention(d_model, rank)
        weight_path = reft_dir / f"L{L:02d}_intervention.pt"
        iv.load_state_dict(torch.load(weight_path, map_location="cpu"))
        iv.to(S.device).eval()
        S.reft_interventions[L] = iv
        # Freeze — inference only.
        for p in iv.parameters():
            p.requires_grad_(False)

    S.mode = "reft"
    print(f"[serve] ReFT mode: {S.model_id} on {S.device}, "
          f"layers={layers} rank={rank} all_tokens={S.reft_all_tokens}", flush=True)


# ---------- Request / response schemas (subset of OpenAI) ------------------

class _Steering(BaseModel):
    alpha: float | None = None
    layer: int | None = None
    all_tokens: bool | None = None


class _Message(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str | None = None
    messages: list[_Message]
    # Accept both legacy `max_tokens` and the reasoning-model variant
    # `max_completion_tokens`. Verifiers/OpenAI Python SDK switched to the
    # latter for reasoning models — falling back to a 512 default silently
    # truncated long thinking traces. _resolve_max() returns the larger value.
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | str | None = None
    steering: _Steering | None = None  # our extension


class CompletionsRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | str | None = None
    steering: _Steering | None = None


def _resolve_max(req) -> int:
    """Take the largest declared budget across the two fields, default 512."""
    a = getattr(req, "max_tokens", None) or 0
    b = getattr(req, "max_completion_tokens", None) or 0
    return max(a, b) or 512


# ---------- Generation core ------------------------------------------------

@torch.no_grad()
def _generate(prompt: str, *, max_tokens: int, temperature: float, top_p: float,
              stop: list[str] | None, alpha: float, layer: int,
              all_tokens: bool) -> tuple[str, int, int]:
    enc = S.tokenizer(prompt, return_tensors="pt", add_special_tokens=False
                      ).to(S.device)
    n_in = enc["input_ids"].shape[1]
    do_sample = temperature > 0.0
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_tokens or 512,
        do_sample=do_sample,
        pad_token_id=S.tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    if S.mode == "reft" and S.reft_interventions:
        ctx_mgr = reft_hook(S.model, S.reft_interventions,
                            S.reft_positions_fn, S.reft_all_tokens)
    else:
        ctx_mgr = steering_hook(S.model, layer, S.vec, alpha, all_tokens)

    with ctx_mgr:
        out = S.model.generate(**enc, **gen_kwargs)
    gen_ids = out[0, n_in:]
    text = S.tokenizer.decode(gen_ids, skip_special_tokens=True)

    if stop:
        for s in (stop if isinstance(stop, list) else [stop]):
            idx = text.find(s)
            if idx >= 0:
                text = text[:idx]
    return text, n_in, gen_ids.shape[0]


def _resolve_steer(req_steering: _Steering | None) -> tuple[float, int, bool]:
    a = S.default_alpha
    L = S.default_layer
    at = S.all_tokens
    if req_steering is not None:
        if req_steering.alpha is not None:
            a = req_steering.alpha
        if req_steering.layer is not None:
            L = req_steering.layer
        if req_steering.all_tokens is not None:
            at = req_steering.all_tokens
    return a, L, at


def _render_chat(messages: list[_Message]) -> str:
    """Use the tokenizer's chat template; fold system into user for Gemma-style
    templates that don't support a system role."""
    msgs = [m.model_dump() for m in messages]
    try:
        return S.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: prepend the first system message into the next user turn.
        sys_buf = ""
        flat: list[dict] = []
        for m in msgs:
            if m["role"] == "system":
                sys_buf = (sys_buf + "\n\n" + m["content"]).strip()
            elif m["role"] == "user" and sys_buf:
                flat.append({"role": "user", "content": f"{sys_buf}\n\n{m['content']}"})
                sys_buf = ""
            else:
                flat.append(m)
        return S.tokenizer.apply_chat_template(
            flat, tokenize=False, add_generation_prompt=True
        )


# ---------- Endpoints ------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"ok": S.model is not None, "model": S.model_id,
            "mode": S.mode, "alpha": S.default_alpha,
            "layer": S.default_layer, "device": S.device}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": S.model_id, "object": "model", "owned_by": "diffmean-serve"}
    ]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest):
    if S.model is None:
        raise HTTPException(503, "model not loaded")
    alpha, layer, all_tok = _resolve_steer(req.steering)
    prompt = _render_chat(req.messages)
    text, n_in, n_out = _generate(
        prompt, max_tokens=_resolve_max(req),
        temperature=req.temperature, top_p=req.top_p,
        stop=([req.stop] if isinstance(req.stop, str) else req.stop),
        alpha=alpha, layer=layer, all_tokens=all_tok,
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or S.model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                  "total_tokens": n_in + n_out},
        "diffmean": {"alpha": alpha, "layer": layer, "all_tokens": all_tok},
    }


@app.post("/v1/completions")
def completions(req: CompletionsRequest):
    if S.model is None:
        raise HTTPException(503, "model not loaded")
    alpha, layer, all_tok = _resolve_steer(req.steering)
    text, n_in, n_out = _generate(
        req.prompt, max_tokens=req.max_tokens or 512,
        temperature=req.temperature, top_p=req.top_p,
        stop=([req.stop] if isinstance(req.stop, str) else req.stop),
        alpha=alpha, layer=layer, all_tokens=all_tok,
    )
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or S.model_id,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                  "total_tokens": n_in + n_out},
        "diffmean": {"alpha": alpha, "layer": layer, "all_tokens": all_tok},
    }
