"""
Qwen3-8B intervention server for the SAE arm experiments.

Serves OpenAI-compatible POST /v1/chat/completions with optional residual-stream
interventions. Each named intervention reduces to "add a scaled vector to the
hidden state at the output of layer L"; this unifies the DiffMean (vector
addition) and SAE-clamp methods at the residual level. Interventions are
declared in a YAML registry and selected per request via the OpenAI `model`
field — one (method, hyperparams) tuple per registered model id.

No streaming, no multi-turn — sized for prime-envs vf-eval.

Install:
    pip install fastapi uvicorn pydantic torch transformers pyyaml

Run (baseline only):
    python sae_arm/server.py --port 8000

Run (with interventions):
    python sae_arm/server.py --registry sae_arm/interventions.yaml --port 8000

Sample interventions.yaml:

    interventions:
      - name: diffmean-l16-a2
        hook_type: diffmean_add
        layer: 16
        scale: -2.0                         # negative pushes away from poisoned
        direction_path: directions/refusal_l16.pt

      - name: sae-l24-feat42117-s5
        hook_type: sae_clamp
        layer: 24
        scale: 5.0
        # Pre-computed offline: a_max * W_dec[:, feat] summed across feats in
        # this intervention. Server applies `scale * direction` at layer L.
        direction_path: directions/sae_l24_feat42117.pt

The "baseline" model id is always registered with no hook.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Intervention:
    name: str
    hook_type: str  # "none" | "diffmean_add" | "sae_clamp"
    layer: int | None = None
    direction: torch.Tensor | None = None  # (d_model,), pre-scaled by `scale`


def _load_registry(
    path: Path | None,
    num_layers: int,
    d_model: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Intervention]:
    registry: dict[str, Intervention] = {
        "baseline": Intervention(name="baseline", hook_type="none"),
    }
    if path is None:
        return registry

    cfg = yaml.safe_load(path.read_text()) or {}
    base_dir = path.parent
    for spec in cfg.get("interventions", []):
        name = spec["name"]
        hook_type = spec.get("hook_type", "none")
        if hook_type == "none":
            registry[name] = Intervention(name=name, hook_type="none")
            continue
        if hook_type not in {"diffmean_add", "sae_clamp"}:
            raise ValueError(f"{name}: unknown hook_type {hook_type!r}")

        layer = int(spec["layer"])
        if not (0 <= layer < num_layers):
            raise ValueError(f"{name}: layer {layer} out of range [0,{num_layers})")
        scale = float(spec.get("scale", 1.0))

        # Direction is stored as a raw .pt tensor; we fold scale in at load
        # time so the hook is a single vector addition.
        direction_path = base_dir / spec["direction_path"]
        d = torch.load(direction_path, map_location=device).to(dtype).flatten()
        if d.shape != (d_model,):
            raise ValueError(
                f"{name}: direction shape {tuple(d.shape)} != ({d_model},)"
            )
        if spec.get("normalize", False):
            d = d / d.norm().clamp(min=1e-8)
        d = d * scale

        registry[name] = Intervention(
            name=name, hook_type=hook_type, layer=layer, direction=d
        )

    return registry


@contextmanager
def _intervention_hook(model, iv: Intervention):
    if iv.hook_type == "none":
        yield
        return

    layer_module = model.model.layers[iv.layer]
    direction = iv.direction

    def hook(_mod, _inp, output):
        # Some transformers versions return a tuple, some a tensor.
        if isinstance(output, tuple):
            return (output[0] + direction,) + output[1:]
        return output + direction

    handle = layer_module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class _ChatMessage(BaseModel):
    role: str
    content: str


class _ChatRequest(BaseModel):
    model: str = "baseline"
    messages: list[_ChatMessage]
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float | None = None
    stream: bool = False


def _build_app(
    model,
    tokenizer,
    registry: dict[str, Intervention],
    device: str,
    default_max_tokens: int,
) -> FastAPI:
    app = FastAPI(title="Qwen3 intervention policy", version="0.1.0")
    # model.generate is sync and blocks the event loop, so a single request
    # serializes by default. Lock makes it explicit and survives any future
    # move to a thread pool.
    gen_lock = asyncio.Lock()

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": k, "object": "model"} for k in registry]}

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: _ChatRequest):
        if req.stream:
            raise HTTPException(400, "streaming not supported")
        if not req.messages:
            raise HTTPException(400, "messages is required")
        if req.model not in registry:
            raise HTTPException(
                404, f"model '{req.model}' not registered: {list(registry)}"
            )
        iv = registry[req.model]
        max_toks = max(1, int(req.max_tokens) if req.max_tokens else default_max_tokens)

        msgs = [m.model_dump() for m in req.messages]
        # transformers 5.x returns BatchEncoding from apply_chat_template; older
        # versions returned a raw tensor. Handle both.
        out = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            enable_thinking=True,
            return_tensors="pt",
        )
        prompt_ids = (out["input_ids"] if hasattr(out, "input_ids") else out).to(device)
        prompt_len = prompt_ids.shape[1]

        gen_kwargs = {"max_new_tokens": max_toks}
        if req.temperature and req.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(req.temperature)
            if req.top_p is not None:
                gen_kwargs["top_p"] = float(req.top_p)
        else:
            gen_kwargs["do_sample"] = False
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id or tokenizer.eos_token_id

        async with gen_lock:
            with torch.inference_mode(), _intervention_hook(model, iv):
                out_ids = model.generate(prompt_ids, **gen_kwargs)

        new_tokens = out_ids[0, prompt_len:]
        # skip_special_tokens=False keeps <think>...</think> in the output so
        # downstream judges can apply their own strip policy.
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        if text.endswith("<|im_end|>"):
            text = text[: -len("<|im_end|>")]

        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--device", default=None)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="YAML with intervention configs. Without this only 'baseline' is exposed.",
    )
    p.add_argument("--default-max-tokens", type=int, default=1024)
    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32

    print(f"[serve] loading {args.model_name} on {args.device} ({dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, device_map=args.device
    ).eval()
    d_model = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    registry = _load_registry(args.registry, num_layers, d_model, args.device, dtype)
    print(f"[serve] interventions registered: {list(registry)}", flush=True)

    app = _build_app(model, tokenizer, registry, args.device, args.default_max_tokens)
    base_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(
        f"[serve] base URL for vf-eval: -b http://{base_host}:{args.port}/v1 -k EMPTY",
        flush=True,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
