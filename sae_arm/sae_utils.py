"""Minimal Qwen-Scope SAE wrapper.

Loads Qwen-Scope's per-layer .pt files (W_enc, W_dec, b_enc, b_dec dicts) and
exposes only the operations the rest of the SAE arm needs: encode, decode,
W_dec column lookup, .to(device).

We do NOT subclass sae_lens.SAE. Qwen-Scope's format isn't sae_lens-native and
the few methods we use are minimal. If we ever want to drop into Arad et al.'s
unmodified code, this class can be extended with hook_dict / hook_sae_error
stubs without breaking callers.

Top-K SAE math (Qwen-Scope is L0_50 by default):
    a    = TopK(ReLU(x @ W_enc + b_enc), k=50)
    x'   = a @ W_dec + b_dec
    err  = x − x'                           # part the SAE can't represent
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).parent
DIRECTIONS_DIR = ROOT / "directions"


def out_dir_for(set_name: str, layer: int, paradigm: str | None, security_risk: str | None) -> Path:
    """Canonical output directory for a build/eval cell.

    Single source of truth for the path scheme; both build_steering_vector.py
    and eval_mcptox.py import this so they can't drift.
    """
    base = DIRECTIONS_DIR / set_name / f"L{layer}"
    if paradigm and security_risk:
        risk_seg = security_risk.replace(" ", "_")
        return base / "by_paradigm_and_security_risk" / f"{paradigm}_{risk_seg}"
    if paradigm:
        return base / "by_paradigm" / paradigm
    if security_risk:
        return base / "by_security_risk" / security_risk.replace(" ", "_")
    return base


class QwenScopeSAE:
    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        k: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        for key in ("W_enc", "W_dec", "b_enc", "b_dec"):
            if key not in weights:
                raise KeyError(f"SAE weights missing key {key!r}; got {list(weights)}")

        W_enc = weights["W_enc"]
        W_dec = weights["W_dec"]
        b_enc = weights["b_enc"]
        b_dec = weights["b_dec"]

        if W_enc.dim() != 2 or W_dec.dim() != 2:
            raise ValueError(f"W_enc/W_dec must be 2D; got {W_enc.shape}/{W_dec.shape}")
        if b_enc.dim() != 1 or b_dec.dim() != 1:
            raise ValueError(f"b_enc/b_dec must be 1D; got {b_enc.shape}/{b_dec.shape}")
        
        d_sae = b_enc.shape[0]
        d_model = b_dec.shape[0]

        if W_enc.shape == (d_model, d_sae):
            pass
        elif W_enc.shape == (d_sae, d_model):
            W_enc = W_enc.T.contiguous()
        else:
            raise ValueError(
                f"W_enc shape {W_enc.shape} matches neither "
                f"(d_model={d_model}, d_sae={d_sae}) nor (d_sae, d_model)"
            )

        if W_dec.shape == (d_sae, d_model):
            pass
        elif W_dec.shape == (d_model, d_sae):
            W_dec = W_dec.T.contiguous()
        else:
            raise ValueError(
                f"W_dec shape {W_dec.shape} matches neither "
                f"(d_sae={d_sae}, d_model={d_model}) nor (d_model, d_sae)"
            )

        # Move + cast in one step at construction. SAEs only ever travel
        # once: from disk (CPU) to wherever they'll be used.
        kwargs: dict = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.W_enc = W_enc.to(**kwargs).requires_grad_(False)
        self.W_dec = W_dec.to(**kwargs).requires_grad_(False)
        self.b_enc = b_enc.to(**kwargs).requires_grad_(False)
        self.b_dec = b_dec.to(**kwargs).requires_grad_(False)
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Qwen-Scope's TopK SAE applies TopK *directly* to the raw pre-
        # activations — no ReLU before TopK (per the model card's reference
        # extraction code). Surviving values keep their sign; clamping to
        # non-negative would lose information the SAE was trained to use.
        x = x.to(self.W_enc.dtype)
        pre = x @ self.W_enc + self.b_enc
        if self.k is None or self.k >= self.d_sae:
            return pre
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        a = a.to(self.W_dec.dtype)
        return a @ self.W_dec + self.b_dec

    def feature_direction(self, feature_idx: int) -> torch.Tensor:
        return self.W_dec[feature_idx, :].clone()


def stratify_activations(
    H_pos: torch.Tensor,
    H_neg: torch.Tensor,
    index_path,
    *,
    paradigm: str | None = None,
    security_risk: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    if paradigm is None and security_risk is None:
        return H_pos, H_neg

    keep_pos: list[int] = []
    keep_neg: list[int] = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            tags = entry.get("tags", {})
            if paradigm is not None and tags.get("paradigm") != paradigm:
                continue
            if security_risk is not None and tags.get("security_risk") != security_risk:
                continue
            i_pos = entry.get("i_pos")
            i_neg = entry.get("i_neg")
            if i_pos is not None:
                keep_pos.append(i_pos)
            if i_neg is not None:
                keep_neg.append(i_neg)

    return H_pos[keep_pos], H_neg[keep_neg]
