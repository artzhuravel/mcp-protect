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

from pathlib import Path

import torch


class QwenScopeSAE:
    def __init__(self, weights: dict[str, torch.Tensor], k: int):
        for key in ("W_enc", "W_dec", "b_enc", "b_dec"):
            if key not in weights:
                raise KeyError(f"SAE weights missing key {key!r}; got {list(weights)}")

        W_enc = weights["W_enc"]
        W_dec = weights["W_dec"]
        b_enc = weights["b_enc"]
        b_dec = weights["b_dec"]

        if W_enc.dim() != 2 or W_dec.dim() != 2:
            raise ValueError(f"W_enc/W_dec must be 2D; got {W_enc.shape}/{W_dec.shape}")

        # We canonicalize to W_enc (d_model, d_sae) and W_dec (d_sae, d_model).
        # Some Qwen-Scope releases ship transposed; transparently handle both.
        a, b = W_enc.shape
        c, d = W_dec.shape
        if a == d and b == c:
            pass
        elif a == c and b == d:
            W_enc = W_enc.T.contiguous()
        else:
            raise ValueError(f"W_enc {W_enc.shape} and W_dec {W_dec.shape} are inconsistent")

        d_model, d_sae = W_enc.shape
        if b_enc.shape != (d_sae,):
            raise ValueError(f"b_enc shape {b_enc.shape} != ({d_sae},)")
        if b_dec.shape != (d_model,):
            raise ValueError(f"b_dec shape {b_dec.shape} != ({d_model},)")

        self.W_enc = W_enc.requires_grad_(False)
        self.W_dec = W_dec.requires_grad_(False)
        self.b_enc = b_enc.requires_grad_(False)
        self.b_dec = b_dec.requires_grad_(False)
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

    @classmethod
    def from_qwen_scope_file(cls, path: Path | str, k: int = 50) -> "QwenScopeSAE":
        weights = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(weights, dict):
            raise TypeError(f"expected dict in {path}, got {type(weights)}")
        return cls(weights, k=k)

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> "QwenScopeSAE":
        for name in ("W_enc", "W_dec", "b_enc", "b_dec"):
            t = getattr(self, name)
            t = t.to(device=device, dtype=dtype) if dtype is not None else t.to(device=device)
            setattr(self, name, t)
        return self

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    @property
    def dtype(self) -> torch.dtype:
        return self.W_enc.dtype

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model)  →  a: (..., d_sae), TopK-sparse, non-negative."""
        x = x.to(self.W_enc.dtype)
        pre = x @ self.W_enc + self.b_enc
        post = torch.relu(pre)
        if self.k is None or self.k >= self.d_sae:
            return post
        topk_vals, topk_idx = post.topk(self.k, dim=-1)
        out = torch.zeros_like(post)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        """a: (..., d_sae)  →  x_recon: (..., d_model)."""
        a = a.to(self.W_dec.dtype)
        return a @ self.W_dec + self.b_dec

    def feature_direction(self, feature_idx: int) -> torch.Tensor:
        """Decoder column for one feature: shape (d_model,)."""
        return self.W_dec[feature_idx, :].clone()
