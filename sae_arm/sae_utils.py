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

import torch


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
        a = a.to(self.W_dec.dtype)
        return a @ self.W_dec + self.b_dec

    def feature_direction(self, feature_idx: int) -> torch.Tensor:
        return self.W_dec[feature_idx, :].clone()
