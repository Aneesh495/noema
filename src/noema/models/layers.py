"""Original neural layers: RMSNorm, Rotary Embeddings, Multi-Head Attention, and Transformer Blocks."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) applied to query and key projections."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute theta frequencies: [dim // 2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        # Outer product: [seq_len, dim // 2]
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate to form [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        d_half = x.shape[-1] // 2
        x1 = x[..., :d_half]
        x2 = x[..., d_half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len].to(x.dtype)
        sin = self.sin_cached[:seq_len].to(x.dtype)
        return cos, sin

    def apply_rope(self, qk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # qk: [batch, heads, seq_len, head_dim]
        # cos, sin: [seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(1)
        return (qk * cos) + (self.rotate_half(qk) * sin)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE and explicit key padding and causal/bidirectional masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        rotary_dim: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Projections: [B, T, D] -> [B, T, H, d_h] -> [B, H, T, d_h]
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(q, seq_len)
        q = self.rope.apply_rope(q, cos, sin)
        k = self.rope.apply_rope(k, cos, sin)

        # Scaled dot-product attention
        # scores: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        if attention_mask is not None:
            # attention_mask: [B, T] where 1 is valid, 0 is padding
            # Expand to [B, 1, 1, T] to mask keys that are padding
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        # If an entire sequence is padding, softmax produces NaN; clamp to zero
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # Context: [B, H, T, d_h]
        context = torch.matmul(attn_weights, v)
        # Reshape to [B, T, D]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class SwiGLUFeedForward(nn.Module):
    """Gated SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (w_gate(x) * SiLU(w_up(x))) * w_down
        gate = self.w_gate(x)
        up = F.silu(self.w_up(x))
        hidden = gate * up
        return self.dropout(self.w_down(hidden))


class TransformerBlock(nn.Module):
    """Independently parameterized Transformer block with Pre-RMSNorm and SwiGLU."""

    def __init__(
        self,
        block_idx: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout=dropout)

        # Telemetry probe for activation tracking
        self.last_activation_norm: float = 0.0

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Residual attention
        normed1 = self.norm1(x)
        attn_out = self.attn(normed1, attention_mask=attention_mask, is_causal=is_causal)
        x = x + attn_out

        # Residual feed-forward
        normed2 = self.norm2(x)
        ffn_out = self.ffn(normed2)
        x = x + ffn_out

        # Track activation norm
        if not self.training:
            self.last_activation_norm = float(x.detach().norm().item())

        return x
