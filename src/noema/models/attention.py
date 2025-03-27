"""Grouped-Query Attention (GQA) with rotary key-value caching for theorem provers."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from noema.models.rope import AdvancedRotaryEmbedding


class KVCache:
    """Dynamic key-value tensor cache for low-latency autoregressive decoding."""

    def __init__(self, max_batch_size: int, max_seq_len: int, num_kv_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.k = torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.current_len = 0

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache with newly computed key and value states.
        
        Args:
            k_new: (batch, num_kv_heads, seq_len_new, head_dim)
            v_new: (batch, num_kv_heads, seq_len_new, head_dim)
        """
        bsz, _, seq_len_new, _ = k_new.shape
        start = self.current_len
        end = start + seq_len_new

        if end > self.max_seq_len:
            # Reallocate larger cache if needed
            new_max = max(self.max_seq_len * 2, end + 1024)
            new_k = torch.zeros(self.max_batch_size, self.num_kv_heads, new_max, self.head_dim, device=self.device, dtype=self.dtype)
            new_v = torch.zeros(self.max_batch_size, self.num_kv_heads, new_max, self.head_dim, device=self.device, dtype=self.dtype)
            new_k[:, :, :start, :] = self.k[:, :, :start, :]
            new_v[:, :, :start, :] = self.v[:, :, :start, :]
            self.k = new_k
            self.v = new_v
            self.max_seq_len = new_max

        self.k[:bsz, :, start:end, :] = k_new
        self.v[:bsz, :, start:end, :] = v_new
        self.current_len = end

        return self.k[:bsz, :, :end, :], self.v[:bsz, :, :end, :]

    def reset(self) -> None:
        """Reset the cache pointer."""
        self.current_len = 0


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads along head dimension when num_queries > num_kv_heads."""
    if n_rep == 1:
        return x
    bsz, n_kv, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bsz, n_kv, n_rep, seq_len, head_dim)
        .reshape(bsz, n_kv * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) layer with RoPE and optional KV-cache."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 4096,
        rope_scaling: str = "yarn",
        scale_factor: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = AdvancedRotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            scaling_type=rope_scaling,
            scale_factor=scale_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for full sequence or incremental single-step decoding."""
        bsz, seq_len, _ = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        start_pos = kv_cache.current_len if kv_cache is not None else 0

        # 2. Rotary embedding
        cos, sin = self.rope(x, seq_len, start_pos=start_pos)
        q, k = AdvancedRotaryEmbedding.apply_rotary_emb(q, k, cos, sin, mscale=self.rope.mscale)

        # 3. Update KV cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # 4. Repeat KV heads to match query heads
        k = repeat_kv(k, self.num_queries_per_kv)
        v = repeat_kv(v, self.num_queries_per_kv)

        # 5. Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        return self.out_proj(context)
