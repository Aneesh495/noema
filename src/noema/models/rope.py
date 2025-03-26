"""Advanced Rotary Position Embeddings: NTK-aware frequency scaling and YaRN."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class AdvancedRotaryEmbedding(nn.Module):
    """Rotary Position Embedding supporting both standard RoPE, NTK-aware scaling, and YaRN.
    
    References:
    - Su et al. (2021) RoFormer: Enhanced Transformer with Rotary Position Embedding
    - bloc97 (2023) NTK-Aware Scaled RoPE
    - Peng et al. (2023) YaRN: Efficient Context Window Extension of Large Language Models
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        scaling_type: str = "yarn",  # "standard", "ntk", "yarn"
        scale_factor: float = 1.0,
        original_max_seq_len: int = 2048,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scale_factor = scale_factor
        self.original_max_seq_len = original_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cached cos and sin tables
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

        # YaRN attention scale factor
        if scaling_type == "yarn" and scale_factor > 1.0:
            # Temperature scaling constant
            self.mscale = float(0.1 * math.log(scale_factor) + 1.0) * attn_factor
        else:
            self.mscale = 1.0

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies according to the selected scaling strategy."""
        pos = torch.arange(0, self.dim, 2, dtype=torch.float32)

        if self.scaling_type == "standard" or self.scale_factor == 1.0:
            return 1.0 / (self.base ** (pos / self.dim))

        elif self.scaling_type == "ntk":
            # NTK-aware scaling modifies the base: base' = base * scale_factor^(dim / (dim - 2))
            ntk_base = self.base * (self.scale_factor ** (self.dim / (self.dim - 2)))
            return 1.0 / (ntk_base ** (pos / self.dim))

        elif self.scaling_type == "yarn":
            # YaRN frequency ramp interpolation
            inv_freq = 1.0 / (self.base ** (pos / self.dim))
            wavelength = 2.0 * math.pi / inv_freq

            # Compute wavelength boundaries
            low = self.original_max_seq_len / self.beta_fast
            high = self.original_max_seq_len / self.beta_slow

            # Compute smooth ramp weights gamma
            # For wavelength < low: gamma = 0 (no interpolation, keep high freq)
            # For wavelength > high: gamma = 1 (full linear interpolation)
            # In between: smooth linear/cosine transition
            gamma = torch.clamp((wavelength - low) / (high - low), min=0.0, max=1.0)
            
            # Interpolated frequency
            inv_freq_interpolated = inv_freq / self.scale_factor
            inv_freq_yarn = (1.0 - gamma) * inv_freq + gamma * inv_freq_interpolated
            return inv_freq_yarn

        else:
            raise ValueError(f"Unknown RoPE scaling_type: {self.scaling_type}")

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tensors sliced to seq_len and broadcastable to x."""
        if seq_len > self.cos_cached.shape[0]:
            # Dynamically recompute if context expands beyond precomputed cache
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            cos = freqs.cos()
            sin = freqs.sin()
        else:
            cos = self.cos_cached[:seq_len].to(x.device)
            sin = self.sin_cached[:seq_len].to(x.device)

        return cos, sin

    @staticmethod
    def apply_rotary_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mscale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to query and key tensors with optional YaRN mscale."""
        # q, k shape: (batch, seq_len, num_heads, head_dim) or (batch, num_heads, seq_len, head_dim)
        if q.dim() == 4 and q.shape[1] != cos.shape[0]:
            # Heads in dim 1: (batch, num_heads, seq_len, head_dim)
            cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, dim/2 or dim)
            sin = sin.unsqueeze(0).unsqueeze(1)
        elif q.dim() == 4:
            # Heads in dim 2: (batch, seq_len, num_heads, head_dim)
            cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif q.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Expand cos and sin to match full head_dim
        if cos.shape[-1] != q.shape[-1]:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)

        q_embed = (q * cos + rotate_half(q) * sin) * mscale
        k_embed = (k * cos + rotate_half(k) * sin) * mscale
        return q_embed, k_embed
