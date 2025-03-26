"""Tests for advanced RoPE scaling: NTK-aware, YaRN, and rotation invariants."""

import math
import pytest
import torch
from noema.models.rope import AdvancedRotaryEmbedding


def test_standard_rope_parity_at_scale_one():
    dim = 64
    seq_len = 128
    
    rope_std = AdvancedRotaryEmbedding(dim=dim, scaling_type="standard", scale_factor=1.0)
    rope_yarn = AdvancedRotaryEmbedding(dim=dim, scaling_type="yarn", scale_factor=1.0)
    rope_ntk = AdvancedRotaryEmbedding(dim=dim, scaling_type="ntk", scale_factor=1.0)

    # All should produce identical inverse frequencies when scale_factor=1.0
    assert torch.allclose(rope_std.inv_freq, rope_yarn.inv_freq, atol=1e-6)
    assert torch.allclose(rope_std.inv_freq, rope_ntk.inv_freq, atol=1e-6)


def test_ntk_aware_frequency_decay():
    dim = 64
    rope_ntk = AdvancedRotaryEmbedding(dim=dim, scaling_type="ntk", scale_factor=2.0)
    rope_std = AdvancedRotaryEmbedding(dim=dim, scaling_type="standard", scale_factor=1.0)

    # NTK scaling increases the base, hence lowering inverse frequencies
    assert torch.all(rope_ntk.inv_freq <= rope_std.inv_freq)
    # Highest frequency should remain close while lowest frequency drops significantly
    ratio_high = rope_ntk.inv_freq[0] / rope_std.inv_freq[0]
    ratio_low = rope_ntk.inv_freq[-1] / rope_std.inv_freq[-1]
    assert ratio_high > ratio_low


def test_yarn_frequency_bands():
    dim = 64
    rope_yarn = AdvancedRotaryEmbedding(
        dim=dim,
        scaling_type="yarn",
        scale_factor=4.0,
        beta_fast=32.0,
        beta_slow=1.0,
    )
    rope_std = AdvancedRotaryEmbedding(dim=dim, scaling_type="standard", scale_factor=1.0)

    # High frequencies (small wavelength) should be identical to standard RoPE
    assert torch.isclose(rope_yarn.inv_freq[0], rope_std.inv_freq[0], rtol=1e-4)
    # Low frequencies (large wavelength) should be divided by scale_factor
    assert torch.isclose(rope_yarn.inv_freq[-1], rope_std.inv_freq[-1] / 4.0, rtol=1e-2)


def test_rotary_embedding_orthogonal_rotation():
    dim = 32
    seq_len = 8
    rope = AdvancedRotaryEmbedding(dim=dim, scaling_type="yarn", scale_factor=2.0)
    x = torch.randn(1, seq_len, 4, dim)

    cos, sin = rope(x, seq_len)
    q_rot, k_rot = AdvancedRotaryEmbedding.apply_rotary_emb(x, x, cos, sin)

    # Vector norm must be preserved under 2D Givens rotations
    norm_orig = torch.norm(x, dim=-1)
    norm_rot = torch.norm(q_rot, dim=-1)
    assert torch.allclose(norm_orig, norm_rot, atol=1e-5)
