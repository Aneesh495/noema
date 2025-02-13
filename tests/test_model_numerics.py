"""Independent analytical float64 reference and numerical boundary tests."""

import numpy as np
import pytest
import torch
from noema.models.layers import MultiHeadAttention, RMSNorm
from noema.models.reference import (
    analytical_rmsnorm,
    analytical_scaled_dot_product_attention,
    analytical_softmax,
)


def test_rmsnorm_float64_parity():
    dim = 32
    eps = 1e-6
    x_np = np.random.randn(4, 16, dim).astype(np.float64)
    w_np = np.random.randn(dim).astype(np.float64)

    # Reference calculation
    ref_out = analytical_rmsnorm(x_np, w_np, eps=eps)

    # PyTorch calculation in double precision
    norm_layer = RMSNorm(dim=dim, eps=eps).to(torch.float64)
    with torch.no_grad():
        norm_layer.weight.copy_(torch.from_numpy(w_np))
        x_torch = torch.from_numpy(x_np)
        py_out = norm_layer(x_torch).numpy()

    np.testing.assert_allclose(ref_out, py_out, rtol=1e-7, atol=1e-7)


def test_attention_all_padding_handling():
    """Verify that an all-padding sequence does not produce NaN crashes."""
    d_model = 32
    n_heads = 2
    attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)

    # Batch with one valid sequence and one all-padding sequence
    x = torch.randn(2, 8, d_model)
    # Sequence 0 is valid, sequence 1 is all padding (zeros in mask)
    mask = torch.tensor([[1] * 8, [0] * 8], dtype=torch.bool)

    out = attn(x, attention_mask=mask)
    assert not torch.isnan(out).any(), "Output contains NaN values from masked attention"


def test_attention_padding_invariance():
    """Verify that changing padded positions does not affect valid positions."""
    d_model = 32
    n_heads = 2
    attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    attn.eval()

    # Base sequence of length 4
    x1 = torch.randn(1, 4, d_model)
    mask1 = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        out1 = attn(x1, attention_mask=mask1)

    # Padded sequence of length 8 with garbage in padded slots [4:8]
    x2 = torch.zeros(1, 8, d_model)
    x2[:, :4] = x1
    x2[:, 4:] = torch.randn(1, 4, d_model) * 100.0 # Extreme garbage
    mask2 = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)

    with torch.no_grad():
        out2 = attn(x2, attention_mask=mask2)

    # First 4 positions should match out1 within numerical tolerance
    diff = (out1 - out2[:, :4]).abs().max().item()
    assert diff < 1e-4, f"Padding leakage detected: max diff {diff}"
