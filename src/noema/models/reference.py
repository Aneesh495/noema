"""Independent analytical float64 reference implementations for numerical tests."""

from __future__ import annotations

import math
import numpy as np
import torch


def analytical_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Exact float64 reference calculation for RMSNorm."""
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    normed = x / np.sqrt(variance + eps)
    return normed * weight


def analytical_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stabilized float64 softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def analytical_scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Exact float64 reference for scaled dot-product attention."""
    # q, k, v: [batch, heads, seq, dim]
    d_k = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    weights = analytical_softmax(scores, axis=-1)
    return np.matmul(weights, v)
