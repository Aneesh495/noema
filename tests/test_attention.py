"""Tests for Grouped-Query Attention (GQA) and incremental KV-caching."""

import pytest
import torch
from noema.models.attention import GroupedQueryAttention, KVCache, repeat_kv


def test_repeat_kv_dimensions():
    # 2 batches, 2 kv heads, 4 seq len, 8 head dim -> repeat by 4 to get 8 heads
    x = torch.randn(2, 2, 4, 8)
    repeated = repeat_kv(x, n_rep=4)
    assert repeated.shape == (2, 8, 4, 8)


def test_grouped_query_attention_forward_shape():
    d_model = 128
    num_heads = 8
    num_kv_heads = 2
    seq_len = 16
    bsz = 2

    gqa = GroupedQueryAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    x = torch.randn(bsz, seq_len, d_model)
    out = gqa(x)

    assert out.shape == (bsz, seq_len, d_model)


def test_kv_cache_incremental_parity():
    d_model = 64
    num_heads = 4
    num_kv_heads = 2
    bsz = 1

    gqa = GroupedQueryAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
    )
    gqa.eval()

    # Prefix of 3 tokens
    prefix = torch.randn(bsz, 3, d_model)
    # Next token
    next_token = torch.randn(bsz, 1, d_model)
    full_seq = torch.cat([prefix, next_token], dim=1)

    # 1. Full sequence pass with causal mask
    causal_mask = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)
    full_out = gqa(full_seq, mask=causal_mask)

    # 2. Incremental pass using KVCache
    cache = KVCache(
        max_batch_size=bsz,
        max_seq_len=10,
        num_kv_heads=num_kv_heads,
        head_dim=d_model // num_heads,
        device=prefix.device,
    )
    # Process prefix
    prefix_mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
    gqa(prefix, kv_cache=cache, mask=prefix_mask)

    # Process next token
    step_out = gqa(next_token, kv_cache=cache)

    # The representation of the 4th token in full pass should closely match the cached step output
    assert torch.allclose(full_out[:, 3:4, :], step_out, atol=1e-4)
