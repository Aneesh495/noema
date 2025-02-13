"""Unit tests for Noema 24-block Transformer architecture, parameter counts, and gradient flow."""

import pytest
import torch
from noema.models import (
    TransformerConfig,
    NoemaTransformer,
)


def test_transformer_24_blocks_and_parameter_threshold():
    config = TransformerConfig.flagship_24block()
    model = NoemaTransformer(config)

    assert len(model.blocks) == 24
    num_params = model.num_parameters
    # Verify >= 40 Million parameters gate
    assert num_params >= 40_000_000, f"Expected >= 40M parameters, got {num_params:,}"
    
    # Verify distinct blocks (no parameter sharing)
    distinct_ok, msg = model.verify_distinct_blocks()
    assert distinct_ok, msg


def test_transformer_forward_pass_and_shapes():
    # Use smaller d_ff for quick test execution
    config = TransformerConfig(
        vocab_size=500,
        d_model=128,
        n_layers=24,
        n_heads=4,
        d_ff=256,
    )
    model = NoemaTransformer(config)
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(1, 500, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # 4 candidate premises
    cand_vectors = torch.randn(batch_size, 4, config.d_model)
    cand_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            candidate_premise_vectors=cand_vectors,
            candidate_mask=cand_mask,
        )

    assert out["token_logits"].shape == (batch_size, seq_len, 500)
    assert out["tactic_class_logits"].shape == (batch_size, config.num_tactic_classes)
    assert out["value"].shape == (batch_size, 1)
    assert (out["value"] >= 0.0).all() and (out["value"] <= 1.0).all()
    assert len(out["activation_norms"]) == 24

    # Check that masked candidate premise slots get -inf
    p_scores = out["premise_scores"]
    assert p_scores is not None
    assert p_scores[0, 3] == float("-inf")
    assert p_scores[1, 2] == float("-inf")
    assert p_scores[1, 3] == float("-inf")


def test_all_24_blocks_receive_gradients():
    """Verify that every single block from 0 to 23 receives nonzero gradients during backprop."""
    config = TransformerConfig(
        vocab_size=200,
        d_model=64,
        n_layers=24,
        n_heads=4,
        d_ff=128,
    )
    model = NoemaTransformer(config)
    model.train()

    input_ids = torch.randint(1, 200, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)

    out = model(input_ids=input_ids, attention_mask=mask)
    loss = out["value"].sum() + out["tactic_class_logits"].sum()
    loss.backward()

    # Check each block has non-zero gradients on its parameters
    for idx, block in enumerate(model.blocks):
        grad_norms = [p.grad.norm().item() for p in block.parameters() if p.grad is not None]
        assert len(grad_norms) > 0, f"Block {idx} has no gradients"
        max_grad = max(grad_norms)
        assert max_grad > 0.0, f"Block {idx} received zero gradient"


def test_gradient_checkpointing_numerical_parity():
    config = TransformerConfig(
        vocab_size=200,
        d_model=64,
        n_layers=4,
        n_heads=2,
        d_ff=128,
        use_gradient_checkpointing=False,
    )
    torch.manual_seed(42)
    model = NoemaTransformer(config)
    model.eval()

    input_ids = torch.randint(1, 200, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)

    # Without checkpointing
    out1 = model(input_ids=input_ids, attention_mask=mask)
    
    # With checkpointing
    model.config.use_gradient_checkpointing = True
    out2 = model(input_ids=input_ids, attention_mask=mask)

    assert torch.allclose(out1["value"], out2["value"], atol=1e-5)
