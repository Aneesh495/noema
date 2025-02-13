"""Unit tests for JointLoss, CheckpointManager, NoemaTrainer, and ExpertIterationEngine."""

import tempfile
import torch
import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec
from noema.corpus.trace_extractor import TraceExtractor
from noema.formal.bridge import FormalBridge
from noema.formal.checker import IndependentChecker
from noema.models import TransformerConfig, NoemaTransformer
from noema.training import (
    JointLoss,
    CheckpointManager,
    NoemaTrainer,
    TrainingConfig,
    ExpertIterationEngine,
)


def test_joint_loss_components():
    loss_fn = JointLoss(lambda_policy=1.0, lambda_pointer=0.5, lambda_value=0.5, pad_token_id=0)

    # Batch of 2
    class_logits = torch.randn(2, 8, requires_grad=True)
    class_targets = torch.tensor([1, 3], dtype=torch.long)

    val_pred = torch.tensor([[0.8], [0.3]], requires_grad=True)
    val_targets = torch.tensor([[1.0], [0.0]])

    token_logits = torch.randn(2, 4, 16, requires_grad=True)
    # Token 0 in sequence 1 is padding (id 0)
    token_targets = torch.tensor([[5, 6, 7, 8], [1, 2, 0, 0]], dtype=torch.long)

    prem_scores = torch.randn(2, 4, requires_grad=True)
    prem_targets = torch.tensor([0, 2], dtype=torch.long)

    losses = loss_fn(
        tactic_class_logits=class_logits,
        tactic_class_targets=class_targets,
        value_pred=val_pred,
        value_targets=val_targets,
        token_logits=token_logits,
        token_targets=token_targets,
        premise_scores=prem_scores,
        premise_targets=prem_targets,
    )

    assert "total_loss" in losses
    assert losses["total_loss"].item() > 0.0
    losses["total_loss"].backward()
    assert class_logits.grad is not None
    assert val_pred.grad is not None


def test_checkpoint_atomic_save_and_deterministic_resume():
    config = TransformerConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=2, d_ff=64)
    model1 = NoemaTransformer(config)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    # Generate some random updates
    dummy_input = torch.randint(1, 100, (2, 8))
    dummy_mask = torch.ones(2, 8, dtype=torch.bool)
    out = model1(dummy_input, attention_mask=dummy_mask)
    loss = out["value"].sum()
    loss.backward()
    optimizer1.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        ckpt_path = mgr.save_checkpoint(
            "test_ckpt.pt",
            model=model1,
            optimizer=optimizer1,
            step=42,
            nonpadding_tokens=1000,
            best_loss=0.123,
        )
        assert ckpt_path.exists()

        # Load into new model
        model2 = NoemaTransformer(config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        meta = mgr.load_checkpoint("test_ckpt.pt", model=model2, optimizer=optimizer2)
        assert meta["step"] == 42
        assert meta["nonpadding_tokens"] == 1000
        assert meta["best_loss"] == 0.123

        # Compare outputs
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1(dummy_input, attention_mask=dummy_mask)
            out2 = model2(dummy_input, attention_mask=dummy_mask)
            assert torch.allclose(out1["value"], out2["value"])


def test_trainer_step_and_gradient_updates():
    config = TransformerConfig(vocab_size=100, d_model=32, n_layers=4, n_heads=2, d_ff=64)
    model = NoemaTransformer(config)
    t_cfg = TrainingConfig(learning_rate=1e-3, total_steps=10, warmup_steps=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = NoemaTrainer(model=model, config=t_cfg, checkpoint_dir=tmpdir)
        
        batch = {
            "input_ids": torch.randint(1, 100, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.bool),
            "tactic_class_targets": torch.tensor([0, 1], dtype=torch.long),
            "value_targets": torch.tensor([[1.0], [0.5]]),
        }

        record = trainer.train_step(batch)
        assert record["step"] == 1
        assert record["total_loss"] > 0.0
        assert record["nonpadding_tokens"] == 16
        assert record["min_block_grad"] >= 0.0


def test_expert_iteration_admission_and_rejection():
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )
    checker = IndependentChecker(env_spec=env)
    bridge = FormalBridge(env_spec=env)
    trace_extractor = TraceExtractor(bridge=bridge)

    engine = ExpertIterationEngine(checker=checker, trace_extractor=trace_extractor)

    # 1. Valid training theorem proof: should be admitted
    thm_valid = TheoremSpec(
        theorem_id="train_nat_001",
        name="test_valid_ind",
        module="Test.Basic",
        statement="∀ (x : Nat), x = x",
    )
    valid_candidates = [(thm_valid, "intro x\nrfl")]
    new_recs, summary = engine.process_discovered_proofs(valid_candidates)
    assert summary.accepted_proofs == 1
    assert summary.rejected_proofs == 0
    assert len(new_recs) > 0

    # 2. Invalid proof with sorry: should be rejected
    thm_invalid = TheoremSpec(
        theorem_id="train_nat_002",
        name="test_invalid_ind",
        module="Test.Basic",
        statement="∀ (x : Nat), x = 0",
    )
    invalid_candidates = [(thm_invalid, "intro x\nsorry")]
    _, summary2 = engine.process_discovered_proofs(invalid_candidates)
    assert summary2.accepted_proofs == 0
    assert summary2.rejected_proofs == 1

    # 3. Evaluation theorem: strictly rejected to avoid contamination
    thm_eval = TheoremSpec(
        theorem_id="eval_heldout_001",
        name="test_eval",
        module="Test.Basic",
        statement="∀ (x : Nat), x = x",
    )
    eval_candidates = [(thm_eval, "intro x\nrfl")]
    _, summary3 = engine.process_discovered_proofs(eval_candidates)
    assert summary3.accepted_proofs == 0
    assert summary3.rejected_proofs == 1
