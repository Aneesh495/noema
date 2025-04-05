"""Tests for trace simplification, dead-end pruning, and self-imitation buffer."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec
from noema.formal.bridge import FormalBridge
from noema.training.distillation import (
    SelfImitationBuffer,
    SimplifiedTrace,
    TraceSimplifier,
)


@pytest.fixture
def bridge() -> FormalBridge:
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )
    return FormalBridge(env_spec=env)


def test_trace_simplifier_pruning_redundant_step(bridge):
    simplifier = TraceSimplifier(bridge=bridge)

    thm = TheoremSpec(
        theorem_id="t_refl",
        name="test_refl_simple",
        module="Test.Basic",
        statement="∀ (x : Nat), x = x",
    )

    # Proof containing an unnecessary detour step:
    # 1. intro x
    # 2. have h : 1 = 1 := rfl  (redundant dead hypothesis!)
    # 3. rfl
    steps = ["intro x", "have h : 1 = 1 := rfl", "rfl"]

    simplified = simplifier.simplify_trace(thm, steps)
    assert simplified.is_valid
    # The redundant step should be pruned, leaving ['intro x', 'rfl']
    assert len(simplified.simplified_steps) < len(steps)
    assert simplified.simplified_steps == ("intro x", "rfl")
    assert simplified.compression_ratio > 0.0


def test_self_imitation_buffer_capacity():
    buf = SelfImitationBuffer(max_capacity=2)
    assert len(buf) == 0

    t1 = SimplifiedTrace(
        theorem_id="t1",
        original_steps=("a", "b", "c"),
        simplified_steps=("a", "c"),
        compression_ratio=0.33,
        is_valid=True,
    )
    t2 = SimplifiedTrace(
        theorem_id="t2",
        original_steps=("x", "y", "z", "w"),
        simplified_steps=("x", "w"),
        compression_ratio=0.50,
        is_valid=True,
    )
    t3 = SimplifiedTrace(
        theorem_id="t3",
        original_steps=("p", "q"),
        simplified_steps=("p", "q"),
        compression_ratio=0.0,
        is_valid=True,
    )

    buf.add_trace(t1)
    buf.add_trace(t2)
    buf.add_trace(t3)

    assert len(buf) == 2
    # Highest compression ratios (t2 and t1) should be kept
    tuples = buf.export_training_tuples()
    assert len(tuples) == 2
    tids = [tid for tid, _ in tuples]
    assert "t2" in tids
