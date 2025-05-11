"""Unit tests for Lean 4 tactic tree builder, combinators, and optimizer."""

import pytest
from noema.formal.tactics import (
    AllGoalsTactic,
    AtomicTactic,
    FirstTactic,
    FocusTactic,
    RepeatTactic,
    SeqTactic,
    TacticTreeOptimizer,
    TryTactic,
)


def test_atomic_tactic_rendering_and_premises():
    t1 = AtomicTactic("rw", ("[Nat.add_comm]", "[Nat.add_assoc]"))
    rendered = t1.to_lean_str(indent_level=1)
    assert rendered == "  rw [Nat.add_comm] [Nat.add_assoc]"
    premises = t1.extract_premises()
    assert "Nat.add_comm" in premises
    assert "Nat.add_assoc" in premises


def test_unsafe_keyword_detection():
    safe = AtomicTactic("exact", ("hp",))
    assert not safe.contains_unsafe_keywords()

    unsafe1 = AtomicTactic("sorry")
    assert unsafe1.contains_unsafe_keywords()

    unsafe2 = AtomicTactic("admit")
    assert unsafe2.contains_unsafe_keywords()


def test_combinator_rendering():
    step1 = AtomicTactic("intro", ("x",))
    step2 = AtomicTactic("rfl")
    seq = SeqTactic((step1, step2))
    assert seq.complexity() == 2
    assert "intro x\nrfl" in seq.to_lean_str()

    rep = RepeatTactic(step2)
    assert "repeat\n  rfl" in rep.to_lean_str()

    fst = FirstTactic((step1, step2))
    fst_str = fst.to_lean_str()
    assert "first" in fst_str
    assert "| intro x" in fst_str
    assert "| rfl" in fst_str

    foc = FocusTactic(step2)
    assert "· rfl" in foc.to_lean_str()


def test_tactic_tree_optimizer_merges_intros():
    i1 = AtomicTactic("intro", ("x",))
    i2 = AtomicTactic("intro", ("y", "z"))
    step = AtomicTactic("rfl")
    seq = SeqTactic((i1, i2, step))

    opt = TacticTreeOptimizer.optimize(seq)
    assert isinstance(opt, SeqTactic)
    assert len(opt.steps) == 2
    assert opt.steps[0].name == "intro"
    assert opt.steps[0].args == ("x", "y", "z")
    assert opt.steps[1].name == "rfl"


def test_tactic_tree_optimizer_nested_repeat():
    base = AtomicTactic("simp")
    nested = RepeatTactic(RepeatTactic(base))
    opt = TacticTreeOptimizer.optimize(nested)
    assert isinstance(opt, RepeatTactic)
    assert opt.body == base
