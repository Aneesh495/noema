"""Tests for tactic elaboration, AST parsing, and proof term analysis."""

import pytest
from noema.contracts.theorems import TheoremSpec
from noema.formal.elaboration import (
    ElaborationError,
    ProofTermSummary,
    TacticASTNode,
    TacticElaborator,
)


def test_decompose_script_semicolons_and_comments():
    script = """
    -- Begin proof
    intro x y;
    /- multi-line comment
       with multiple explanations -/
    rw [Nat.add_comm]; rfl
    """
    steps = TacticElaborator.decompose_script(script)
    assert steps == ["intro x y", "rw [Nat.add_comm]", "rfl"]


def test_extract_premises_varieties():
    step1 = "rw [Nat.add_comm, Nat.zero_add]"
    prems1 = TacticElaborator.extract_premises(step1)
    assert prems1 == ["Nat.add_comm", "Nat.zero_add"]

    step2 = "apply Nat.le_trans"
    prems2 = TacticElaborator.extract_premises(step2)
    assert prems2 == ["Nat.le_trans"]

    step3 = "simp only [← Nat.add_assoc, h1]"
    prems3 = TacticElaborator.extract_premises(step3)
    assert prems3 == ["Nat.add_assoc", "h1"]


def test_parse_tactic_tree():
    script = "intro x\nrw [Nat.add_comm]\nrfl"
    nodes = TacticElaborator.parse_tactic_tree(script)
    assert len(nodes) == 3
    assert not nodes[0].is_terminal
    assert nodes[1].premises == ["Nat.add_comm"]
    assert nodes[2].is_terminal


def test_audit_proof_soundness_catches_sorry():
    sound_script = "intro x\nrfl"
    is_sound, msg = TacticElaborator.audit_proof_soundness(sound_script)
    assert is_sound

    unsafe_script = "intro x\nsorry"
    is_sound2, msg2 = TacticElaborator.audit_proof_soundness(unsafe_script)
    assert not is_sound2
    assert "sorry" in msg2

    admit_script = "intro x\nadmit"
    is_sound3, msg3 = TacticElaborator.audit_proof_soundness(admit_script)
    assert not is_sound3
    assert "admit" in msg3


def test_summarize_proof_metrics():
    thm = TheoremSpec(
        theorem_id="t1",
        name="test_thm",
        module="Test.Basic",
        statement="∀ (n : Nat), n + 0 = n",
    )
    steps = ["intro n", "rfl"]
    summary = TacticElaborator.summarize_proof(thm, steps)
    assert summary.tactic_count == 2
    assert summary.is_axiom_free
    assert not summary.has_sorry
    assert summary.complexity_score > 0.0


def test_format_lean_theorem():
    thm = TheoremSpec(
        theorem_id="t1",
        name="add_zero_refl",
        module="Test.Basic",
        statement="∀ (n : Nat), n + 0 = n",
        namespace="TestMath",
    )
    formatted = TacticElaborator.format_lean_theorem(thm, ["intro n", "rfl"])
    assert "namespace TestMath" in formatted
    assert "theorem add_zero_refl : ∀ (n : Nat), n + 0 = n := by" in formatted
    assert "  intro n" in formatted
    assert "  rfl" in formatted
    assert "end TestMath" in formatted
