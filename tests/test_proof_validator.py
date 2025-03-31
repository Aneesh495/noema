"""Tests for ProofTermASTValidator and axiom dependency analysis."""

import pytest
from noema.artifacts.proof_term_validator import ProofTermASTValidator
from noema.contracts.certificates import VerificationStatus
from noema.contracts.theorems import TheoremSpec


@pytest.fixture
def sample_theorem() -> TheoremSpec:
    return TheoremSpec(
        theorem_id="t1",
        name="test_id",
        module="Test.Basic",
        statement="∀ (x : Nat), x = x",
    )


def test_parse_s_expression_tree():
    text = "(fun (x : Nat) (rfl x))"
    ast = ProofTermASTValidator.parse_s_expression(text)
    assert ast.term_kind == "lam"
    assert ast.head == "fun"
    assert len(ast.args) == 2
    assert ProofTermASTValidator._compute_depth(ast) == 3
    assert ProofTermASTValidator._compute_node_count(ast) == 6


def test_validate_term_detects_unassigned_mvars(sample_theorem):
    term = "(fun (x : Nat) (?m.42 x))"
    res = ProofTermASTValidator.validate_term(sample_theorem, term)
    assert not res.is_valid
    assert res.status == VerificationStatus.REJECTED
    assert "?m.42" in res.unassigned_mvars


def test_validate_term_detects_sorry(sample_theorem):
    term = "(fun (x : Nat) (sorryAx Nat false))"
    res = ProofTermASTValidator.validate_term(sample_theorem, term)
    assert not res.is_valid
    assert "sorryAx" in res.diagnostics


def test_validate_term_constructive_filtering(sample_theorem):
    term = "(fun (x : Nat) (Classical.choice x))"
    
    # Allowed in standard classical logic
    res_classical = ProofTermASTValidator.validate_term(
        sample_theorem,
        term,
        allowed_axioms=["Classical.choice"],
        require_constructive=False,
    )
    assert res_classical.is_valid

    # Rejected when constructive proofs are mandated
    res_constructive = ProofTermASTValidator.validate_term(
        sample_theorem,
        term,
        allowed_axioms=["Classical.choice"],
        require_constructive=True,
    )
    assert not res_constructive.is_valid
    assert "Classical.choice" in res_constructive.diagnostics


def test_validate_clean_proof_term(sample_theorem):
    term = "(fun (x : Nat) (Eq.refl x))"
    res = ProofTermASTValidator.validate_term(sample_theorem, term)
    assert res.is_valid
    assert res.status == VerificationStatus.VERIFIED
    assert res.node_count > 0
