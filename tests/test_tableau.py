"""Unit tests for analytic semantic tableau automated theorem prover."""

import pytest
from noema.search.tableau import (
    And,
    Atom,
    Imp,
    Not,
    Or,
    SemanticTableauProver,
    SignedFormula,
    TableauBranch,
)


def test_tableau_branch_closure():
    p = Atom("P")
    branch = TableauBranch(
        formulas=[
            SignedFormula(sign=True, formula=p),
            SignedFormula(sign=False, formula=p),
        ]
    )
    assert branch.check_closure()
    assert branch.is_closed
    assert branch.closing_atom == "P"


def test_prove_identity_implication():
    # P -> P
    p = Atom("P")
    formula = Imp(p, p)
    prover = SemanticTableauProver()
    valid, branches = prover.prove(formula)
    assert valid
    assert all(b.is_closed for b in branches)


def test_prove_law_of_excluded_middle():
    # P ∨ ¬P
    p = Atom("P")
    formula = Or(p, Not(p))
    prover = SemanticTableauProver()
    valid, branches = prover.prove(formula)
    assert valid


def test_prove_modus_ponens():
    # (P ∧ (P → Q)) → Q
    p = Atom("P")
    q = Atom("Q")
    premise = And(p, Imp(p, q))
    formula = Imp(premise, q)

    prover = SemanticTableauProver()
    valid, branches = prover.prove(formula)
    assert valid


def test_prove_transitivity_of_implication():
    # (P → Q) → (Q → R) → (P → R)
    p = Atom("P")
    q = Atom("Q")
    r = Atom("R")
    formula = Imp(Imp(p, q), Imp(Imp(q, r), Imp(p, r)))

    prover = SemanticTableauProver()
    valid, branches = prover.prove(formula)
    assert valid


def test_non_tautology_fails_refutation():
    # P ∨ Q is not a tautology
    formula = Or(Atom("P"), Atom("Q"))
    prover = SemanticTableauProver()
    valid, branches = prover.prove(formula)
    assert not valid
    assert any(not b.is_closed for b in branches)
