"""Tests for first-order pattern unification and coupled metavariable tracking."""

import pytest
from noema.contracts.states import ProofGoal, ProofState
from noema.search.unification import (
    CoupledMetavariableTracker,
    Term,
    UnificationEngine,
    UnificationError,
)


def test_term_parsing():
    t1 = Term.parse("?m")
    assert t1.is_mvar
    assert t1.name == "?m"

    t2 = Term.parse("f(x, ?y)")
    assert not t2.is_mvar
    assert t2.name == "f"
    assert len(t2.args) == 2
    assert t2.args[0].name == "x"
    assert t2.args[1].is_mvar


def test_unification_simple_success():
    # unify ?x with const(a)
    t1 = Term.parse("?x")
    t2 = Term.parse("a")
    subst = UnificationEngine.unify(t1, t2)
    assert "?x" in subst
    assert subst["?x"].name == "a"

    # unify f(?x, b) with f(a, ?y)
    t3 = Term.parse("f(?x, b)")
    t4 = Term.parse("f(a, ?y)")
    subst2 = UnificationEngine.unify(t3, t4)
    assert subst2["?x"].name == "a"
    assert subst2["?y"].name == "b"


def test_unification_occurs_check_fails():
    # unify ?x with f(?x) -> occurs check failure
    t1 = Term.parse("?x")
    t2 = Term.parse("f(?x)")
    with pytest.raises(UnificationError, match="Occurs check failed"):
        UnificationEngine.unify(t1, t2)


def test_unification_symbol_clash():
    t1 = Term.parse("f(a)")
    t2 = Term.parse("g(a)")
    with pytest.raises(UnificationError, match="Symbol mismatch"):
        UnificationEngine.unify(t1, t2)


def test_coupled_mvar_propagation_across_goals():
    tracker = CoupledMetavariableTracker()
    
    # Assign ?m1 := Nat.add(x, y)
    ok1 = tracker.assign_mvar("?m1", "add(x, y)")
    assert ok1

    # Attempt conflicting assignment ?m1 := mul(a, b) -> should fail
    ok2 = tracker.assign_mvar("?m1", "mul(a, b)")
    assert not ok2

    # Propagate assignment to state containing coupled goal
    g = ProofGoal(
        goal_id="g1",
        mvar_id="?m0",
        target="eq(?m1, z)",
        hypotheses=[],
        coupled_mvars=["?m1"],
    )
    s = ProofState(state_id="s1", goals=[g])
    new_s = tracker.propagate_to_state(s)

    # The goal target should now be eq(add(x, y), z)
    assert new_s.goals[0].target == "eq(add(x, y), z)"
