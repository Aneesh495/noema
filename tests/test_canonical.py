"""Tests for goal canonicalization, alpha-equivalence, and cycle pruning."""

import pytest
from noema.contracts.states import LocalHypothesis, ProofGoal, ProofState
from noema.search.canonical import CanonicalGoal, CyclePruner, GoalCanonicalizer


def test_goal_canonicalization_variable_renaming_invariance():
    # Goal 1: x : Nat, y : Nat |- x + y = y + x
    g1 = ProofGoal(
        goal_id="g1",
        mvar_id="?m1",
        target="x + y = y + x",
        hypotheses=[
            LocalHypothesis(name="x", type_str="Nat"),
            LocalHypothesis(name="y", type_str="Nat"),
        ],
    )

    # Goal 2: a : Nat, b : Nat |- a + b = b + a
    g2 = ProofGoal(
        goal_id="g2",
        mvar_id="?m2",
        target="a + b = b + a",
        hypotheses=[
            LocalHypothesis(name="a", type_str="Nat"),
            LocalHypothesis(name="b", type_str="Nat"),
        ],
    )

    cg1 = GoalCanonicalizer.canonicalize_goal(g1)
    cg2 = GoalCanonicalizer.canonicalize_goal(g2)

    assert cg1.fingerprint == cg2.fingerprint
    assert cg1.canonical_target == "_h0 + _h1 = _h1 + _h0"
    assert GoalCanonicalizer.are_goals_equivalent(g1, g2)


def test_goal_canonicalization_detects_non_equivalent():
    g1 = ProofGoal(
        goal_id="g1",
        mvar_id="?m1",
        target="x + y = y + x",
        hypotheses=[
            LocalHypothesis(name="x", type_str="Nat"),
            LocalHypothesis(name="y", type_str="Nat"),
        ],
    )
    g2 = ProofGoal(
        goal_id="g2",
        mvar_id="?m1",
        target="x * y = y * x",
        hypotheses=[
            LocalHypothesis(name="x", type_str="Nat"),
            LocalHypothesis(name="y", type_str="Nat"),
        ],
    )

    assert not GoalCanonicalizer.are_goals_equivalent(g1, g2)


def test_state_canonicalization_permutation_invariance():
    gA = ProofGoal(
        goal_id="ga",
        mvar_id="?ma",
        target="A",
        hypotheses=[],
    )
    gB = ProofGoal(
        goal_id="gb",
        mvar_id="?mb",
        target="B",
        hypotheses=[],
    )

    s1 = ProofState(
        state_id="s1",
        goals=[gA, gB],
    )
    s2 = ProofState(
        state_id="s2",
        goals=[gB, gA],
    )

    assert GoalCanonicalizer.are_states_equivalent(s1, s2)


def test_cycle_pruner_detects_loops():
    pruner = CyclePruner(max_recurrence_allowed=1)

    g1 = ProofGoal(
        goal_id="g1",
        mvar_id="?m1",
        target="n + 0 = n",
        hypotheses=[LocalHypothesis(name="n", type_str="Nat")],
    )
    g2 = ProofGoal(
        goal_id="g2",
        mvar_id="?m1",
        target="k + 0 = k",
        hypotheses=[LocalHypothesis(name="k", type_str="Nat")],
    )

    s1 = ProofState(state_id="s1", goals=[g1])
    s2 = ProofState(state_id="s2", goals=[g2])

    path = [s1]
    # s2 is alpha-equivalent to s1, so transitioning to s2 is a cycle
    assert pruner.is_cycle(s2, path)

    # A different state is not a cycle
    g3 = ProofGoal(
        goal_id="g3",
        mvar_id="?m1",
        target="k + 1 = 1 + k",
        hypotheses=[LocalHypothesis(name="k", type_str="Nat")],
    )
    s3 = ProofState(state_id="s3", goals=[g3])
    assert not pruner.is_cycle(s3, path)
