"""Tests for dependency-directed backtracking and nogood constraint learning."""

import pytest
from noema.contracts.actions import TacticAction
from noema.contracts.states import LocalHypothesis, ProofGoal, ProofState
from noema.search.backtracking import (
    BacktrackDecision,
    ConflictAnalyzer,
    DependencyDirectedBacktracker,
    NogoodStore,
)


@pytest.fixture
def dummy_state() -> ProofState:
    g = ProofGoal(
        goal_id="g1",
        mvar_id="?m1",
        target="x = y",
        hypotheses=[
            LocalHypothesis(name="x", type_str="Nat"),
            LocalHypothesis(name="y", type_str="Nat"),
        ],
    )
    return ProofState(state_id="s1", goals=[g])


def test_nogood_store_pruning(dummy_state):
    store = NogoodStore()
    assert len(store) == 0

    # Action using conflicting premise
    act1 = TacticAction.from_string("rw [BadLemma.foo]")
    act2 = TacticAction.from_string("rfl")

    # Add a nogood for dummy_state prohibiting BadLemma.foo
    store.add_nogood(
        goal_fingerprint="dummy_fp",
        forbidden_premises=["BadLemma.foo"],
        reason="causes contradiction",
    )

    # Use state with matching fingerprint
    from noema.search.canonical import GoalCanonicalizer
    state_fp = GoalCanonicalizer.canonicalize_state(dummy_state)
    store.add_nogood(
        goal_fingerprint=state_fp,
        forbidden_premises=["BadLemma.foo"],
        reason="fatal type mismatch",
    )

    is_pruned1, reason1 = store.is_pruned(dummy_state, act1)
    assert is_pruned1
    assert "fatal type mismatch" in reason1
    assert store.hits_count == 1

    is_pruned2, _ = store.is_pruned(dummy_state, act2)
    assert not is_pruned2


def test_dependency_directed_backtracker_flow(dummy_state):
    ddb = DependencyDirectedBacktracker()

    # Step 0: Initial choice
    ddb.push_decision(dummy_state, TacticAction.from_string("intro x"))

    # Step 1: Bad premise introduced
    bad_action = TacticAction.from_string("have h := BadLemma.bar")
    ddb.push_decision(dummy_state, bad_action)

    # Step 2: Another benign choice
    ddb.push_decision(dummy_state, TacticAction.from_string("clear x"))

    assert len(ddb.decisions) == 3

    # Conflict occurs referencing BadLemma.bar
    failing_action = TacticAction.from_string("exact BadLemma.bar")
    target_level = ddb.handle_conflict(
        dummy_state,
        failing_action,
        error_message="unknown identifier BadLemma.bar",
    )

    # Jumped back to level 1 where BadLemma.bar was introduced
    assert target_level == 1
    assert len(ddb.decisions) == 1
    assert ddb.non_chronological_jumps == 1
    assert len(ddb.store) == 1
