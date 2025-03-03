"""Unit and integration tests for MCTS proof search with PUCT."""

import pytest
from noema.contracts.actions import TacticAction
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec
from noema.formal.bridge import FormalBridge
from noema.search.mcts import MCTSProofSearch


@pytest.fixture
def lean_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )


def test_mcts_solves_simple_reflexivity(lean_env):
    bridge = FormalBridge(env_spec=lean_env)
    mcts = MCTSProofSearch(bridge=bridge, c_puct=1.414, max_depth=5)

    thm = TheoremSpec(
        theorem_id="mcts_test_refl",
        name="test_mcts_refl",
        module="Test.Basic",
        statement="∀ (x : Nat), x = x",
    )
    init_state, _ = bridge.get_initial_state(thm)

    # Policy function providing candidate tactics
    def mock_policy(state):
        if state.depth == 0:
            return [
                (TacticAction.from_string("intro x"), 0.8),
                (TacticAction.from_string("rfl"), 0.2),
            ]
        elif state.depth == 1:
            return [
                (TacticAction.from_string("rfl"), 0.9),
                (TacticAction.from_string("intro y"), 0.1),
            ]
        return []

    # Heuristic value function
    def mock_value(state):
        return 0.5

    proved, actions, stats = mcts.search(
        theorem=thm,
        initial_state=init_state,
        policy_fn=mock_policy,
        value_fn=mock_value,
        max_iterations=10,
    )

    assert proved
    assert len(actions) > 0
    assert stats["iterations"] <= 10
    assert stats["nodes"] > 1


def test_mcts_handles_unsolvable_gracefully(lean_env):
    bridge = FormalBridge(env_spec=lean_env)
    mcts = MCTSProofSearch(bridge=bridge, c_puct=1.414, max_depth=3)

    thm = TheoremSpec(
        theorem_id="mcts_test_false",
        name="test_mcts_false",
        module="Test.Basic",
        statement="1 = 2",
    )
    init_state, _ = bridge.get_initial_state(thm)

    def mock_policy(state):
        return [
            (TacticAction.from_string("rfl"), 0.5),
            (TacticAction.from_string("decide"), 0.5),
        ]

    def mock_value(state):
        return 0.1

    proved, actions, stats = mcts.search(
        theorem=thm,
        initial_state=init_state,
        policy_fn=mock_policy,
        value_fn=mock_value,
        max_iterations=5,
    )

    assert not proved
    assert len(actions) == 0
