"""Unit and integration tests for ProofSearchEngine and hypergraph search."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.states import LocalHypothesis, ProofGoal, ProofState
from noema.contracts.theorems import TheoremSpec
from noema.contracts.certificates import VerificationStatus
from noema.formal.bridge import FormalBridge
from noema.formal.checker import IndependentChecker
from noema.search import (
    SearchBudget,
    HypergraphEngine,
    SearchStrategy,
    PriorityScorer,
    ProofSearchEngine,
)


def test_hypergraph_engine_transitions_and_extraction():
    g0 = ProofGoal(goal_id="g0", target="x = x", mvar_id="?m.0")
    s0 = ProofState(state_id="s0", goals=[g0], depth=0)
    
    engine = HypergraphEngine(root_state=s0)
    assert not engine.is_root_proved()

    # Apply rfl transition yielding empty goals (proved state)
    s1 = ProofState(state_id="s1", goals=[], depth=1, is_solved=True)
    from noema.contracts.actions import TacticAction
    action = TacticAction.from_string("rfl")

    engine.register_transition(parent_state_id="s0", child_state=s1, action=action)
    assert engine.is_root_proved()
    
    script = engine.extract_proof_script()
    assert script == ["rfl"]


def test_priority_scorer_strategies():
    g = ProofGoal(goal_id="g0", target="P", mvar_id="?m.0")
    s = ProofState(state_id="s0", goals=[g], depth=2)
    
    # 1. BFS prioritizes purely by depth
    bfs_scorer = PriorityScorer(strategy=SearchStrategy.UNIFORM_BFS)
    assert bfs_scorer.compute_score(s) == 2.0

    # 2. Neural priority rewards higher value estimates
    neural_scorer = PriorityScorer(strategy=SearchStrategy.NEURAL_GUIDED, beta_value=2.0)
    score_low_val = neural_scorer.compute_score(s, value_estimate=0.1)
    score_high_val = neural_scorer.compute_score(s, value_estimate=0.9)
    # Higher value should yield lower score (higher priority)
    assert score_high_val < score_low_val


def test_proof_search_engine_proves_theorem():
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )
    bridge = FormalBridge(env_spec=env)
    checker = IndependentChecker(env_spec=env)
    
    searcher = ProofSearchEngine(
        bridge=bridge,
        checker=checker,
        strategy=SearchStrategy.UNIFORM_BFS,
    )
    
    # Simple multi-step theorem: intro then rfl
    thm = TheoremSpec(
        theorem_id="t_identity",
        name="thm_id_test",
        module="Test.Basic",
        statement="∀ (x : Nat), x = x",
    )
    
    budget = SearchBudget(max_transitions=20, max_wallclock_sec=15.0)
    result = searcher.search(thm, budget=budget)
    
    assert result.is_proved
    assert len(result.proof_script) > 0
    assert result.certificate is not None
    assert result.certificate.status == VerificationStatus.VERIFIED
    assert result.accounting.valid_transitions > 0


def test_proof_search_engine_budget_exhaustion():
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )
    bridge = FormalBridge(env_spec=env)
    checker = IndependentChecker(env_spec=env)
    
    searcher = ProofSearchEngine(
        bridge=bridge,
        checker=checker,
        strategy=SearchStrategy.UNIFORM_BFS,
    )
    
    # Unsolvable false theorem: should terminate without crashing
    thm = TheoremSpec(
        theorem_id="t_false",
        name="thm_false_test",
        module="Test.Basic",
        statement="∀ (x : Nat), x = x + 100",
    )
    
    # Constrain transitions strictly to 3
    budget = SearchBudget(max_transitions=3, max_wallclock_sec=5.0)
    result = searcher.search(thm, budget=budget)
    
    assert not result.is_proved
    assert result.accounting.attempted_transitions <= 4
