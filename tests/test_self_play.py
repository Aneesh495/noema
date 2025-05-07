"""Unit tests for curriculum self-play and regret matching."""

import pytest
from noema.contracts.theorems import TheoremSpec
from noema.training.self_play import (
    ConjectureMutator,
    CurriculumSelfPlayEngine,
    RegretMatchingPolicy,
    SelfPlayProofTask,
)


@pytest.fixture
def sample_theorem() -> TheoremSpec:
    return TheoremSpec(
        theorem_id="thm_selfplay_0",
        name="nat_zero_add",
        module="Init.Data.Nat.Basic",
        statement="∀ (x : Nat), 0 + x = x",
        family="arithmetic",
    )


def test_conjecture_mutator_generalization(sample_theorem):
    mutator = ConjectureMutator(seed=42)
    mutant = mutator.mutate_generalization(sample_theorem, idx=1)

    assert mutant.theorem_id != sample_theorem.theorem_id
    assert "∀ (z : Nat)" in mutant.statement
    assert mutant.family == "arithmetic_mutation"


def test_conjecture_mutator_duality():
    mutator = ConjectureMutator(seed=42)
    base = TheoremSpec(
        theorem_id="thm_lattice_0",
        name="lattice_absorb",
        module="Init.Core",
        statement="A ∧ (A ∨ B) ↔ A",
        family="lattice",
    )
    dual = mutator.mutate_duality(base, idx=1)
    assert dual.statement == "A ∨ (A ∧ B) ↔ A"


def test_regret_matching_policy():
    actions = ["intro", "simp", "omega", "exact"]
    policy = RegretMatchingPolicy(actions)

    # Initially uniform
    probs = policy.get_action_probabilities()
    assert all(abs(p - 0.25) < 1e-5 for p in probs.values())

    # Update regrets
    policy.update_regret(
        chosen_action="simp",
        actual_reward=0.2,
        counterfactual_rewards={"omega": 1.0, "simp": 0.2, "intro": 0.0, "exact": 0.0},
    )

    probs_after = policy.get_action_probabilities()
    # omega should have highest probability now
    assert probs_after["omega"] > probs_after["simp"]

    action = policy.select_action()
    assert action in actions


def test_curriculum_self_play_engine(sample_theorem):
    engine = CurriculumSelfPlayEngine(base_theorems=[sample_theorem], target_success_rate=0.5)
    assert len(engine.active_pool) == 1

    task = engine.sample_task()
    assert task.theorem.theorem_id == sample_theorem.theorem_id

    # Record fast success (should spawn mutant task)
    new_tasks = engine.record_attempt(task, solved=True, steps_taken=1)
    assert len(new_tasks) == 1
    assert len(engine.active_pool) == 2
    assert len(engine.solved_tasks) == 1

    # Record failure on mutant
    fail_task = new_tasks[0]
    engine.record_attempt(fail_task, solved=False, steps_taken=10)
    assert len(engine.failed_tasks) == 1
    assert fail_task.difficulty_score < 0.65
