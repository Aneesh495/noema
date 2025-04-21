"""Unit tests for diverse beam search and speculative tactic tree search."""

import pytest
from noema.contracts.states import ProofGoal, ProofState
from noema.search.beam import (
    BeamCandidate,
    DiverseBeamSearch,
    SpeculativeTacticTree,
)


def make_test_state(state_id: str, num_goals: int) -> ProofState:
    goals = [
        ProofGoal(
            goal_id=f"g_{i}",
            target=f"Target_{i}",
            mvar_id=f"?m_{i}",
        )
        for i in range(num_goals)
    ]
    return ProofState(state_id=state_id, goals=goals, depth=0)


def test_beam_candidate_normalized_score():
    state = make_test_state("s0", 1)
    cand = BeamCandidate(
        trajectory_id="t1",
        tactics=("intro x", "exact x"),
        cumulative_log_prob=-0.5,
        state=state,
        group_id=0,
    )
    assert cand.length == 2
    # length penalty = ((5 + 2) / 6)^0.6 > 1, so normalized score is scaled
    score = cand.normalized_score(alpha=0.6)
    assert score < 0.0


def test_diverse_beam_search_finds_proof():
    init_state = make_test_state("root", 2)
    solved_state = make_test_state("solved", 0)

    def dummy_policy(state: ProofState) -> list[tuple[str, float]]:
        if len(state.goals) == 2:
            return [("intro x", -0.1), ("constructor", -0.4), ("simp", -0.8)]
        elif len(state.goals) == 1:
            return [("exact h", -0.05), ("rfl", -0.2)]
        return []

    def dummy_step(state: ProofState, tactic: str) -> tuple[ProofState, bool]:
        if tactic == "intro x":
            return make_test_state("s_intro", 1), True
        if tactic == "constructor":
            return make_test_state("s_constr", 1), True
        if tactic == "exact h":
            return solved_state, True
        if tactic == "rfl":
            return solved_state, True
        return state, False

    beam = DiverseBeamSearch(num_groups=2, beam_width=2, diversity_penalty=0.5, max_depth=5)
    candidates = beam.search(init_state, dummy_policy, dummy_step)

    assert len(candidates) > 0
    # Best candidate should be solved
    best = candidates[0]
    assert best.is_solved
    assert len(best.tactics) == 2
    assert best.tactics[0] in ["intro x", "constructor"]
    assert best.tactics[1] in ["exact h", "rfl"]


def test_diverse_beam_search_diversity_enforcement():
    init_state = make_test_state("root", 1)
    # Both groups should receive penalties for repeating tactics
    def repetitive_policy(state: ProofState) -> list[tuple[str, float]]:
        return [("tactic_A", -0.1), ("tactic_B", -0.2)]

    def dummy_step(state: ProofState, tactic: str) -> tuple[ProofState, bool]:
        return make_test_state(f"next_{tactic}", 1), True

    beam = DiverseBeamSearch(num_groups=2, beam_width=1, diversity_penalty=2.0, max_depth=1)
    candidates = beam.search(init_state, repetitive_policy, dummy_step)
    assert len(candidates) >= 2


def test_speculative_tactic_tree():
    init_state = make_test_state("root", 1)
    solved_state = make_test_state("solved", 0)

    def policy_fn(state: ProofState) -> list[tuple[str, float]]:
        return [("try_fast", -0.1), ("try_deep", -0.3)]

    def step_fn(state: ProofState, tactic: str) -> tuple[ProofState, bool]:
        if tactic == "try_fast":
            return solved_state, True
        if tactic == "try_deep":
            return make_test_state("deep", 1), True
        return state, False

    speculator = SpeculativeTacticTree(branching_factor=2, max_speculative_depth=2)
    spec_results = speculator.speculate(init_state, policy_fn, step_fn)

    assert len(spec_results) >= 2
    # At least one path solved
    solved_paths = [path for path, _, is_sol in spec_results if is_sol]
    assert len(solved_paths) >= 1
    assert solved_paths[0] == ["try_fast"]
