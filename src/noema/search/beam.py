"""Diverse beam search and speculative tactic tree search for neural theorem proving."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence
from noema.contracts.actions import TacticAction
from noema.contracts.states import ProofState


@dataclass(frozen=True)
class BeamCandidate:
    """A candidate tactic sequence trajectory in the beam."""
    trajectory_id: str
    tactics: tuple[str, ...]
    cumulative_log_prob: float
    state: ProofState
    group_id: int = 0
    is_solved: bool = False

    @property
    def length(self) -> int:
        return len(self.tactics)

    def normalized_score(self, alpha: float = 0.6) -> float:
        """Compute length-normalized score: log_p / ((5 + length)^alpha / 6^alpha)."""
        if self.length == 0:
            return self.cumulative_log_prob
        penalty = ((5.0 + self.length) / 6.0) ** alpha
        return self.cumulative_log_prob / penalty


@dataclass
class BeamGroup:
    """A group in Diverse Beam Search maintaining a subset of candidate hypotheses."""
    group_id: int
    candidates: list[BeamCandidate] = field(default_factory=list)

    def best_candidate(self) -> BeamCandidate | None:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.normalized_score())


class DiverseBeamSearch:
    """Diverse Beam Search (Vijayakumar et al.) adapted for interactive theorem proving."""

    def __init__(
        self,
        num_groups: int = 3,
        beam_width: int = 4,
        diversity_penalty: float = 1.0,
        length_penalty_alpha: float = 0.6,
        max_depth: int = 15,
    ):
        self.num_groups = num_groups
        self.beam_width = beam_width
        self.diversity_penalty = diversity_penalty
        self.alpha = length_penalty_alpha
        self.max_depth = max_depth

    def search(
        self,
        initial_state: ProofState,
        policy_fn: Callable[[ProofState], list[tuple[str, float]]],
        step_fn: Callable[[ProofState, str], tuple[ProofState, bool]],
    ) -> list[BeamCandidate]:
        """Execute diverse beam search over tactic sequences.
        
        Args:
            initial_state: The starting root proof state with unproved goals.
            policy_fn: Maps state -> list of (tactic_text, log_prob) sorted by score.
            step_fn: Maps (state, tactic) -> (next_state, is_valid).
        """
        # Initialize group 0 with the empty root candidate
        groups = [BeamGroup(group_id=g) for g in range(self.num_groups)]
        for g in groups:
            g.candidates = [
                BeamCandidate(
                    trajectory_id="root",
                    tactics=(),
                    cumulative_log_prob=0.0,
                    state=initial_state,
                    group_id=g.group_id,
                    is_solved=len(initial_state.goals) == 0,
                )
            ]

        completed_solutions: list[BeamCandidate] = []

        for depth in range(self.max_depth):
            all_groups_idle = True
            selected_tactics_previous_groups: list[set[str]] = [set() for _ in range(self.num_groups)]

            for g_idx in range(self.num_groups):
                group = groups[g_idx]
                if not group.candidates:
                    continue

                candidates_to_expand = [c for c in group.candidates if not c.is_solved]
                if candidates_to_expand:
                    all_groups_idle = False

                expanded_pool: list[BeamCandidate] = []

                for cand in candidates_to_expand:
                    policy_proposals = policy_fn(cand.state)
                    for tactic_str, log_p in policy_proposals:
                        # Compute diversity penalty from earlier groups
                        div_pen = 0.0
                        for prev_g in range(g_idx):
                            if tactic_str in selected_tactics_previous_groups[prev_g]:
                                div_pen += self.diversity_penalty

                        next_state, valid = step_fn(cand.state, tactic_str)
                        if not valid:
                            continue

                        is_now_solved = len(next_state.goals) == 0
                        new_log_prob = cand.cumulative_log_prob + log_p - div_pen
                        new_cand = BeamCandidate(
                            trajectory_id=f"{cand.trajectory_id}_{len(cand.tactics)}_{tactic_str[:8]}",
                            tactics=cand.tactics + (tactic_str,),
                            cumulative_log_prob=new_log_prob,
                            state=next_state,
                            group_id=g_idx,
                            is_solved=is_now_solved,
                        )

                        if is_now_solved:
                            completed_solutions.append(new_cand)
                        else:
                            expanded_pool.append(new_cand)

                # Keep top beam_width candidates in this group
                expanded_pool.sort(key=lambda c: c.normalized_score(self.alpha), reverse=True)
                group.candidates = expanded_pool[: self.beam_width]

                # Record tactics selected by this group at this depth
                for c in group.candidates:
                    if c.tactics:
                        selected_tactics_previous_groups[g_idx].add(c.tactics[-1])

            if all_groups_idle or completed_solutions:
                break

        # Collect all candidates across groups
        all_candidates: list[BeamCandidate] = list(completed_solutions)
        for g in groups:
            all_candidates.extend(g.candidates)

        all_candidates.sort(key=lambda c: (c.is_solved, c.normalized_score(self.alpha)), reverse=True)
        return all_candidates


class SpeculativeTacticTree:
    """Speculatively verifies prefix branches to prune unproductive trajectories early."""

    def __init__(self, branching_factor: int = 3, max_speculative_depth: int = 2):
        self.branching_factor = branching_factor
        self.max_depth = max_speculative_depth

    def speculate(
        self,
        state: ProofState,
        policy_fn: Callable[[ProofState], list[tuple[str, float]]],
        step_fn: Callable[[ProofState, str], tuple[ProofState, bool]],
        current_depth: int = 0,
    ) -> list[tuple[list[str], ProofState, bool]]:
        """Recursively explore speculative lookahead trees up to max_depth.
        
        Returns list of (tactic_path, final_state, is_solved).
        """
        if current_depth >= self.max_depth or len(state.goals) == 0:
            return [([], state, len(state.goals) == 0)]

        proposals = policy_fn(state)[: self.branching_factor]
        results: list[tuple[list[str], ProofState, bool]] = []

        for tactic, _ in proposals:
            next_state, valid = step_fn(state, tactic)
            if not valid:
                continue

            if len(next_state.goals) == 0:
                results.append(([tactic], next_state, True))
                continue

            sub_results = self.speculate(next_state, policy_fn, step_fn, current_depth + 1)
            for sub_path, sub_state, sub_solved in sub_results:
                results.append(([tactic] + sub_path, sub_state, sub_solved))

        return results
