"""Dependency-directed backtracking, conflict analysis, and nogood clause learning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence
from noema.contracts.actions import TacticAction
from noema.contracts.states import ProofState
from noema.search.canonical import GoalCanonicalizer


@dataclass(frozen=True)
class Nogood:
    """A learned constraint representing a known failing goal or incompatible premise set."""
    nogood_id: str
    goal_fingerprint: str
    forbidden_premises: tuple[str, ...]
    conflict_reason: str
    level_introduced: int


@dataclass
class BacktrackDecision:
    """Record of a decision point along a proof search branch."""
    level: int
    state_id: str
    goal_fingerprint: str
    action_taken: TacticAction
    premises_used: list[str] = field(default_factory=list)


class NogoodStore:
    """In-memory index of learned failure constraints and incompatible premise combinations."""

    def __init__(self):
        self._by_goal_fp: dict[str, list[Nogood]] = {}
        self._all_nogoods: list[Nogood] = []
        self.hits_count: int = 0

    def add_nogood(
        self,
        goal_fingerprint: str,
        forbidden_premises: Sequence[str],
        reason: str,
        level: int = 0,
    ) -> Nogood:
        """Register a newly discovered conflict pattern."""
        sorted_prems = tuple(sorted(list(set(forbidden_premises))))
        content = f"{goal_fingerprint}::{','.join(sorted_prems)}::{reason}"
        ng_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        ng = Nogood(
            nogood_id=ng_id,
            goal_fingerprint=goal_fingerprint,
            forbidden_premises=sorted_prems,
            conflict_reason=reason,
            level_introduced=level,
        )
        self._all_nogoods.append(ng)
        self._by_goal_fp.setdefault(goal_fingerprint, []).append(ng)
        return ng

    def is_pruned(self, state: ProofState, action: TacticAction) -> tuple[bool, str | None]:
        """Check if executing action from state matches an existing nogood."""
        state_fp = GoalCanonicalizer.canonicalize_state(state)
        matching_nogoods = self._by_goal_fp.get(state_fp, [])

        action_prems = set(action.premises)
        for ng in matching_nogoods:
            # If the action introduces the exact forbidden premise set
            if ng.forbidden_premises:
                if set(ng.forbidden_premises).issubset(action_prems):
                    self.hits_count += 1
                    return True, f"Pruned by nogood {ng.nogood_id}: {ng.conflict_reason}"
            else:
                # Goal alone is universally failing for this action kind
                self.hits_count += 1
                return True, f"Pruned by goal nogood {ng.nogood_id}: {ng.conflict_reason}"

        return False, None

    def __len__(self) -> int:
        return len(self._all_nogoods)


class ConflictAnalyzer:
    """Analyzes search branch failures to determine root causes and non-chronological jump targets."""

    @classmethod
    def analyze_failure(
        cls,
        decision_stack: Sequence[BacktrackDecision],
        failed_state: ProofState,
        failed_action: TacticAction,
        diagnostics: str,
    ) -> tuple[int, list[str]]:
        """Identify the minimal conflict level and culpable premises.
        
        Returns (target_backtrack_level, culpable_premises).
        """
        if not decision_stack:
            return 0, failed_action.premises

        culpable = list(failed_action.premises)
        target_level = decision_stack[-1].level

        # If a specific premise caused the failure, find where that premise was first used or introduced
        for prem in culpable:
            for dec in decision_stack:
                if prem in dec.premises_used:
                    # Found ancestor decision that introduced the conflicting premise
                    target_level = min(target_level, dec.level)
                    break

        return target_level, culpable


class DependencyDirectedBacktracker:
    """Coordinates non-chronological backtracking and learning during proof search."""

    def __init__(self):
        self.store = NogoodStore()
        self.decisions: list[BacktrackDecision] = []
        self.non_chronological_jumps: int = 0

    def push_decision(
        self,
        state: ProofState,
        action: TacticAction,
    ) -> None:
        """Push a choice point onto the active branch stack."""
        level = len(self.decisions)
        fp = GoalCanonicalizer.canonicalize_state(state)
        dec = BacktrackDecision(
            level=level,
            state_id=state.state_id,
            goal_fingerprint=fp,
            action_taken=action,
            premises_used=list(action.premises),
        )
        self.decisions.append(dec)

    def handle_conflict(
        self,
        failed_state: ProofState,
        failed_action: TacticAction,
        error_message: str,
    ) -> int:
        """Learn a nogood and compute the non-chronological backtrack target level."""
        target_level, culpable = ConflictAnalyzer.analyze_failure(
            self.decisions,
            failed_state,
            failed_action,
            error_message,
        )

        state_fp = GoalCanonicalizer.canonicalize_state(failed_state)
        self.store.add_nogood(
            goal_fingerprint=state_fp,
            forbidden_premises=culpable,
            reason=error_message,
            level=target_level,
        )

        current_level = len(self.decisions)
        # Non-chronological if jumped more than 1 step back
        if current_level - target_level > 1:
            self.non_chronological_jumps += 1

        # Pop decisions back to target level
        self.decisions = self.decisions[:target_level]
        return target_level
