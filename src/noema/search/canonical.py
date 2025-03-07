"""Alpha-equivalence canonicalization, variable renaming, and search cycle pruning."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence
from noema.contracts.states import LocalHypothesis, ProofGoal, ProofState


@dataclass(frozen=True)
class CanonicalGoal:
    """A mathematically canonical representation of a proof goal invariant under renaming."""
    fingerprint: str
    canonical_target: str
    canonical_hypotheses: tuple[tuple[str, str], ...]
    coupled_mvars_count: int


class GoalCanonicalizer:
    """Canonicalizes Lean 4 proof goals to enable deduplication and cycle pruning."""

    _IDENT_REGEX = re.compile(r"\b[A-Za-z_][A-Za-z0-9_.]*\b")
    _MVAR_REGEX = re.compile(r"\?[A-Za-z0-9_.]+")

    @classmethod
    def canonicalize_goal(cls, goal: ProofGoal) -> CanonicalGoal:
        """Convert a proof goal into its alpha-equivalent canonical form.
        
        Systematically maps local variable names to `_h0, _h1, ...` in declaration order,
        and metavariables to `?m0, ?m1, ...`.
        """
        var_map: dict[str, str] = {}
        mvar_map: dict[str, str] = {}

        # 1. Map hypotheses in local context order
        canon_hyps: list[tuple[str, str]] = []
        for idx, h in enumerate(goal.hypotheses):
            canon_name = f"_h{idx}"
            var_map[h.name] = canon_name
            # Rename in hypothesis type
            renamed_type = cls._substitute_tokens(h.type_str, var_map, mvar_map)
            canon_hyps.append((canon_name, renamed_type))

        # 2. Map metavariables in target
        for idx, mv in enumerate(goal.coupled_mvars):
            mvar_map[mv] = f"?m{idx}"
        if goal.mvar_id and goal.mvar_id not in mvar_map:
            mvar_map[goal.mvar_id] = f"?m{len(mvar_map)}"

        # 3. Rename in target
        canon_target = cls._substitute_tokens(goal.target, var_map, mvar_map)

        # 4. Compute deterministic fingerprint
        rep = f"{canon_target}::" + ";;".join(f"{n}:{t}" for n, t in canon_hyps)
        fingerprint = hashlib.sha256(rep.encode("utf-8")).hexdigest()

        return CanonicalGoal(
            fingerprint=fingerprint,
            canonical_target=canon_target,
            canonical_hypotheses=tuple(canon_hyps),
            coupled_mvars_count=len(goal.coupled_mvars),
        )

    @classmethod
    def canonicalize_state(cls, state: ProofState) -> str:
        """Compute an alpha-invariant fingerprint for an entire multi-goal proof state."""
        if not state.goals:
            return "empty_solved_state"

        goal_fingerprints = [
            cls.canonicalize_goal(g).fingerprint
            for g in state.goals
        ]
        # Sort goal fingerprints so goal order permutations are recognized as equivalent
        combined = "||".join(sorted(goal_fingerprints))
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @classmethod
    def are_goals_equivalent(cls, g1: ProofGoal, g2: ProofGoal) -> bool:
        """Return True if two goals are mathematically alpha-equivalent."""
        cg1 = cls.canonicalize_goal(g1)
        cg2 = cls.canonicalize_goal(g2)
        return cg1.fingerprint == cg2.fingerprint

    @classmethod
    def are_states_equivalent(cls, s1: ProofState, s2: ProofState) -> bool:
        """Return True if two proof states represent identical goal sets modulo alpha-renaming."""
        return cls.canonicalize_state(s1) == cls.canonicalize_state(s2)

    @classmethod
    def _substitute_tokens(
        cls,
        text: str,
        var_map: dict[str, str],
        mvar_map: dict[str, str],
    ) -> str:
        """Safely substitute variable and metavariable identifiers without replacing substrings."""
        if not text:
            return ""

        # First replace metavariables
        def replace_mvar(match: re.Match) -> str:
            val = match.group(0)
            return mvar_map.get(val, val)

        text = cls._MVAR_REGEX.sub(replace_mvar, text)

        # Next replace variables
        def replace_ident(match: re.Match) -> str:
            val = match.group(0)
            return var_map.get(val, val)

        return cls._IDENT_REGEX.sub(replace_ident, text)


class CyclePruner:
    """Detects and prunes repetitive tactical cycles and equivalent state recurrences."""

    def __init__(self, max_recurrence_allowed: int = 1):
        self.max_recurrence = max_recurrence_allowed
        self._ancestor_fingerprints: dict[str, int] = {}

    def is_cycle(self, state: ProofState, path_states: Sequence[ProofState]) -> bool:
        """Determine if transitioning to `state` creates an unproductive loop along `path_states`."""
        target_fp = GoalCanonicalizer.canonicalize_state(state)
        
        # Count occurrences of identical canonical state along active path
        count = 0
        for ancestor in path_states:
            anc_fp = GoalCanonicalizer.canonicalize_state(ancestor)
            if anc_fp == target_fp:
                count += 1
                if count >= self.max_recurrence:
                    return True
        return False
