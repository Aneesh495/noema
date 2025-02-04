"""Formal proof state, goal, and metavariable tracking contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from pydantic import BaseModel, Field


class LocalHypothesis(BaseModel):
    """Local context hypothesis in a goal."""
    name: str
    type_str: str
    val_str: str | None = None
    is_let_binding: bool = False

    def canonical_repr(self) -> str:
        if self.is_let_binding and self.val_str:
            return f"{self.name} : {self.type_str} := {self.val_str}"
        return f"{self.name} : {self.type_str}"


class ProofGoal(BaseModel):
    """An individual proof obligation (subgoal) with its own context and target."""
    goal_id: str
    target: str
    hypotheses: list[LocalHypothesis] = Field(default_factory=list)
    mvar_id: str = Field(..., description="Lean metavariable identifier, e.g. ?m.123")
    coupled_mvars: list[str] = Field(
        default_factory=list,
        description="List of other metavariables shared with sibling subgoals"
    )
    is_solved: bool = False

    def context_target_str(self) -> str:
        hyps = "\n".join(h.canonical_repr() for h in self.hypotheses)
        if hyps:
            return f"{hyps}\n⊢ {self.target}"
        return f"⊢ {self.target}"

    def goal_hash(self) -> str:
        ctx_hyps = [h.canonical_repr() for h in self.hypotheses]
        data = {
            "mvar": self.mvar_id,
            "target": self.target.strip(),
            "hypotheses": ctx_hyps,
            "coupled": sorted(self.coupled_mvars),
        }
        raw = json.dumps(data, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class MetavariableAssignment(BaseModel):
    """Assignment to a metavariable produced by an action."""
    mvar_id: str
    term_expr: str
    introduced_mvars: list[str] = Field(default_factory=list)


class ProofState(BaseModel):
    """Global proof state containing all active goals and metavariable assignments."""
    state_id: str
    goals: list[ProofGoal] = Field(default_factory=list)
    assignments: dict[str, str] = Field(
        default_factory=dict,
        description="Map of mvar_id to assigned term expression"
    )
    parent_state_id: str | None = None
    depth: int = 0
    is_terminal: bool = False
    is_solved: bool = False

    @classmethod
    def compute_state_id(
        cls,
        goals: list[ProofGoal],
        assignments: dict[str, str],
        env_hash: str = ""
    ) -> str:
        goal_hashes = [g.goal_hash() for g in goals if not g.is_solved]
        assign_data = sorted(assignments.items())
        data = {
            "goals": goal_hashes,
            "assignments": assign_data,
            "env_hash": env_hash,
        }
        raw = json.dumps(data, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_open_goals(self) -> list[ProofGoal]:
        return [g for g in self.goals if not g.is_solved]

    def has_open_goals(self) -> bool:
        return any(not g.is_solved for g in self.goals)
