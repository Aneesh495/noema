"""Formal transition types and error categories."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field
from noema.contracts.actions import TacticAction
from noema.contracts.states import ProofGoal, ProofState


class TransitionStatus(str, Enum):
    SUCCESS = "success"
    PROVED = "proved"
    INVALID_TACTIC = "invalid_tactic"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    PROCESS_ERROR = "process_error"
    UNRESOLVED_METAVARS = "unresolved_metavars"
    HOLE_DETECTED = "hole_detected"


class TransitionResult(BaseModel):
    """Result of attempting a formal tactic transition."""
    status: TransitionStatus
    before_state_id: str
    action: TacticAction
    after_state: ProofState | None = None
    open_goals: list[ProofGoal] = Field(default_factory=list)
    diagnostics: str = ""
    execution_time_ms: float = 0.0
    is_valid_transition: bool = False
    state_changed: bool = False

    def is_proved(self) -> bool:
        return self.status == TransitionStatus.PROVED or (
            self.status == TransitionStatus.SUCCESS
            and self.after_state is not None
            and not self.after_state.has_open_goals()
        )
