"""Search budget definitions, resource constraints, and detailed accounting."""

from __future__ import annotations

import time
from enum import Enum
from pydantic import BaseModel, Field


class TerminationReason(str, Enum):
    PROVED = "proved"
    TRANSITIONS_EXHAUSTED = "transitions_exhausted"
    TIMEOUT = "timeout"
    PROPOSALS_EXHAUSTED = "proposals_exhausted"
    QUEUE_EMPTY = "queue_empty"
    WORKER_ERROR = "worker_error"


class SearchBudget(BaseModel):
    """Enforceable computational and formal transition budgets."""
    max_transitions: int = 100
    max_proposals: int = 500
    max_wallclock_sec: float = 30.0
    max_tokens: int = 200_000
    beam_width: int = 8


class SearchAccounting(BaseModel):
    """Comprehensive resource and outcome tracking for an individual proof search attempt."""
    theorems_attempted: int = 1
    generated_proposals: int = 0
    attempted_transitions: int = 0
    valid_transitions: int = 0
    rejected_actions: int = 0
    duplicate_states: int = 0
    subgoal_closures: int = 0
    kernel_time_ms: float = 0.0
    neural_time_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_consumed: int = 0
    is_proved: bool = False
    termination_reason: TerminationReason = TerminationReason.TRANSITIONS_EXHAUSTED

    def can_continue(self, budget: SearchBudget, start_time: float) -> tuple[bool, TerminationReason | None]:
        elapsed_sec = time.perf_counter() - start_time
        if elapsed_sec >= budget.max_wallclock_sec:
            return False, TerminationReason.TIMEOUT
        if self.attempted_transitions >= budget.max_transitions:
            return False, TerminationReason.TRANSITIONS_EXHAUSTED
        if self.generated_proposals >= budget.max_proposals:
            return False, TerminationReason.PROPOSALS_EXHAUSTED
        return True, None
