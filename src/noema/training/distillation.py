"""Proof trace simplification, dead-end pruning, and self-imitation distillation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from noema.contracts.actions import TacticAction
from noema.contracts.theorems import TheoremSpec
from noema.formal.bridge import FormalBridge
from noema.formal.types import TransitionStatus


@dataclass(frozen=True)
class SimplifiedTrace:
    """A minimal, simplified proof trace with redundant exploratory steps removed."""
    theorem_id: str
    original_steps: tuple[str, ...]
    simplified_steps: tuple[str, ...]
    compression_ratio: float
    is_valid: bool


class TraceSimplifier:
    """Simplifies discovered proof traces by pruning detour tactics and dead hypotheses."""

    def __init__(self, bridge: FormalBridge):
        self.bridge = bridge

    def simplify_trace(self, theorem: TheoremSpec, tactic_steps: Sequence[str]) -> SimplifiedTrace:
        """Attempt greedy backward elimination of intermediate steps that can be safely skipped."""
        original = tuple(tactic_steps)
        if len(tactic_steps) <= 1:
            return SimplifiedTrace(
                theorem_id=theorem.theorem_id,
                original_steps=original,
                simplified_steps=original,
                compression_ratio=0.0,
                is_valid=True,
            )

        current_steps = list(tactic_steps)
        # Try removing steps from front to second-to-last
        i = 0
        while i < len(current_steps) - 1:
            # Candidate elimination: remove step i
            candidate = current_steps[:i] + current_steps[i + 1 :]
            if self._verify_script(theorem, candidate):
                # Successfully eliminated step i
                current_steps = candidate
            else:
                # Step i was necessary, proceed to next step
                i += 1

        comp_ratio = (len(original) - len(current_steps)) / float(len(original))
        return SimplifiedTrace(
            theorem_id=theorem.theorem_id,
            original_steps=original,
            simplified_steps=tuple(current_steps),
            compression_ratio=comp_ratio,
            is_valid=True,
        )

    def _verify_script(self, theorem: TheoremSpec, steps: Sequence[str]) -> bool:
        """Verify if a candidate tactic sequence successfully closes the theorem."""
        init_state, _ = self.bridge.get_initial_state(theorem)
        curr = init_state
        history: list[str] = []

        for step in steps:
            act = TacticAction.from_string(step)
            res = self.bridge.step(theorem, curr, act, tactic_history=history)
            if not res.is_valid_transition:
                return False
            history.append(step)
            if res.status == TransitionStatus.PROVED or (res.after_state and res.after_state.is_solved):
                return True
            curr = res.after_state

        return False


class SelfImitationBuffer:
    """Prioritized replay buffer collecting simplified proofs for expert iteration."""

    def __init__(self, max_capacity: int = 1000):
        self.max_capacity = max_capacity
        self.traces: list[SimplifiedTrace] = []

    def add_trace(self, trace: SimplifiedTrace) -> None:
        """Insert a simplified trace, maintaining capacity bounded by quality."""
        if not trace.is_valid or not trace.simplified_steps:
            return

        self.traces.append(trace)
        # Sort so highest compression and shortest proofs are preserved
        if len(self.traces) > self.max_capacity:
            self.traces.sort(key=lambda t: (t.compression_ratio, -len(t.simplified_steps)), reverse=True)
            self.traces.pop()

    def export_training_tuples(self) -> list[tuple[str, list[str]]]:
        """Export (theorem_id, simplified_proof_steps) for neural policy fine-tuning."""
        return [(t.theorem_id, list(t.simplified_steps)) for t in self.traces]

    def __len__(self) -> int:
        return len(self.traces)
