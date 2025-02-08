"""Validated proof trace extraction and state-action record indexing."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence
from pydantic import BaseModel, Field
from noema.contracts.actions import TacticAction
from noema.contracts.theorems import TheoremSpec
from noema.formal.bridge import FormalBridge


class TraceRecord(BaseModel):
    """An individual verified state-action-state transition record."""
    record_id: str
    theorem_id: str
    state_id: str
    target: str
    hypotheses: list[str] = Field(default_factory=list)
    action: str
    next_state_id: str
    used_premises: list[str] = Field(default_factory=list)
    metavariables: list[str] = Field(default_factory=list)
    is_terminal: bool = False
    depth: int = 0

    def compute_hash(self) -> str:
        s = f"{self.theorem_id}::{self.state_id}::{self.action}::{self.next_state_id}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class TraceExtractor:
    """Extracts verified state-action transitions from reference proofs."""

    def __init__(self, bridge: FormalBridge):
        self.bridge = bridge

    def extract_from_theorem(self, theorem: TheoremSpec) -> list[TraceRecord]:
        """Step through reference proof and record all valid state-action pairs."""
        if not theorem.reference_proof:
            return []

        # Parse tactic lines from reference proof (skipping 'by')
        lines = theorem.reference_proof.strip().splitlines()
        tactic_lines = []
        for line in lines:
            s = line.strip()
            if s.startswith("by"):
                s = s[2:].strip()
            if s:
                tactic_lines.append(s)

        if not tactic_lines:
            return []

        records = []
        current_state, _ = self.bridge.get_initial_state(theorem)
        history: list[str] = []

        for depth, tac in enumerate(tactic_lines):
            action = TacticAction.from_string(tac)
            res = self.bridge.step(
                theorem,
                current_state,
                action,
                tactic_history=history,
            )

            if not res.is_valid_transition:
                # If an intermediate step fails, stop trace extraction
                break

            rec_id = f"{theorem.theorem_id}_step_{depth}"
            open_goals = current_state.get_open_goals()
            target = open_goals[0].target if open_goals else ""
            hyps = [h.canonical_repr() for h in (open_goals[0].hypotheses if open_goals else [])]
            mvars = open_goals[0].coupled_mvars if open_goals else []

            next_id = res.after_state.state_id if res.after_state else "terminal"
            is_term = res.is_proved() or (res.after_state is not None and not res.after_state.has_open_goals())

            record = TraceRecord(
                record_id=rec_id,
                theorem_id=theorem.theorem_id,
                state_id=current_state.state_id,
                target=target,
                hypotheses=hyps,
                action=tac,
                next_state_id=next_id,
                used_premises=action.premises,
                metavariables=mvars,
                is_terminal=is_term,
                depth=depth,
            )
            records.append(record)

            history.append(tac)
            if res.after_state:
                current_state = res.after_state
            if is_term:
                break

        return records

    def batch_extract(self, theorems: Sequence[TheoremSpec]) -> list[TraceRecord]:
        """Extract all traces from a sequence of theorems."""
        all_records = []
        for th in theorems:
            recs = self.extract_from_theorem(th)
            all_records.extend(recs)
        return all_records
