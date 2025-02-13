"""Verified training-only expert iteration engine."""

from __future__ import annotations

from typing import Sequence
from pydantic import BaseModel, Field
from noema.contracts.certificates import VerificationStatus
from noema.contracts.theorems import TheoremSpec
from noema.corpus.trace_extractor import TraceExtractor, TraceRecord
from noema.formal.checker import IndependentChecker


class ExpertIterationSummary(BaseModel):
    iteration_round: int
    attempted_proofs: int = 0
    accepted_proofs: int = 0
    rejected_proofs: int = 0
    new_trace_records: int = 0
    acceptance_rate: float = 0.0


class ExpertIterationEngine:
    """Collects and audits newly discovered proofs from the training task pool."""

    def __init__(
        self,
        checker: IndependentChecker,
        trace_extractor: TraceExtractor,
    ):
        self.checker = checker
        self.trace_extractor = trace_extractor
        self.replay_buffer: list[TraceRecord] = []
        self.iteration_round = 0

    def process_discovered_proofs(
        self,
        discovered_candidates: list[tuple[TheoremSpec, str]],
    ) -> tuple[list[TraceRecord], ExpertIterationSummary]:
        """Verify candidate proofs and admit validated ones to the training replay buffer."""
        self.iteration_round += 1
        accepted = 0
        rejected = 0
        new_records = []

        for thm, script in discovered_candidates:
            # Prevent benchmark evaluation contamination
            if "eval" in thm.theorem_id or "heldout" in thm.theorem_id or "minif2f" in thm.theorem_id:
                rejected += 1
                continue

            # Independent verification in fresh Lean process
            cert, audit = self.checker.verify_proof(thm, script)
            if cert.status == VerificationStatus.VERIFIED and audit.is_valid:
                accepted += 1
                # Create a version of the theorem with the verified proof for trace extraction
                thm_verified = thm.model_copy(update={"reference_proof": f"by\n  {script}"})
                extracted = self.trace_extractor.extract_from_theorem(thm_verified)
                new_records.extend(extracted)
            else:
                rejected += 1

        self.replay_buffer.extend(new_records)
        rate = accepted / max(1, len(discovered_candidates))

        summary = ExpertIterationSummary(
            iteration_round=self.iteration_round,
            attempted_proofs=len(discovered_candidates),
            accepted_proofs=accepted,
            rejected_proofs=rejected,
            new_trace_records=len(new_records),
            acceptance_rate=rate,
        )

        return new_records, summary
