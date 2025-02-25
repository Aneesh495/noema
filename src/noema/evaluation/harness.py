"""Benchmark evaluation harness across controlled, public mathlib, and miniF2F suites."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence
from pydantic import BaseModel, Field
from noema.contracts.certificates import VerificationStatus
from noema.contracts.theorems import TheoremSpec
from noema.search.budget import SearchBudget
from noema.search.engine import ProofSearchEngine, SearchResult


class BenchmarkMetrics(BaseModel):
    benchmark_name: str
    total_theorems: int
    attempted: int
    solved: int
    solve_rate: float
    total_transitions: int
    valid_transitions: int
    avg_transitions_per_solved: float
    total_time_ms: float
    avg_time_ms: float
    outcomes: dict[str, int] = Field(default_factory=dict)


class BenchmarkHarness:
    """Executes evaluation across frozen benchmark suites under fixed budgets."""

    def __init__(self, search_engine: ProofSearchEngine):
        self.search_engine = search_engine

    def evaluate_suite(
        self,
        name: str,
        theorems: Sequence[TheoremSpec],
        budget: SearchBudget | None = None,
        max_theorems: int | None = None,
        log_callback: callable | None = None,
    ) -> tuple[list[SearchResult], BenchmarkMetrics]:
        """Evaluate a sequence of theorems under a fixed search budget."""
        b = budget or SearchBudget(max_transitions=30, max_wallclock_sec=5.0)
        target_theorems = list(theorems[:max_theorems]) if max_theorems else list(theorems)
        
        results: list[SearchResult] = []
        solved_count = 0
        total_transitions = 0
        valid_transitions = 0
        total_time = 0.0
        solved_transitions = []
        outcomes: dict[str, int] = {}

        for idx, thm in enumerate(target_theorems):
            res = self.search_engine.search(thm, budget=b)
            results.append(res)
            
            acc = res.accounting
            total_transitions += acc.attempted_transitions
            valid_transitions += acc.valid_transitions
            total_time += acc.total_time_ms
            
            reason = acc.termination_reason.value
            outcomes[reason] = outcomes.get(reason, 0) + 1

            if res.is_proved and res.certificate and res.certificate.status == VerificationStatus.VERIFIED:
                solved_count += 1
                solved_transitions.append(acc.valid_transitions)

            if log_callback:
                log_callback(idx + 1, len(target_theorems), res)

        num_attempted = len(target_theorems)
        solve_rate = float(solved_count) / max(1, num_attempted)
        avg_solved_trans = (
            sum(solved_transitions) / float(len(solved_transitions))
            if solved_transitions
            else 0.0
        )
        avg_time = total_time / max(1, num_attempted)

        metrics = BenchmarkMetrics(
            benchmark_name=name,
            total_theorems=len(theorems),
            attempted=num_attempted,
            solved=solved_count,
            solve_rate=solve_rate,
            total_transitions=total_transitions,
            valid_transitions=valid_transitions,
            avg_transitions_per_solved=avg_solved_trans,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            outcomes=outcomes,
        )

        return results, metrics
