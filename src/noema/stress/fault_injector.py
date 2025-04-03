"""Chaos fault injection and worker process resiliency testing harness."""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from typing import Callable
from noema.contracts.actions import TacticAction
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec
from noema.formal.types import TransitionResult, TransitionStatus
from noema.workers.pool import BoundedWorkerPool


@dataclass(frozen=True)
class FaultInjectionEvent:
    """Record of an artificially injected worker fault."""
    fault_type: str  # "segfault", "sigkill", "timeout", "corrupted_output", "oom"
    worker_pid: int | None
    timestamp: float
    recovered: bool


class WorkerFaultInjector:
    """Injects synthetic crashes, timeouts, and process terminations into Lean worker pools."""

    def __init__(self, pool: BoundedWorkerPool):
        self.pool = pool
        self.history: list[FaultInjectionEvent] = []

    def inject_timeout_task(self, timeout_sec: float = 0.5) -> TransitionResult:
        """Submit a task guaranteed to exceed the allocated execution budget."""
        thm = TheoremSpec(
            theorem_id="stress_timeout",
            name="stress_timeout_lemma",
            module="Test.Stress",
            statement="∀ (n : Nat), n = n",
        )
        init_state, _ = self.pool.bridge.get_initial_state(thm)
        act = TacticAction.from_string("sleep_100_seconds")

        t0 = time.time()
        # Step with extremely short timeout
        res = self.pool.execute_step(thm, init_state, act, tactic_history=[], timeout_sec=timeout_sec)
        t_elapsed = time.time() - t0

        recovered = self.pool.is_healthy()
        self.history.append(
            FaultInjectionEvent(
                fault_type="timeout",
                worker_pid=None,
                timestamp=t0,
                recovered=recovered,
            )
        )
        return res

    def inject_corrupted_response(self) -> TransitionResult:
        """Verify graceful error reporting when worker returns malformed syntax or unparseable diagnostics."""
        thm = TheoremSpec(
            theorem_id="stress_corrupt",
            name="stress_corrupt_lemma",
            module="Test.Stress",
            statement="∀ (x : Nat), x = x",
        )
        init_state, _ = self.pool.bridge.get_initial_state(thm)
        # Corrupted tactic with null bytes or unclosed strings
        act = TacticAction.from_string("exact \x00\xff invalid_binary_noise")

        t0 = time.time()
        res = self.pool.execute_step(thm, init_state, act, tactic_history=[])
        recovered = self.pool.is_healthy()

        self.history.append(
            FaultInjectionEvent(
                fault_type="corrupted_output",
                worker_pid=None,
                timestamp=t0,
                recovered=recovered,
            )
        )
        return res

    def verify_pool_survives_burst_failures(self, num_failures: int = 5) -> bool:
        """Subject the worker pool to a rapid burst of failing tasks and verify continuous operation."""
        for _ in range(num_failures):
            self.inject_corrupted_response()

        # Submit a healthy legitimate task to ensure pool is still fully functional
        thm_healthy = TheoremSpec(
            theorem_id="stress_recovery_check",
            name="stress_recovery_check_lemma",
            module="Test.Stress",
            statement="∀ (x : Nat), x = x",
        )
        init_state, _ = self.pool.bridge.get_initial_state(thm_healthy)
        res = self.pool.execute_step(
            thm_healthy,
            init_state,
            TacticAction.from_string("intro x"),
            tactic_history=[],
        )
        return res.is_valid_transition and self.pool.is_healthy()
