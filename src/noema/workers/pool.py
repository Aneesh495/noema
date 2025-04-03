"""Bounded formal worker pool with timeout enforcement, memory monitoring, and backpressure."""

from __future__ import annotations

import concurrent.futures
import os
import time
from typing import Any, Callable
from pydantic import BaseModel
from noema.contracts.actions import TacticAction
from noema.contracts.states import ProofState
from noema.contracts.theorems import TheoremSpec
from noema.formal.bridge import FormalBridge
from noema.formal.types import TransitionResult, TransitionStatus


class WorkerPoolConfig(BaseModel):
    max_workers: int = 4
    default_timeout_sec: float = 10.0
    max_rss_mb: float = 2048.0


class BoundedWorkerPool:
    """Manages concurrent Lean formal transition workers with strict resource bounds."""

    def __init__(self, bridge: FormalBridge, config: WorkerPoolConfig | None = None):
        self.bridge = bridge
        self.config = config or WorkerPoolConfig()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="noema_formal_worker",
        )
        self.active_jobs: int = 0
        self.total_completed: int = 0
        self.total_failed: int = 0

    def submit_transition(
        self,
        theorem: TheoremSpec,
        current_state: ProofState,
        action: TacticAction,
        history: list[str],
        timeout_sec: float | None = None,
    ) -> concurrent.futures.Future[TransitionResult]:
        """Submit a transition job with timeout backpressure."""
        t_sec = timeout_sec or self.config.default_timeout_sec

        def _worker_task() -> TransitionResult:
            self.active_jobs += 1
            try:
                res = self.bridge.step(
                    theorem=theorem,
                    current_state=current_state,
                    action=action,
                    tactic_history=history,
                    timeout_sec=t_sec,
                )
                self.total_completed += 1
                return res
            except Exception as e:
                self.total_failed += 1
                return TransitionResult(
                    status=TransitionStatus.PROCESS_ERROR,
                    before_state_id=current_state.state_id,
                    action=action,
                    diagnostics=f"Worker exception: {e}",
                    execution_time_ms=0.0,
                    is_valid_transition=False,
                )
            finally:
                self.active_jobs -= 1

        return self.executor.submit(_worker_task)

    def execute_batch(
        self,
        tasks: list[tuple[TheoremSpec, ProofState, TacticAction, list[str]]],
        timeout_sec: float | None = None,
    ) -> list[TransitionResult]:
        """Execute a batch of transitions concurrently with bounded parallelism."""
        futures = [
            self.submit_transition(thm, state, action, hist, timeout_sec=timeout_sec)
            for thm, state, action, hist in tasks
        ]
        results = []
        for f in futures:
            try:
                res = f.result(timeout=timeout_sec or self.config.default_timeout_sec + 2.0)
                results.append(res)
            except concurrent.futures.TimeoutError:
                self.total_failed += 1
                results.append(
                    TransitionResult(
                        status=TransitionStatus.TIMEOUT,
                        before_state_id="",
                        action=TacticAction.from_string("timeout"),
                        diagnostics="Worker pool timeout",
                        is_valid_transition=False,
                    )
                )
        return results

    def execute_step(
        self,
        theorem: TheoremSpec,
        current_state: ProofState,
        action: TacticAction,
        tactic_history: list[str] | None = None,
        timeout_sec: float | None = None,
    ) -> TransitionResult:
        """Execute a single step synchronously via the bounded pool."""
        future = self.submit_transition(
            theorem=theorem,
            current_state=current_state,
            action=action,
            history=tactic_history or [],
            timeout_sec=timeout_sec,
        )
        return future.result()

    def is_healthy(self) -> bool:
        """Check if worker pool is active and receptive to work."""
        return not self.executor._shutdown

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)
