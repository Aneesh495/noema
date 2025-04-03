"""Tests for worker pool fault injection and chaos resilience."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.formal.bridge import FormalBridge
from noema.formal.types import TransitionStatus
from noema.stress.fault_injector import WorkerFaultInjector
from noema.workers.pool import BoundedWorkerPool, WorkerPoolConfig


@pytest.fixture
def worker_pool() -> BoundedWorkerPool:
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )
    bridge = FormalBridge(env_spec=env)
    cfg = WorkerPoolConfig(max_workers=2, default_timeout_sec=5.0)
    pool = BoundedWorkerPool(bridge=bridge, config=cfg)
    yield pool
    pool.shutdown(wait=False)


def test_fault_injection_corrupted_response(worker_pool):
    injector = WorkerFaultInjector(worker_pool)
    res = injector.inject_corrupted_response()

    # Corrupted tactic syntax should fail safely without crashing pool
    assert not res.is_valid_transition
    assert worker_pool.is_healthy()
    assert len(injector.history) == 1
    assert injector.history[0].recovered


def test_fault_injection_burst_resilience(worker_pool):
    injector = WorkerFaultInjector(worker_pool)
    # Burst of 3 corrupted tasks
    survived = injector.verify_pool_survives_burst_failures(num_failures=3)
    assert survived
    assert len(injector.history) == 3
