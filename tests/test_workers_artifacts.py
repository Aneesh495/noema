"""Unit tests for BoundedWorkerPool, CertificateManager, and AxiomAuditor."""

import tempfile
from pathlib import Path
import pytest
from noema.artifacts import AxiomAuditor, CertificateManager
from noema.contracts import (
    EnvironmentSpec,
    ProofCertificate,
    TacticAction,
    TheoremSpec,
    VerificationStatus,
)
from noema.formal.bridge import FormalBridge
from noema.formal.types import TransitionStatus
from noema.workers import BoundedWorkerPool, WorkerPoolConfig


def test_bounded_worker_pool_concurrency():
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )
    bridge = FormalBridge(env_spec=env)
    config = WorkerPoolConfig(max_workers=2, default_timeout_sec=5.0)
    pool = BoundedWorkerPool(bridge=bridge, config=config)

    thm = TheoremSpec(
        theorem_id="t_conc",
        name="thm_conc_test",
        module="Test.Basic",
        statement="∀ (x : Nat), x = x",
    )
    init_state, _ = bridge.get_initial_state(thm)

    # Submit batch of 2 transitions
    tasks = [
        (thm, init_state, TacticAction.from_string("intro x"), []),
        (thm, init_state, TacticAction.from_string("rfl"), []),
    ]
    results = pool.execute_batch(tasks)
    pool.shutdown(wait=True)

    assert len(results) == 2
    assert pool.total_completed == 2


def test_certificate_manager_save_and_verify():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CertificateManager(artifacts_dir=tmpdir)
        cert = ProofCertificate(
            theorem_id="thm_001",
            theorem_name="test_thm",
            statement="∀ (x : Nat), x = x",
            statement_hash="hash_1",
            environment_hash="hash_env",
            tactic_script="intro x\nrfl",
            used_axioms=["propext"],
            status=VerificationStatus.VERIFIED,
        )
        saved_path = mgr.save_certificate(cert)
        assert saved_path.exists()
        
        # Verify replay lean file was created
        replay_path = Path(tmpdir) / "thm_001_replay.lean"
        assert replay_path.exists()
        with open(replay_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "theorem test_thm" in content
            assert "rfl" in content

        # Load and verify
        loaded_cert, audit = mgr.load_and_verify(saved_path)
        assert audit.is_valid
        assert audit.no_sorry
        assert audit.axioms_permitted


def test_axiom_auditor_rejects_unpermitted_axiom():
    auditor = AxiomAuditor(allowed_axioms={"propext"})
    cert = ProofCertificate(
        theorem_id="thm_fake",
        theorem_name="fake_thm",
        statement="x = y",
        statement_hash="h1",
        environment_hash="h2",
        tactic_script="intro x y\nrfl",
        used_axioms=["CustomOracleAxiom"],
        status=VerificationStatus.VERIFIED,
    )
    cert.seal()
    audit = auditor.audit_certificate(cert)
    assert not audit.is_valid
    assert not audit.axioms_permitted
    assert "CustomOracleAxiom" in audit.rejection_reason
