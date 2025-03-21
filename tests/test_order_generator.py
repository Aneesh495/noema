"""Tests for order theory and lattice theorem generator."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.corpus.generators.order import OrderTheoremGenerator
from noema.formal.checker import IndependentChecker


@pytest.fixture
def lean_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )


def test_order_generator_batch():
    gen = OrderTheoremGenerator(seed=777)
    batch = gen.generate_order_batch(count=20)
    assert len(batch) == 20
    for thm, proof in batch:
        assert thm.statement
        assert thm.family.startswith("order_")
        assert len(proof) >= 2


def test_order_theorems_verified_in_lean(lean_env):
    gen = OrderTheoremGenerator(seed=777)
    checker = IndependentChecker(env_spec=lean_env)

    # 1. Transitivity chain
    thm_t, proof_t = gen.generate_transitivity_chain(chain_len=3, idx=1)
    cert_t, audit_t = checker.verify_proof(thm_t, "\n".join(proof_t))
    assert audit_t.is_valid, f"Transitivity proof failed: {cert_t.kernel_diagnostics}"

    # 2. Monotonicity sum
    thm_m, proof_m = gen.generate_monotonicity_sum(idx=1)
    cert_m, audit_m = checker.verify_proof(thm_m, "\n".join(proof_m))
    assert audit_m.is_valid, f"Monotonicity proof failed: {cert_m.kernel_diagnostics}"

    # 3. Min/max lattice
    thm_l, proof_l = gen.generate_min_max_lattice(idx=1)
    cert_l, audit_l = checker.verify_proof(thm_l, "\n".join(proof_l))
    assert audit_l.is_valid, f"Lattice proof failed: {cert_l.kernel_diagnostics}"

    # 4. Strict inequality to le
    thm_s, proof_s = gen.generate_trichotomy_decision(idx=1)
    cert_s, audit_s = checker.verify_proof(thm_s, "\n".join(proof_s))
    assert audit_s.is_valid, f"Trichotomy proof failed: {cert_s.kernel_diagnostics}"
