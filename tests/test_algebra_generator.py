"""Tests for algebraic domain theorem generator."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.corpus.generators.algebra import AlgebraTheoremGenerator
from noema.formal.checker import IndependentChecker


@pytest.fixture
def lean_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )


def test_algebra_generator_generation_batch():
    gen = AlgebraTheoremGenerator(seed=123)
    batch = gen.generate_algebra_batch(count=15)
    assert len(batch) == 15
    for thm, proof in batch:
        assert thm.statement
        assert thm.family.startswith("algebra_")
        assert len(proof) >= 2


def test_algebra_theorems_verified_in_lean(lean_env):
    gen = AlgebraTheoremGenerator(seed=42)
    checker = IndependentChecker(env_spec=lean_env)

    # Test monoid chain
    thm_m, proof_m = gen.generate_monoid_associativity_chain(chain_len=3, idx=1)
    cert_m, audit_m = checker.verify_proof(thm_m, "\n".join(proof_m))
    assert audit_m.is_valid, f"Monoid proof failed: {cert_m.kernel_diagnostics}"

    # Test group cancellation
    thm_c, proof_c = gen.generate_group_cancellation(idx=1)
    cert_c, audit_c = checker.verify_proof(thm_c, "\n".join(proof_c))
    assert audit_c.is_valid, f"Cancellation proof failed: {cert_c.kernel_diagnostics}"

    # Test ring distributivity
    thm_r, proof_r = gen.generate_ring_distributivity(idx=1)
    cert_r, audit_r = checker.verify_proof(thm_r, "\n".join(proof_r))
    assert audit_r.is_valid, f"Ring proof failed: {cert_r.kernel_diagnostics}"
