"""Tests for inductive datatype and structural induction theorem generator."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.corpus.generators.inductive import InductiveTheoremGenerator
from noema.formal.checker import IndependentChecker


@pytest.fixture
def lean_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic", "Init.Data.List.Basic"],
    )


def test_inductive_generator_batch():
    gen = InductiveTheoremGenerator(seed=999)
    batch = gen.generate_inductive_batch(count=20)
    assert len(batch) == 20
    for thm, proof in batch:
        assert thm.statement
        assert thm.family.startswith("inductive_")
        assert len(proof) >= 3


def test_inductive_peano_verified_in_lean(lean_env):
    gen = InductiveTheoremGenerator(seed=999)
    checker = IndependentChecker(env_spec=lean_env)

    # 1. Peano 0 + n = n via structural induction
    thm_p0, proof_p0 = gen.generate_peano_addition_zero(idx=1)
    cert_p0, audit_p0 = checker.verify_proof(thm_p0, "\n".join(proof_p0))
    assert audit_p0.is_valid, f"Peano zero_add proof failed: {cert_p0.kernel_diagnostics}"
