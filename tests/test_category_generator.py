"""Unit tests for category theory and monad laws theorem generator."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.corpus.generators.category import CategoryTheoremGenerator
from noema.formal.checker import IndependentChecker


@pytest.fixture
def lean_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Core"],
    )


def test_category_generator_initialization():
    gen = CategoryTheoremGenerator(seed=123)
    assert len(gen.PREMISES_CATEGORY) > 0


def test_generate_comp_associativity():
    gen = CategoryTheoremGenerator(seed=42)
    spec, tactics = gen.generate_comp_associativity(idx=1)
    assert "(h ∘ g) ∘ f = h ∘ (g ∘ f)" in spec.statement
    assert "rfl" in tactics


def test_generate_monad_laws():
    gen = CategoryTheoremGenerator(seed=42)
    spec_l, tac_l = gen.generate_option_monad_left_identity(idx=1)
    assert "(Option.some x >>= f) = f x" in spec_l.statement

    spec_r, tac_r = gen.generate_option_monad_right_identity(idx=1)
    assert "(m >>= Option.some) = m" in spec_r.statement

    spec_a, tac_a = gen.generate_option_monad_associativity(idx=1)
    assert "((m >>= f) >>= g)" in spec_a.statement


def test_generate_batch():
    gen = CategoryTheoremGenerator(seed=99)
    batch = gen.generate_batch(total=20)
    assert len(batch) == 20
    theorem_ids = {spec.theorem_id for spec, _ in batch}
    assert len(theorem_ids) == 20


def test_lean_verification_of_category_theorems(lean_env):
    """Verify that generated categorical theorems pass real Lean 4 kernel verification."""
    checker = IndependentChecker(env_spec=lean_env)
    gen = CategoryTheoremGenerator(seed=42)

    # 1. Composition associativity
    spec_comp, tac_comp = gen.generate_comp_associativity(idx=99)
    cert_comp, audit_comp = checker.verify_proof(spec_comp, "\n".join(tac_comp))
    assert audit_comp.is_valid, f"Comp assoc failed: {cert_comp.kernel_diagnostics}"

    # 2. Monad left identity
    spec_ml, tac_ml = gen.generate_option_monad_left_identity(idx=99)
    cert_ml, audit_ml = checker.verify_proof(spec_ml, "\n".join(tac_ml))
    assert audit_ml.is_valid, f"Monad left id failed: {cert_ml.kernel_diagnostics}"

    # 3. Monad right identity
    spec_mr, tac_mr = gen.generate_option_monad_right_identity(idx=99)
    cert_mr, audit_mr = checker.verify_proof(spec_mr, "\n".join(tac_mr))
    assert audit_mr.is_valid, f"Monad right id failed: {cert_mr.kernel_diagnostics}"

    # 4. Monad associativity
    spec_ma, tac_ma = gen.generate_option_monad_associativity(idx=99)
    cert_ma, audit_ma = checker.verify_proof(spec_ma, "\n".join(tac_ma))
    assert audit_ma.is_valid, f"Monad assoc failed: {cert_ma.kernel_diagnostics}"
