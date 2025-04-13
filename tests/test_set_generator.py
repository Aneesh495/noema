"""Tests for propositional logic and set theory theorem generator."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.corpus.generators.set_theory import SetTheoryTheoremGenerator
from noema.formal.checker import IndependentChecker


@pytest.fixture
def lean_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Core"],
    )


def test_generator_initialization():
    gen = SetTheoryTheoremGenerator(seed=123)
    assert len(gen.PREMISES_LOGIC) > 0
    assert len(gen.PREMISES_SET) > 0


def test_generate_de_morgan():
    gen = SetTheoryTheoremGenerator(seed=42)
    spec, tactics = gen.generate_propositional_de_morgan(idx=1)
    assert "¬(A ∨ B) ↔ (¬A ∧ ¬B)" in spec.statement
    assert "constructor" in tactics
    assert len(tactics) >= 4


def test_generate_absorption():
    gen = SetTheoryTheoremGenerator(seed=42)
    spec, tactics = gen.generate_absorption_identity(idx=2)
    assert "(A ∧ (A ∨ B)) ↔ A" in spec.statement
    assert len(tactics) >= 4


def test_generate_distributivity():
    gen = SetTheoryTheoremGenerator(seed=42)
    spec, tactics = gen.generate_distributivity_law(idx=3)
    assert "((A ∧ B) ∨ (A ∧ C))" in spec.statement
    assert len(tactics) >= 4


def test_generate_implication_transitivity():
    gen = SetTheoryTheoremGenerator(seed=42)
    spec, tactics = gen.generate_implication_transitivity_chain(chain_len=3, idx=1)
    assert "P0 → P3" in spec.statement or "(P0 → P3)" in spec.statement
    assert any("exact" in t for t in tactics)


def test_generate_batch():
    gen = SetTheoryTheoremGenerator(seed=99)
    batch = gen.generate_batch(total=16)
    assert len(batch) == 16
    theorem_ids = {spec.theorem_id for spec, _ in batch}
    assert len(theorem_ids) == 16


def test_lean_verification_of_set_theorems(lean_env):
    """Verify that generated logic theorems pass real Lean 4 verification."""
    checker = IndependentChecker(env_spec=lean_env)
    gen = SetTheoryTheoremGenerator(seed=42)

    # Check absorption law in Lean
    spec_abs, tactics_abs = gen.generate_absorption_identity(idx=99)
    cert_abs, audit_abs = checker.verify_proof(spec_abs, "\n".join(tactics_abs))
    assert audit_abs.is_valid, f"Lean absorption failed: {cert_abs.kernel_diagnostics}"

    # Check contraposition in Lean
    spec_cp, tactics_cp = gen.generate_contraposition(idx=99)
    cert_cp, audit_cp = checker.verify_proof(spec_cp, "\n".join(tactics_cp))
    assert audit_cp.is_valid, f"Lean contraposition failed: {cert_cp.kernel_diagnostics}"

    # Check de Morgan in Lean
    spec_dm, tactics_dm = gen.generate_propositional_de_morgan(idx=99)
    cert_dm, audit_dm = checker.verify_proof(spec_dm, "\n".join(tactics_dm))
    assert audit_dm.is_valid, f"Lean de Morgan failed: {cert_dm.kernel_diagnostics}"

    # Check intersection comm in Lean
    spec_int, tactics_int = gen.generate_intersection_comm(idx=99)
    cert_int, audit_int = checker.verify_proof(spec_int, "\n".join(tactics_int))
    assert audit_int.is_valid, f"Lean intersection failed: {cert_int.kernel_diagnostics}"
