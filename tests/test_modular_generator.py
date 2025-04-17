"""Tests for modular arithmetic and number theory theorem generator."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.corpus.generators.modular import ModularTheoremGenerator
from noema.formal.checker import IndependentChecker


@pytest.fixture
def lean_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )


def test_generator_initialization():
    gen = ModularTheoremGenerator(seed=123)
    assert len(gen.PREMISES_DIVISIBILITY) > 0
    assert len(gen.PREMISES_MODULAR) > 0
    assert len(gen.PREMISES_GCD) > 0


def test_generate_divisibility_transitivity():
    gen = ModularTheoremGenerator(seed=42)
    spec, tactics = gen.generate_divisibility_transitivity(idx=1)
    assert "a ∣ b → b ∣ c → a ∣ c" in spec.statement
    assert any("Nat.dvd_trans" in t for t in tactics)


def test_generate_mod_idempotence():
    gen = ModularTheoremGenerator(seed=42)
    spec, tactics = gen.generate_mod_idempotence(idx=1)
    assert "(a % m) % m = a % m" in spec.statement
    assert any("Nat.mod_mod" in t for t in tactics)


def test_generate_gcd_associativity():
    gen = ModularTheoremGenerator(seed=42)
    spec, tactics = gen.generate_gcd_associativity(idx=1)
    assert "Nat.gcd" in spec.statement
    assert any("Nat.gcd_assoc" in t for t in tactics)


def test_generate_batch():
    gen = ModularTheoremGenerator(seed=77)
    batch = gen.generate_batch(total=20)
    assert len(batch) == 20
    theorem_ids = {spec.theorem_id for spec, _ in batch}
    assert len(theorem_ids) == 20


def test_lean_verification_of_modular_theorems(lean_env):
    """Verify that generated number theory theorems pass real Lean 4 kernel verification."""
    checker = IndependentChecker(env_spec=lean_env)
    gen = ModularTheoremGenerator(seed=42)

    # 1. Divisibility transitivity
    spec_dvd, tactics_dvd = gen.generate_divisibility_transitivity(idx=99)
    cert_dvd, audit_dvd = checker.verify_proof(spec_dvd, "\n".join(tactics_dvd))
    assert audit_dvd.is_valid, f"Lean divisibility failed: {cert_dvd.kernel_diagnostics}"

    # 2. Mod idempotence
    spec_mod, tactics_mod = gen.generate_mod_idempotence(idx=99)
    cert_mod, audit_mod = checker.verify_proof(spec_mod, "\n".join(tactics_mod))
    assert audit_mod.is_valid, f"Lean mod idempotence failed: {cert_mod.kernel_diagnostics}"

    # 3. Add mod identity
    spec_add, tactics_add = gen.generate_add_mod_identity(idx=99)
    cert_add, audit_add = checker.verify_proof(spec_add, "\n".join(tactics_add))
    assert audit_add.is_valid, f"Lean add mod failed: {cert_add.kernel_diagnostics}"

    # 4. GCD commutativity
    spec_gcd, tactics_gcd = gen.generate_gcd_commutativity(idx=99)
    cert_gcd, audit_gcd = checker.verify_proof(spec_gcd, "\n".join(tactics_gcd))
    assert audit_gcd.is_valid, f"Lean gcd failed: {cert_gcd.kernel_diagnostics}"
