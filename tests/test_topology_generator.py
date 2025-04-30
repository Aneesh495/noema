"""Unit tests for point-set topology and continuous map theorem generator."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.corpus.generators.topology import TopologyTheoremGenerator
from noema.formal.checker import IndependentChecker


@pytest.fixture
def lean_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Core"],
    )


def test_generator_initialization():
    gen = TopologyTheoremGenerator(seed=123)
    assert len(gen.PREMISES_TOPOLOGY) > 0


def test_generate_preimage_intersection():
    gen = TopologyTheoremGenerator(seed=42)
    spec, tactics = gen.generate_preimage_intersection(idx=1)
    assert "s (f x) ∧ t (f x)" in spec.statement
    assert "constructor" in tactics


def test_generate_preimage_union():
    gen = TopologyTheoremGenerator(seed=42)
    spec, tactics = gen.generate_preimage_union(idx=1)
    assert "s (f x) ∨ t (f x)" in spec.statement
    assert "constructor" in tactics


def test_generate_preimage_monotonicity():
    gen = TopologyTheoremGenerator(seed=42)
    spec, tactics = gen.generate_preimage_monotonicity(idx=1)
    assert "∀ y, s y → t y" in spec.statement
    assert any("exact h (f x) hs" in t for t in tactics)


def test_generate_batch():
    gen = TopologyTheoremGenerator(seed=99)
    batch = gen.generate_batch(total=20)
    assert len(batch) == 20
    theorem_ids = {spec.theorem_id for spec, _ in batch}
    assert len(theorem_ids) == 20


def test_lean_verification_of_topology_theorems(lean_env):
    """Verify that generated topology theorems pass real Lean 4 kernel verification."""
    checker = IndependentChecker(env_spec=lean_env)
    gen = TopologyTheoremGenerator(seed=42)

    # 1. Preimage intersection
    spec_int, tactics_int = gen.generate_preimage_intersection(idx=99)
    cert_int, audit_int = checker.verify_proof(spec_int, "\n".join(tactics_int))
    assert audit_int.is_valid, f"Lean preimage intersection failed: {cert_int.kernel_diagnostics}"

    # 2. Preimage union
    spec_un, tactics_un = gen.generate_preimage_union(idx=99)
    cert_un, audit_un = checker.verify_proof(spec_un, "\n".join(tactics_un))
    assert audit_un.is_valid, f"Lean preimage union failed: {cert_un.kernel_diagnostics}"

    # 3. Preimage monotonicity
    spec_mono, tactics_mono = gen.generate_preimage_monotonicity(idx=99)
    cert_mono, audit_mono = checker.verify_proof(spec_mono, "\n".join(tactics_mono))
    assert audit_mono.is_valid, f"Lean preimage monotonicity failed: {cert_mono.kernel_diagnostics}"

    # 4. Filter intersection
    spec_flt, tactics_flt = gen.generate_filter_intersection(idx=99)
    cert_flt, audit_flt = checker.verify_proof(spec_flt, "\n".join(tactics_flt))
    assert audit_flt.is_valid, f"Lean filter intersection failed: {cert_flt.kernel_diagnostics}"
