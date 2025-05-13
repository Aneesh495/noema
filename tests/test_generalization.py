"""Unit tests for out-of-distribution generalization and length extrapolation."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec
from noema.evaluation.generalization import (
    CombinatorialTransferEvaluator,
    HorizonEvaluationPoint,
    ProofHorizonExtrapolator,
)
from noema.formal.checker import IndependentChecker


@pytest.fixture
def checker() -> IndependentChecker:
    return IndependentChecker(env_spec=EnvironmentSpec(toolchain="leanprover/lean4:v4.8.0"))


def test_proof_horizon_theorem_generation(checker):
    extrapolator = ProofHorizonExtrapolator(checker=checker, eval_depths=(2, 4))
    thm, tactics = extrapolator.generate_depth_chain_theorem(depth=3, idx=1)

    assert "P0 → P3" in thm.statement or "(P0 → P3)" in thm.statement
    assert len(tactics) >= 4

    # Verify directly with Lean
    cert, audit = checker.verify_proof(thm, "\n".join(tactics))
    assert audit.is_valid, f"Proof horizon depth 3 failed: {cert.kernel_diagnostics}"


def test_evaluate_depth_scaling(checker):
    extrapolator = ProofHorizonExtrapolator(checker=checker, eval_depths=(2, 3))

    def mock_prover(t: TheoremSpec) -> tuple[bool, int, float]:
        # Solves depth 2 easily, struggles at depth 3
        d = int(t.parameters.get("depth", "2"))
        if d <= 2:
            return True, 5, 0.05
        return False, 20, 0.15

    points = extrapolator.evaluate_depth_scaling(mock_prover, problems_per_depth=2)
    assert len(points) == 2
    assert points[0].pass_at_1 == 1.0
    assert points[1].pass_at_1 == 0.0


def test_fit_exponential_decay():
    points = [
        HorizonEvaluationPoint(depth=2, num_problems=10, num_solved=10, pass_at_1=1.0, mean_iterations=5.0, mean_wall_time_sec=0.1),
        HorizonEvaluationPoint(depth=4, num_problems=10, num_solved=5, pass_at_1=0.5, mean_iterations=10.0, mean_wall_time_sec=0.2),
        HorizonEvaluationPoint(depth=6, num_problems=10, num_solved=2, pass_at_1=0.25, mean_iterations=20.0, mean_wall_time_sec=0.4),
    ]
    rate = ProofHorizonExtrapolator.fit_exponential_decay(points)
    assert rate > 0.0


def test_combinatorial_transfer_evaluator():
    in_thms = [
        TheoremSpec(theorem_id=f"in_{i}", name=f"in_{i}", module="Init.Core", statement="A → A", family="monoid")
        for i in range(4)
    ]
    ood_thms = [
        TheoremSpec(theorem_id=f"ood_{i}", name=f"ood_{i}", module="Init.Core", statement="A ∧ B ↔ B ∧ A", family="ring")
        for i in range(4)
    ]

    evaluator = CombinatorialTransferEvaluator(in_thms, ood_thms)

    def mock_prover(t: TheoremSpec) -> bool:
        return t.family == "monoid"

    report = evaluator.evaluate_transfer(mock_prover, name="MonoidToRing")
    assert report.in_domain_pass_1 == 1.0
    assert report.out_of_domain_pass_1 == 0.0
    assert report.generalization_gap == 1.0

    md = report.to_markdown()
    assert "MonoidToRing" in md
    assert "Generalization Gap: 1.0000" in md
