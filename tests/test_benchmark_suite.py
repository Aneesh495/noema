"""Unit tests for formal benchmark harness and pass@k calculation."""

import pytest
from noema.benchmarks.benchmark_suite import (
    BenchmarkSuiteSummary,
    FormalBenchmarkHarness,
    ProblemEvaluationResult,
    compute_pass_at_k,
)
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec
from noema.formal.checker import IndependentChecker


def test_compute_pass_at_k_bounds():
    # 0 correct out of 10 -> pass@k = 0
    assert compute_pass_at_k(10, 0, 1) == 0.0
    assert compute_pass_at_k(10, 0, 5) == 0.0

    # 10 correct out of 10 -> pass@k = 1.0
    assert compute_pass_at_k(10, 10, 1) == 1.0
    assert compute_pass_at_k(10, 10, 5) == 1.0

    # 1 correct out of 10 -> pass@1 = 0.1, pass@10 = 1.0
    assert abs(compute_pass_at_k(10, 1, 1) - 0.1) < 1e-5
    assert abs(compute_pass_at_k(10, 1, 10) - 1.0) < 1e-5

    # Invalid cases
    assert compute_pass_at_k(10, 5, 0) == 0.0
    assert compute_pass_at_k(10, 5, 11) == 0.0


def test_benchmark_harness_evaluate_problem():
    checker = IndependentChecker(env_spec=EnvironmentSpec(toolchain="leanprover/lean4:v4.8.0"))
    harness = FormalBenchmarkHarness(checker=checker, k_values=(1, 2))

    thm = TheoremSpec(
        theorem_id="thm_bench_1",
        name="bench_id",
        module="Init.Core",
        statement="∀ (P : Prop), P → P",
        family="logic",
    )

    def mock_prover(t: TheoremSpec, n: int):
        # 1 correct proposal, 1 failing proposal
        return [
            (["intro P h", "exact h"], 0.9),
            (["intro P h", "exact P"], 0.1),
        ]

    res = harness.evaluate_problem(thm, mock_prover, num_samples=2, time_limit_per_problem=10.0)

    assert res.num_samples == 2
    assert res.num_correct == 1
    assert abs(res.pass_at_1 - 0.5) < 1e-4
    assert res.axioms_clean
    assert res.total_tactics_explored == 4


def test_benchmark_harness_evaluate_suite():
    checker = IndependentChecker(env_spec=EnvironmentSpec(toolchain="leanprover/lean4:v4.8.0"))
    harness = FormalBenchmarkHarness(checker=checker, k_values=(1,))

    thms = [
        TheoremSpec(
            theorem_id=f"thm_suite_{i}",
            name=f"suite_thm_{i}",
            module="Init.Core",
            statement="∀ (P : Prop), P → P",
            family="logic",
        )
        for i in range(2)
    ]

    def mock_prover(t: TheoremSpec, n: int):
        return [(["intro P h", "exact h"], 1.0)]

    summary = harness.evaluate_suite(
        suite_name="MiniF2F_Mock",
        theorems=thms,
        prover_fn=mock_prover,
        num_samples=1,
    )

    assert summary.total_problems == 2
    assert summary.solved_problems == 2
    assert summary.overall_pass_at_1 == 1.0
    assert summary.throughput_tactics_per_second > 0.0

    md = summary.to_markdown_table()
    assert "MiniF2F_Mock" in md
    assert "pass@1" in md
