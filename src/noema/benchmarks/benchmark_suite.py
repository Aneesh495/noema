"""Formal theorem proving benchmark harness: multi-dataset evaluation, pass@k, and latency profiling."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence
from noema.contracts.certificates import VerificationStatus
from noema.contracts.theorems import TheoremSpec
from noema.formal.checker import IndependentChecker


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute unbiased estimator for pass@k metric: 1 - comb(n - c, k) / comb(n, k)."""
    if n <= 0 or k <= 0 or k > n:
        return 0.0
    if n - c < k:
        return 1.0

    # Numerically stable calculation using log-gamma
    log_comb_n_minus_c_k = (
        math.lgamma(n - c + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - c - k + 1)
    )
    log_comb_n_k = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )
    ratio = math.exp(log_comb_n_minus_c_k - log_comb_n_k)
    return 1.0 - ratio


@dataclass
class ProblemEvaluationResult:
    """Outcome of attempting to prove a single benchmark problem."""
    theorem_id: str
    num_samples: int
    num_correct: int
    pass_at_1: float
    pass_at_k_estimates: dict[int, float]
    wall_time_seconds: float
    total_tactics_explored: int
    axioms_clean: bool = True
    error_summary: str = ""


@dataclass
class BenchmarkSuiteSummary:
    """Aggregated evaluation metrics across an entire theorem proving benchmark."""
    suite_name: str
    total_problems: int
    solved_problems: int
    overall_pass_at_1: float
    overall_pass_at_k: dict[int, float]
    mean_wall_time_per_problem: float
    total_wall_time_seconds: float
    throughput_tactics_per_second: float
    results: list[ProblemEvaluationResult] = field(default_factory=list)

    def to_markdown_table(self) -> str:
        """Render markdown summary table."""
        lines = [
            f"### Benchmark Suite: {self.suite_name}",
            f"- Total Problems: {self.total_problems}",
            f"- Solved Problems: {self.solved_problems} ({100.0 * self.solved_problems / max(1, self.total_problems):.1f}%)",
            f"- Overall pass@1: {self.overall_pass_at_1:.4f}",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        for k, v in sorted(self.overall_pass_at_k.items()):
            lines.append(f"| pass@{k} | {v:.4f} |")
        lines.append(f"| Mean Time / Problem | {self.mean_wall_time_per_problem:.2f} s |")
        lines.append(f"| Tactic Throughput | {self.throughput_tactics_per_second:.1f} steps/s |")
        return "\n".join(lines)


class FormalBenchmarkHarness:
    """Multi-dataset evaluator executing neural theorem provers against formal problems."""

    def __init__(
        self,
        checker: IndependentChecker,
        k_values: Sequence[int] = (1, 5, 10),
    ):
        self.checker = checker
        self.k_values = list(k_values)

    def evaluate_problem(
        self,
        theorem: TheoremSpec,
        prover_fn: Callable[[TheoremSpec, int], list[tuple[list[str], float]]],
        num_samples: int = 10,
        time_limit_per_problem: float = 30.0,
    ) -> ProblemEvaluationResult:
        """Evaluate candidate proof generation on a single benchmark theorem.
        
        Args:
            theorem: Target formal theorem specification.
            prover_fn: Takes (theorem, num_samples) and returns list of (tactic_lines, confidence).
            num_samples: Number of sample proof trajectories generated.
            time_limit_per_problem: Wall time budget in seconds.
        """
        start_time = time.perf_counter()
        proposals = prover_fn(theorem, num_samples)
        num_correct = 0
        total_tactics = 0
        axioms_clean = True
        error_msg = ""

        for tactic_lines, _ in proposals:
            total_tactics += len(tactic_lines)
            script = "\n".join(tactic_lines)
            cert, audit = self.checker.verify_proof(theorem, script)
            if audit.is_valid:
                num_correct += 1
                if not audit.axioms_permitted:
                    axioms_clean = False
            else:
                if not error_msg:
                    error_msg = cert.kernel_diagnostics

            if time.perf_counter() - start_time > time_limit_per_problem:
                break

        elapsed = time.perf_counter() - start_time
        pass_at_k_map: dict[int, float] = {}
        for k in self.k_values:
            if k <= num_samples:
                pass_at_k_map[k] = compute_pass_at_k(num_samples, num_correct, k)

        pass_1 = pass_at_k_map.get(1, float(num_correct > 0))

        return ProblemEvaluationResult(
            theorem_id=theorem.theorem_id,
            num_samples=num_samples,
            num_correct=num_correct,
            pass_at_1=pass_1,
            pass_at_k_estimates=pass_at_k_map,
            wall_time_seconds=elapsed,
            total_tactics_explored=total_tactics,
            axioms_clean=axioms_clean,
            error_summary=error_msg,
        )

    def evaluate_suite(
        self,
        suite_name: str,
        theorems: Sequence[TheoremSpec],
        prover_fn: Callable[[TheoremSpec, int], list[tuple[list[str], float]]],
        num_samples: int = 10,
        time_limit_per_problem: float = 30.0,
    ) -> BenchmarkSuiteSummary:
        """Run complete benchmark suite across a sequence of theorems."""
        start_suite = time.perf_counter()
        results: list[ProblemEvaluationResult] = []
        solved_count = 0
        total_tactics_all = 0

        for thm in theorems:
            res = self.evaluate_problem(
                thm,
                prover_fn,
                num_samples=num_samples,
                time_limit_per_problem=time_limit_per_problem,
            )
            results.append(res)
            if res.num_correct > 0:
                solved_count += 1
            total_tactics_all += res.total_tactics_explored

        total_time = time.perf_counter() - start_suite
        num_problems = len(theorems)

        # Compute macro-averaged pass@k
        avg_pass_at_k: dict[int, float] = {}
        for k in self.k_values:
            vals = [r.pass_at_k_estimates.get(k, 0.0) for r in results if k in r.pass_at_k_estimates]
            avg_pass_at_k[k] = sum(vals) / max(1, len(vals))

        overall_pass_1 = avg_pass_at_k.get(1, 0.0)
        mean_time = total_time / max(1, num_problems)
        tps = total_tactics_all / max(1e-6, total_time)

        return BenchmarkSuiteSummary(
            suite_name=suite_name,
            total_problems=num_problems,
            solved_problems=solved_count,
            overall_pass_at_1=overall_pass_1,
            overall_pass_at_k=avg_pass_at_k,
            mean_wall_time_per_problem=mean_time,
            total_wall_time_seconds=total_time,
            throughput_tactics_per_second=tps,
            results=results,
        )
