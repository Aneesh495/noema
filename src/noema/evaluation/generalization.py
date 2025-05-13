"""Out-of-distribution (OOD) generalization, length extrapolation, and transfer evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence
from noema.contracts.theorems import TheoremSpec
from noema.formal.checker import IndependentChecker


@dataclass
class HorizonEvaluationPoint:
    """Performance metrics at a specific proof horizon depth."""
    depth: int
    num_problems: int
    num_solved: int
    pass_at_1: float
    mean_iterations: float
    mean_wall_time_sec: float


@dataclass
class GeneralizationReport:
    """Comprehensive OOD generalization and proof length extrapolation report."""
    evaluation_name: str
    in_domain_pass_1: float
    out_of_domain_pass_1: float
    generalization_gap: float
    horizon_curve: list[HorizonEvaluationPoint] = field(default_factory=list)
    decay_rate_lambda: float = 0.0

    def to_markdown(self) -> str:
        """Render report as markdown summary."""
        lines = [
            f"### Generalization & Horizon Extrapolation: {self.evaluation_name}",
            f"- In-Domain pass@1: {self.in_domain_pass_1:.4f}",
            f"- Out-of-Domain pass@1: {self.out_of_domain_pass_1:.4f}",
            f"- Generalization Gap: {self.generalization_gap:.4f}",
            f"- Estimated Exponential Decay Rate λ: {self.decay_rate_lambda:.4f}",
            "",
            "| Horizon Depth | Problems | Solved | pass@1 | Mean Iterations | Time (s) |",
            "|---|---|---|---|---|---|",
        ]
        for p in self.horizon_curve:
            lines.append(
                f"| {p.depth} | {p.num_problems} | {p.num_solved} | {p.pass_at_1:.4f} | {p.mean_iterations:.1f} | {p.mean_wall_time_sec:.2f} |"
            )
        return "\n".join(lines)


class ProofHorizonExtrapolator:
    """Measures neural prover degradation as required proof depth extrapolates beyond training horizon."""

    def __init__(
        self,
        checker: IndependentChecker,
        eval_depths: Sequence[int] = (2, 4, 6, 8, 12),
    ):
        self.checker = checker
        self.eval_depths = list(eval_depths)

    def generate_depth_chain_theorem(self, depth: int, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Synthesize an implication transitivity chain of exact depth."""
        props = [f"P{i}" for i in range(depth + 1)]
        prop_decls = " ".join(props)
        hyps = [f"({props[i]} → {props[i+1]})" for i in range(depth)]
        stmt = f"∀ ({prop_decls} : Prop), " + " → ".join(hyps) + f" → ({props[0]} → {props[-1]})"
        
        thm_name = f"gen_horizon_chain_{depth}_{idx}"
        tactics = [f"intro {' '.join(props)}"]
        tactics.extend([f"intro h{i}" for i in range(depth)])
        tactics.append("intro hp0")
        
        cur = "hp0"
        for i in range(depth):
            cur = f"h{i} ({cur})"
        tactics.append(f"exact {cur}")

        thm = TheoremSpec(
            theorem_id=f"thm_horizon_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=stmt,
            family="horizon_extrapolation",
            parameters={"depth": str(depth)},
        )
        return thm, tactics

    def evaluate_depth_scaling(
        self,
        prover_fn: Callable[[TheoremSpec], tuple[bool, int, float]],
        problems_per_depth: int = 5,
    ) -> list[HorizonEvaluationPoint]:
        """Evaluate prover success rate and search iterations across proof depths."""
        points: list[HorizonEvaluationPoint] = []

        for d in self.eval_depths:
            solved_count = 0
            total_iters = 0
            total_time = 0.0

            for i in range(problems_per_depth):
                thm, _ = self.generate_depth_chain_theorem(depth=d, idx=i)
                solved, iters, wall_time = prover_fn(thm)
                if solved:
                    solved_count += 1
                total_iters += iters
                total_time += wall_time

            pass_1 = solved_count / max(1, problems_per_depth)
            mean_iter = total_iters / max(1, problems_per_depth)
            mean_time = total_time / max(1, problems_per_depth)

            points.append(
                HorizonEvaluationPoint(
                    depth=d,
                    num_problems=problems_per_depth,
                    num_solved=solved_count,
                    pass_at_1=pass_1,
                    mean_iterations=mean_iter,
                    mean_wall_time_sec=mean_time,
                )
            )

        return points

    @staticmethod
    def fit_exponential_decay(points: Sequence[HorizonEvaluationPoint]) -> float:
        """Estimate decay rate parameter lambda in P(d) = exp(-lambda * d)."""
        valid_points = [p for p in points if p.pass_at_1 > 0.0]
        if len(valid_points) < 2:
            return 0.0

        # Linear regression on log(pass_1) vs depth
        x_vals = [float(p.depth) for p in valid_points]
        y_vals = [math.log(max(1e-4, p.pass_at_1)) for p in valid_points]

        n = len(x_vals)
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n

        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        den = sum((x - x_mean) ** 2 for x in x_vals) + 1e-8

        slope = num / den
        # lambda is negative slope
        return max(0.0, -slope)


class CombinatorialTransferEvaluator:
    """Quantifies zero-shot transfer performance across unseen algebraic structures."""

    def __init__(self, in_domain_theorems: Sequence[TheoremSpec], ood_theorems: Sequence[TheoremSpec]):
        self.in_theorems = list(in_domain_theorems)
        self.ood_theorems = list(ood_theorems)

    def evaluate_transfer(
        self,
        prover_fn: Callable[[TheoremSpec], bool],
        name: str = "DomainTransfer",
    ) -> GeneralizationReport:
        """Execute in-domain and out-of-domain evaluation and calculate transfer gap."""
        in_solved = sum(1 for t in self.in_theorems if prover_fn(t))
        in_pass = in_solved / max(1, len(self.in_theorems))

        ood_solved = sum(1 for t in self.ood_theorems if prover_fn(t))
        ood_pass = ood_solved / max(1, len(self.ood_theorems))

        gap = max(0.0, in_pass - ood_pass)

        return GeneralizationReport(
            evaluation_name=name,
            in_domain_pass_1=in_pass,
            out_of_domain_pass_1=ood_pass,
            generalization_gap=gap,
        )
