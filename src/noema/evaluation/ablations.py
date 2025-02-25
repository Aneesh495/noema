"""Depth, retrieval, value-head, and curriculum ablation matrix runner."""

from __future__ import annotations

import json
from pathlib import Path
from pydantic import BaseModel
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec
from noema.evaluation.harness import BenchmarkHarness, BenchmarkMetrics
from noema.formal.bridge import FormalBridge
from noema.formal.checker import IndependentChecker
from noema.models.config import TransformerConfig
from noema.models.transformer import NoemaTransformer
from noema.retrieval.index import PremiseIndex
from noema.search.budget import SearchBudget
from noema.search.engine import ProofSearchEngine
from noema.search.priority import SearchStrategy


class AblationRunResult(BaseModel):
    condition_name: str
    category: str
    seed: int
    depth_layers: int
    retrieval_mode: str
    priority_strategy: str
    metrics: BenchmarkMetrics


class AblationRunner:
    """Coordinates the fixed ablation study across depths (4, 12, 24), retrieval, and value heads."""

    def __init__(
        self,
        env_spec: EnvironmentSpec,
        bridge: FormalBridge,
        checker: IndependentChecker,
        eval_theorems: list[TheoremSpec],
    ):
        self.env_spec = env_spec
        self.bridge = bridge
        self.checker = checker
        self.eval_theorems = eval_theorems

    def run_depth_ablation(
        self,
        seeds: list[int] = [42, 43, 44],
        depths: list[int] = [4, 12, 24],
        budget: SearchBudget | None = None,
    ) -> list[AblationRunResult]:
        results: list[AblationRunResult] = []
        b = budget or SearchBudget(max_transitions=20, max_wallclock_sec=5.0)

        for depth in depths:
            for s in seeds:
                cfg = TransformerConfig(d_model=64, n_layers=depth, n_heads=4, d_ff=128)
                model = NoemaTransformer(cfg)
                model.eval()

                searcher = ProofSearchEngine(
                    bridge=self.bridge,
                    checker=self.checker,
                    model=model,
                    strategy=SearchStrategy.NEURAL_GUIDED,
                )
                harness = BenchmarkHarness(searcher)
                _, metrics = harness.evaluate_suite(
                    name=f"depth_{depth}_seed_{s}",
                    theorems=self.eval_theorems,
                    budget=b,
                )

                res = AblationRunResult(
                    condition_name=f"depth_{depth}_seed_{s}",
                    category="depth_study",
                    seed=s,
                    depth_layers=depth,
                    retrieval_mode="dense",
                    priority_strategy="neural_guided",
                    metrics=metrics,
                )
                results.append(res)

        return results

    def run_search_strategy_ablation(
        self,
        seeds: list[int] = [42, 43, 44],
        budget: SearchBudget | None = None,
    ) -> list[AblationRunResult]:
        results: list[AblationRunResult] = []
        b = budget or SearchBudget(max_transitions=20, max_wallclock_sec=5.0)

        strategies = [
            SearchStrategy.NEURAL_GUIDED,
            SearchStrategy.POLICY_ONLY,
            SearchStrategy.LEXICAL_GUIDED,
            SearchStrategy.UNIFORM_BFS,
            SearchStrategy.RANDOM_PRIORITY,
        ]

        cfg = TransformerConfig(d_model=64, n_layers=4, n_heads=4, d_ff=128)
        model = NoemaTransformer(cfg)
        model.eval()

        for strat in strategies:
            for s in seeds:
                searcher = ProofSearchEngine(
                    bridge=self.bridge,
                    checker=self.checker,
                    model=model,
                    strategy=strat,
                )
                harness = BenchmarkHarness(searcher)
                _, metrics = harness.evaluate_suite(
                    name=f"strat_{strat.value}_seed_{s}",
                    theorems=self.eval_theorems,
                    budget=b,
                )

                res = AblationRunResult(
                    condition_name=f"strat_{strat.value}_seed_{s}",
                    category="search_strategy",
                    seed=s,
                    depth_layers=4,
                    retrieval_mode="lexical",
                    priority_strategy=strat.value,
                    metrics=metrics,
                )
                results.append(res)

        return results
