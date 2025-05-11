"""Knowledge graph premise retriever with Personalized PageRank and dependency topological filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass
class PremiseNode:
    """A mathematical entity or theorem in the dependency knowledge graph."""
    name: str
    module: str
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)


class PremiseDependencyGraph:
    """Directed graph capturing transitive dependency relationships between formal mathematical lemmas."""

    def __init__(self):
        self.nodes: dict[str, PremiseNode] = {}

    def add_premise(self, name: str, module: str = "", dependencies: Sequence[str] | None = None) -> None:
        """Register a lemma and its declared upstream dependency axioms/lemmas."""
        deps = list(dependencies or [])
        if name not in self.nodes:
            self.nodes[name] = PremiseNode(name=name, module=module, dependencies=deps)
        else:
            self.nodes[name].dependencies = deps

        for d in deps:
            if d not in self.nodes:
                self.nodes[d] = PremiseNode(name=d, module=module, dependents=[name])
            else:
                if name not in self.nodes[d].dependents:
                    self.nodes[d].dependents.append(name)

    def transitive_dependencies(self, root_name: str) -> set[str]:
        """Compute the complete upstream transitive dependency closure of a theorem."""
        visited: set[str] = set()
        queue = [root_name]

        while queue:
            curr = queue.pop(0)
            if curr not in self.nodes:
                continue
            for d in self.nodes[curr].dependencies:
                if d not in visited:
                    visited.add(d)
                    queue.append(d)

        return visited

    def has_circular_dependency(self) -> bool:
        """Check for cycles using topological Kahn's algorithm."""
        in_degree = {name: len(node.dependencies) for name, node in self.nodes.items()}
        zero_in = [name for name, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while zero_in:
            curr = zero_in.pop(0)
            visited_count += 1
            if curr in self.nodes:
                for dep in self.nodes[curr].dependents:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        zero_in.append(dep)

        return visited_count < len(self.nodes)


class PersonalizedPageRankRetriever:
    """Ranks mathematical premises using random walks with restart on the premise dependency graph."""

    def __init__(
        self,
        graph: PremiseDependencyGraph,
        damping: float = 0.85,
        max_iterations: int = 40,
        tol: float = 1e-6,
    ):
        self.graph = graph
        self.alpha = damping
        self.max_iter = max_iterations
        self.tol = tol

    def compute_ppr(self, seed_weights: Mapping[str, float]) -> dict[str, float]:
        """Compute stationary distribution vector under personalized teleport distribution."""
        all_nodes = sorted(self.graph.nodes.keys())
        n = len(all_nodes)
        if n == 0:
            return {}

        # Construct normalized teleport vector
        total_seed = sum(seed_weights.get(name, 0.0) for name in all_nodes)
        if total_seed <= 0.0:
            teleport = {name: 1.0 / n for name in all_nodes}
        else:
            teleport = {name: seed_weights.get(name, 0.0) / total_seed for name in all_nodes}

        # Initialize uniform probability distribution
        p = {name: 1.0 / n for name in all_nodes}

        for _ in range(self.max_iter):
            next_p = {name: (1.0 - self.alpha) * teleport[name] for name in all_nodes}

            for u_name in all_nodes:
                u_node = self.graph.nodes[u_name]
                out_edges = u_node.dependencies
                if not out_edges:
                    # Dangling node teleports uniformly
                    for v_name in all_nodes:
                        next_p[v_name] += self.alpha * p[u_name] / n
                else:
                    weight_per_edge = self.alpha * p[u_name] / len(out_edges)
                    for v_name in out_edges:
                        if v_name in next_p:
                            next_p[v_name] += weight_per_edge

            # Check convergence L1 norm
            diff = sum(abs(next_p[name] - p[name]) for name in all_nodes)
            p = next_p
            if diff < self.tol:
                break

        return p

    def rank_premises(
        self,
        dense_scores: Mapping[str, float],
        target_conjecture: str | None = None,
        top_k: int = 10,
        graph_weight: float = 0.4,
    ) -> list[tuple[str, float]]:
        """Perform hybrid ranking combining dense semantic similarity and structural PPR centrality."""
        ppr_scores = self.compute_ppr(dense_scores)

        # Disallow circular premises that depend on the target conjecture
        forbidden: set[str] = set()
        if target_conjecture and target_conjecture in self.graph.nodes:
            # Avoid using anything that depends on target
            forbidden = set(self.graph.nodes[target_conjecture].dependents)
            forbidden.add(target_conjecture)

        max_dense = max(dense_scores.values()) if dense_scores else 1.0
        max_dense = max(max_dense, 1e-8)
        max_ppr = max(ppr_scores.values()) if ppr_scores else 1.0
        max_ppr = max(max_ppr, 1e-8)

        hybrid_scores: dict[str, float] = {}
        for name in self.graph.nodes:
            if name in forbidden:
                continue
            norm_dense = dense_scores.get(name, 0.0) / max_dense
            norm_ppr = ppr_scores.get(name, 0.0) / max_ppr
            hybrid_scores[name] = (1.0 - graph_weight) * norm_dense + graph_weight * norm_ppr

        sorted_ranked = sorted(hybrid_scores.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_ranked[:top_k]
