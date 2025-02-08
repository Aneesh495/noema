"""Dependency closure calculation and declaration prefix auditing."""

from __future__ import annotations

import networkx as nx
from noema.contracts.environment import EnvironmentSpec, PremiseDecl
from noema.contracts.theorems import TheoremSpec


class DependencyGraph:
    """Directed acyclic graph of mathematical premise dependencies."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_declaration(self, name: str, dependencies: list[str], module: str = "") -> None:
        self.graph.add_node(name, module=module)
        for dep in dependencies:
            self.graph.add_edge(name, dep)

    def get_transitive_dependencies(self, name: str) -> set[str]:
        """Compute the set of all premises required to prove or define name."""
        if name not in self.graph:
            return set()
        return set(nx.descendants(self.graph, name))

    def get_downstream_dependents(self, name: str) -> set[str]:
        """Compute all declarations that depend on name."""
        if name not in self.graph:
            return set()
        return set(nx.ancestors(self.graph, name))

    def verify_no_downstream_leakage(
        self,
        target_name: str,
        candidate_premises: list[str],
    ) -> tuple[bool, list[str]]:
        """Verify that no candidate premise depends on the target theorem."""
        downstream = self.get_downstream_dependents(target_name)
        downstream.add(target_name)
        
        leaked = [p for p in candidate_premises if p in downstream]
        return (len(leaked) == 0, leaked)

    def build_eligible_environment(
        self,
        target_name: str,
        all_premises: dict[str, PremiseDecl],
        toolchain: str = "leanprover/lean4:v4.8.0",
    ) -> EnvironmentSpec:
        """Create an EnvironmentSpec excluding target_name and all downstream dependents."""
        downstream = self.get_downstream_dependents(target_name)
        downstream.add(target_name)

        eligible = {
            k: v for k, v in all_premises.items()
            if k not in downstream
        }

        return EnvironmentSpec(
            toolchain=toolchain,
            eligible_premises=eligible,
            declaration_prefix=target_name,
        )
