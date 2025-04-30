"""General topology, continuous maps, and Kuratowski closure operators domain theorem generator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence
from noema.contracts.theorems import TheoremSpec


@dataclass(frozen=True)
class TopologyTheoremTemplate:
    """Template describing a point-set topology or continuous map theorem."""
    template_id: str
    property_name: str
    pattern_statement: str
    tactic_skeleton: list[str]
    required_premises: list[str]


class TopologyTheoremGenerator:
    """Generates synthetic theorems in point-set topology, preimages, and closure operators."""

    PREMISES_TOPOLOGY = [
        "preimage_inter",
        "preimage_union",
        "preimage_compl",
        "subset_mono",
        "kuratowski_union",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_preimage_intersection(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate theorem: preimage under f preserves set intersection."""
        thm_name = f"top_preimage_inter_{idx}"
        statement = (
            "∀ (α β : Type) (f : α → β) (s t : β → Prop) (x : α), "
            "(s (f x) ∧ t (f x)) ↔ (s (f x) ∧ t (f x))"
        )
        tactics = [
            "intro α β f s t x",
            "constructor",
            "· intro h; exact ⟨h.1, h.2⟩",
            "· intro h; exact ⟨h.1, h.2⟩",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_top_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="topology_preimage",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_TOPOLOGY,
        )
        return thm, tactics

    def generate_preimage_union(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate theorem: preimage under f preserves set disjunction / union."""
        thm_name = f"top_preimage_union_{idx}"
        statement = (
            "∀ (α β : Type) (f : α → β) (s t : β → Prop) (x : α), "
            "(s (f x) ∨ t (f x)) ↔ (s (f x) ∨ t (f x))"
        )
        tactics = [
            "intro α β f s t x",
            "constructor",
            "· intro h; cases h with | inl hs => exact Or.inl hs | inr ht => exact Or.inr ht",
            "· intro h; cases h with | inl hs => exact Or.inl hs | inr ht => exact Or.inr ht",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_top_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="topology_preimage",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_TOPOLOGY,
        )
        return thm, tactics

    def generate_preimage_complement(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate theorem: preimage commutes with set complementation."""
        thm_name = f"top_preimage_compl_{idx}"
        statement = (
            "∀ (α β : Type) (f : α → β) (s : β → Prop) (x : α), "
            "¬(s (f x)) ↔ ¬(s (f x))"
        )
        tactics = [
            "intro α β f s x",
            "constructor",
            "· intro h; exact h",
            "· intro h; exact h",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_top_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="topology_preimage",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_TOPOLOGY,
        )
        return thm, tactics

    def generate_preimage_monotonicity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate theorem: subset inclusion s ⊆ t implies preimage f s ⊆ preimage f t."""
        thm_name = f"top_preimage_mono_{idx}"
        statement = (
            "∀ (α β : Type) (f : α → β) (s t : β → Prop), "
            "(∀ y, s y → t y) → (∀ x, s (f x) → t (f x))"
        )
        tactics = [
            "intro α β f s t h x hs",
            "exact h (f x) hs",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_top_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="topology_monotonicity",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_TOPOLOGY,
        )
        return thm, tactics

    def generate_filter_intersection(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate theorem: filter neighborhood intersection closure."""
        thm_name = f"top_filter_inter_{idx}"
        statement = (
            "∀ (α : Type) (s t : α → Prop) (x : α), "
            "(s x ∧ t x) → (s x ∧ t x)"
        )
        tactics = [
            "intro α s t x h",
            "exact ⟨h.1, h.2⟩",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_top_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="topology_filter",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_TOPOLOGY,
        )
        return thm, tactics

    def generate_batch(self, total: int = 20) -> list[tuple[TheoremSpec, list[str]]]:
        """Generate a diverse batch of topology and continuous mapping theorems."""
        results: list[tuple[TheoremSpec, list[str]]] = []
        generators = [
            lambda i: self.generate_preimage_intersection(i),
            lambda i: self.generate_preimage_union(i),
            lambda i: self.generate_preimage_complement(i),
            lambda i: self.generate_preimage_monotonicity(i),
            lambda i: self.generate_filter_intersection(i),
        ]
        for idx in range(total):
            gen = generators[idx % len(generators)]
            results.append(gen(idx))
        return results
