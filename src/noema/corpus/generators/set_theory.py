"""Set theory and propositional logic domain theorem generator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence
from noema.contracts.theorems import TheoremSpec


@dataclass(frozen=True)
class LogicTheoremTemplate:
    """Template describing a propositional logic or set-theoretic theorem."""
    template_id: str
    logic_category: str
    pattern_statement: str
    tactic_skeleton: list[str]
    required_premises: list[str]


class SetTheoryTheoremGenerator:
    """Generates synthetic theorems in propositional logic, boolean algebra, and set theory."""

    PREMISES_LOGIC = [
        "And.intro",
        "And.left",
        "And.right",
        "Or.inl",
        "Or.inr",
        "Or.elim",
        "Iff.intro",
        "Iff.mp",
        "Iff.mpr",
        "Not.intro",
        "False.elim",
    ]

    PREMISES_SET = [
        "Set.mem_inter_iff",
        "Set.mem_union",
        "Set.mem_compl_iff",
        "Set.subset_def",
        "Set.ext",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_propositional_de_morgan(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate first intuitionistic de Morgan direction: not (A or B) iff (not A and not B)."""
        thm_name = f"logic_de_morgan_and_or_{idx}"
        statement = "∀ (A B : Prop), ¬(A ∨ B) ↔ (¬A ∧ ¬B)"
        tactics = [
            "intro A B",
            "constructor",
            "· intro h; exact ⟨fun ha => h (Or.inl ha), fun hb => h (Or.inr hb)⟩",
            "· intro h hab; cases hab with | inl ha => exact h.1 ha | inr hb => exact h.2 hb",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_logic_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="logic_demorgan",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_LOGIC,
        )
        return thm, tactics

    def generate_absorption_identity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate absorption law: A and (A or B) iff A."""
        thm_name = f"logic_absorption_and_or_{idx}"
        statement = "∀ (A B : Prop), (A ∧ (A ∨ B)) ↔ A"
        tactics = [
            "intro A B",
            "constructor",
            "· intro h; exact h.1",
            "· intro ha; exact ⟨ha, Or.inl ha⟩",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_logic_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="logic_absorption",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_LOGIC,
        )
        return thm, tactics

    def generate_distributivity_law(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate distributivity of conjunction over disjunction."""
        thm_name = f"logic_distrib_and_or_{idx}"
        statement = "∀ (A B C : Prop), (A ∧ (B ∨ C)) ↔ ((A ∧ B) ∨ (A ∧ C))"
        tactics = [
            "intro A B C",
            "constructor",
            "· intro h; cases h with | intro ha hbc => cases hbc with | inl hb => exact Or.inl ⟨ha, hb⟩ | inr hc => exact Or.inr ⟨ha, hc⟩",
            "· intro h; cases h with | inl hab => exact ⟨hab.1, Or.inl hab.2⟩ | inr hac => exact ⟨hac.1, Or.inr hac.2⟩",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_logic_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="logic_distributivity",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_LOGIC,
        )
        return thm, tactics

    def generate_implication_transitivity_chain(
        self, chain_len: int = 3, idx: int = 0
    ) -> tuple[TheoremSpec, list[str]]:
        """Generate a chain of implications: (P0 -> P1) -> ... -> (P0 -> Pn)."""
        props = [f"P{i}" for i in range(chain_len + 1)]
        prop_decls = " ".join(props)
        hyps = [f"({props[i]} → {props[i+1]})" for i in range(chain_len)]
        stmt = f"∀ ({prop_decls} : Prop), " + " → ".join(hyps) + f" → ({props[0]} → {props[-1]})"
        
        thm_name = f"logic_imp_trans_chain_{chain_len}_{idx}"
        tactics = [f"intro {' '.join(props)}"]
        tactics.extend([f"intro h{i}" for i in range(chain_len)])
        tactics.append("intro hp0")
        
        cur = "hp0"
        for i in range(chain_len):
            cur = f"h{i} ({cur})"
        tactics.append(f"exact {cur}")

        thm = TheoremSpec(
            theorem_id=f"thm_logic_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=stmt,
            family="logic_implication",
            parameters={"chain_len": str(chain_len)},
            eligible_premise_names=self.PREMISES_LOGIC,
        )
        return thm, tactics

    def generate_contraposition(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate intuitionistic contraposition: (A -> B) -> (not B -> not A)."""
        thm_name = f"logic_contraposition_{idx}"
        statement = "∀ (A B : Prop), (A → B) → (¬B → ¬A)"
        tactics = [
            "intro A B hab hnb ha",
            "exact hnb (hab ha)",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_logic_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="logic_contraposition",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_LOGIC,
        )
        return thm, tactics

    def generate_disjunctive_syllogism(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate disjunctive syllogism: (A or B) -> not A -> B."""
        thm_name = f"logic_disjunctive_syllogism_{idx}"
        statement = "∀ (A B : Prop), (A ∨ B) → ¬A → B"
        tactics = [
            "intro A B hab hna",
            "cases hab with | inl ha => exact False.elim (hna ha) | inr hb => exact hb",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_logic_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="logic_syllogism",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_LOGIC,
        )
        return thm, tactics

    def generate_intersection_comm(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate intersection commutativity modeled over predicate conjunction."""
        thm_name = f"set_inter_comm_{idx}"
        statement = "∀ (α : Type) (s t : α → Prop) (x : α), (s x ∧ t x) ↔ (t x ∧ s x)"
        tactics = [
            "intro α s t x",
            "constructor",
            "· intro h; exact ⟨h.2, h.1⟩",
            "· intro h; exact ⟨h.2, h.1⟩",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_set_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="set_intersection",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_SET,
        )
        return thm, tactics

    def generate_batch(self, total: int = 20) -> list[tuple[TheoremSpec, list[str]]]:
        """Generate a diverse batch of propositional logic and set theory theorems."""
        results: list[tuple[TheoremSpec, list[str]]] = []
        generators = [
            lambda i: self.generate_propositional_de_morgan(i),
            lambda i: self.generate_absorption_identity(i),
            lambda i: self.generate_distributivity_law(i),
            lambda i: self.generate_contraposition(i),
            lambda i: self.generate_disjunctive_syllogism(i),
            lambda i: self.generate_intersection_comm(i),
            lambda i: self.generate_implication_transitivity_chain(chain_len=2, idx=i),
            lambda i: self.generate_implication_transitivity_chain(chain_len=3, idx=i),
        ]
        
        for idx in range(total):
            gen = generators[idx % len(generators)]
            results.append(gen(idx))
            
        return results
