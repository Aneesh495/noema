"""Category theory, functoriality, and monad laws domain theorem generator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence
from noema.contracts.theorems import TheoremSpec


@dataclass(frozen=True)
class CategoryTheoremTemplate:
    """Template describing a category theory or monad axiom theorem."""
    template_id: str
    categorical_concept: str
    pattern_statement: str
    tactic_skeleton: list[str]
    required_premises: list[str]


class CategoryTheoremGenerator:
    """Generates synthetic theorems in category theory, arrow composition, and monad laws."""

    PREMISES_CATEGORY = [
        "Function.comp_assoc",
        "Function.comp_id",
        "Function.id_comp",
        "Option.bind_some",
        "Option.bind_assoc",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_comp_associativity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate associativity of morphism composition: (h ∘ g) ∘ f = h ∘ (g ∘ f)."""
        thm_name = f"cat_comp_assoc_{idx}"
        statement = (
            "∀ (α β γ δ : Type) (f : α → β) (g : β → γ) (h : γ → δ), "
            "(h ∘ g) ∘ f = h ∘ (g ∘ f)"
        )
        tactics = [
            "intro α β γ δ f g h",
            "rfl",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_cat_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="category_morphism",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_CATEGORY,
        )
        return thm, tactics

    def generate_comp_identity_left(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate left unit identity: id ∘ f = f."""
        thm_name = f"cat_comp_id_left_{idx}"
        statement = "∀ (α β : Type) (f : α → β), id ∘ f = f"
        tactics = [
            "intro α β f",
            "rfl",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_cat_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="category_morphism",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_CATEGORY,
        )
        return thm, tactics

    def generate_comp_identity_right(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate right unit identity: f ∘ id = f."""
        thm_name = f"cat_comp_id_right_{idx}"
        statement = "∀ (α β : Type) (f : α → β), f ∘ id = f"
        tactics = [
            "intro α β f",
            "rfl",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_cat_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="category_morphism",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_CATEGORY,
        )
        return thm, tactics

    def generate_option_monad_left_identity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate monad left unit law: (some x >>= f) = f x."""
        thm_name = f"cat_monad_left_id_{idx}"
        statement = (
            "∀ (α β : Type) (x : α) (f : α → Option β), "
            "(Option.some x >>= f) = f x"
        )
        tactics = [
            "intro α β x f",
            "rfl",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_cat_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="category_monad",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_CATEGORY,
        )
        return thm, tactics

    def generate_option_monad_right_identity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate monad right unit law: (m >>= some) = m."""
        thm_name = f"cat_monad_right_id_{idx}"
        statement = "∀ (α : Type) (m : Option α), (m >>= Option.some) = m"
        tactics = [
            "intro α m",
            "cases m <;> rfl",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_cat_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="category_monad",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_CATEGORY,
        )
        return thm, tactics

    def generate_option_monad_associativity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate monad associativity: ((m >>= f) >>= g) = (m >>= (fun x => f x >>= g))."""
        thm_name = f"cat_monad_assoc_{idx}"
        statement = (
            "∀ (α β γ : Type) (m : Option α) (f : α → Option β) (g : β → Option γ), "
            "((m >>= f) >>= g) = (m >>= (fun x => f x >>= g))"
        )
        tactics = [
            "intro α β γ m f g",
            "cases m <;> rfl",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_cat_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="category_monad",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_CATEGORY,
        )
        return thm, tactics

    def generate_naturality_square(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate commuting square for identity natural transformation."""
        thm_name = f"cat_naturality_id_{idx}"
        statement = (
            "∀ (α β : Type) (f : α → β) (x : α), "
            "(id (f x)) = f (id x)"
        )
        tactics = [
            "intro α β f x",
            "rfl",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_cat_{thm_name}",
            name=thm_name,
            module="Init.Core",
            statement=statement,
            family="category_functor",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_CATEGORY,
        )
        return thm, tactics

    def generate_batch(self, total: int = 20) -> list[tuple[TheoremSpec, list[str]]]:
        """Generate a diverse batch of category theory and monad law theorems."""
        results: list[tuple[TheoremSpec, list[str]]] = []
        generators = [
            lambda i: self.generate_comp_associativity(i),
            lambda i: self.generate_comp_identity_left(i),
            lambda i: self.generate_comp_identity_right(i),
            lambda i: self.generate_option_monad_left_identity(i),
            lambda i: self.generate_option_monad_right_identity(i),
            lambda i: self.generate_option_monad_associativity(i),
            lambda i: self.generate_naturality_square(i),
        ]
        for idx in range(total):
            gen = generators[idx % len(generators)]
            results.append(gen(idx))
        return results
