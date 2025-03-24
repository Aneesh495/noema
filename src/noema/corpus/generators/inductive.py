"""Inductive datatype and structural induction theorem generator."""

from __future__ import annotations

import random
from noema.contracts.theorems import TheoremSpec


class InductiveTheoremGenerator:
    """Generates inductive propositions over numbers, lists, and trees with induction proofs."""

    PREMISES_INDUCTION = [
        "Nat.zero_add",
        "Nat.add_zero",
        "Nat.succ_add",
        "Nat.add_succ",
        "List.append_nil",
        "List.nil_append",
        "List.append_assoc",
        "List.length_append",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_peano_addition_zero(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate left addition identity requiring induction on Peano Nat."""
        thm_name = f"peano_zero_add_inductive_{idx}"
        statement = "∀ (n : Nat), 0 + n = n"
        
        # Genuine multi-step induction proof
        proof = [
            "intro n",
            "induction n with",
            "| zero => rfl",
            "| succ d hd => rw [Nat.add_succ, hd]",
        ]

        thm = TheoremSpec(
            theorem_id=f"thm_ind_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="inductive_peano",
            parameters={},
            eligible_premise_names=self.PREMISES_INDUCTION,
        )
        return thm, proof

    def generate_peano_succ_add(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate successor distribution over addition."""
        thm_name = f"peano_succ_add_inductive_{idx}"
        statement = "∀ (n m : Nat), Nat.succ n + m = Nat.succ (n + m)"
        proof = [
            "intro n m",
            "induction m with",
            "| zero => rfl",
            "| succ d hd => rw [Nat.add_succ, Nat.add_succ, hd]",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_ind_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="inductive_peano",
            parameters={},
            eligible_premise_names=self.PREMISES_INDUCTION,
        )
        return thm, proof

    def generate_list_append_nil(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate list right identity under append requiring structural induction."""
        thm_name = f"list_append_nil_inductive_{idx}"
        statement = "∀ (α : Type) (xs : List α), xs ++ [] = xs"
        proof = [
            "intro α xs",
            "induction xs with",
            "| nil => rfl",
            "| cons h t ih =>",
            "  simp only [List.cons_append]",
            "  rw [ih]",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_ind_{thm_name}",
            name=thm_name,
            module="Init.Data.List.Basic",
            statement=statement,
            family="inductive_list",
            parameters={},
            eligible_premise_names=self.PREMISES_INDUCTION,
        )
        return thm, proof

    def generate_list_append_assoc(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate associativity of list concatenation."""
        thm_name = f"list_append_assoc_inductive_{idx}"
        statement = "∀ (α : Type) (xs ys zs : List α), (xs ++ ys) ++ zs = xs ++ (ys ++ zs)"
        proof = [
            "intro α xs ys zs",
            "induction xs with",
            "| nil => rfl",
            "| cons h t ih =>",
            "  simp only [List.cons_append]",
            "  rw [ih]",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_ind_{thm_name}",
            name=thm_name,
            module="Init.Data.List.Basic",
            statement=statement,
            family="inductive_list",
            parameters={},
            eligible_premise_names=self.PREMISES_INDUCTION,
        )
        return thm, proof

    def generate_inductive_batch(self, count: int = 100) -> list[tuple[TheoremSpec, list[str]]]:
        """Generate a diverse batch of structural induction theorems."""
        results: list[tuple[TheoremSpec, list[str]]] = []
        for i in range(count):
            choice = i % 4
            if choice == 0:
                thm, prf = self.generate_peano_addition_zero(idx=i)
            elif choice == 1:
                thm, prf = self.generate_peano_succ_add(idx=i)
            elif choice == 2:
                thm, prf = self.generate_list_append_nil(idx=i)
            else:
                thm, prf = self.generate_list_append_assoc(idx=i)
            results.append((thm, prf))
        return results
