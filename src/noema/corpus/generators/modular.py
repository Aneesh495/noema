"""Modular arithmetic and number theory domain theorem generator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence
from noema.contracts.theorems import TheoremSpec


@dataclass(frozen=True)
class NumberTheoryTheoremTemplate:
    """Template describing a divisibility or modular arithmetic theorem."""
    template_id: str
    property_type: str
    pattern_statement: str
    tactic_skeleton: list[str]
    required_premises: list[str]


class ModularTheoremGenerator:
    """Generates synthetic theorems in divisibility, modular arithmetic, and greatest common divisors."""

    PREMISES_DIVISIBILITY = [
        "Nat.dvd_trans",
        "Nat.dvd_add",
        "Nat.dvd_mul_right",
        "Nat.dvd_refl",
    ]

    PREMISES_MODULAR = [
        "Nat.mod_mod",
        "Nat.add_mod",
        "Nat.mul_mod",
        "Nat.zero_mod",
    ]

    PREMISES_GCD = [
        "Nat.gcd_comm",
        "Nat.gcd_assoc",
        "Nat.gcd_zero_right",
        "Nat.gcd_self",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_divisibility_transitivity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate transitivity of divisibility: a ∣ b -> b ∣ c -> a ∣ c."""
        thm_name = f"nat_dvd_trans_{idx}"
        statement = "∀ (a b c : Nat), a ∣ b → b ∣ c → a ∣ c"
        tactics = [
            "intro a b c hab hbc",
            "exact Nat.dvd_trans hab hbc",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_nt_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="number_theory_divisibility",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_DIVISIBILITY,
        )
        return thm, tactics

    def generate_divisibility_linear_combination(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate divisibility of sum: a ∣ b -> a ∣ c -> a ∣ (b + c)."""
        thm_name = f"nat_dvd_add_{idx}"
        statement = "∀ (a b c : Nat), a ∣ b → a ∣ c → a ∣ (b + c)"
        tactics = [
            "intro a b c hab hac",
            "exact Nat.dvd_add hab hac",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_nt_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="number_theory_divisibility",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_DIVISIBILITY,
        )
        return thm, tactics

    def generate_mod_idempotence(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate idempotence of remainder operator: (a % m) % m = a % m."""
        thm_name = f"nat_mod_idemp_{idx}"
        statement = "∀ (a m : Nat), (a % m) % m = a % m"
        tactics = [
            "intro a m",
            "exact Nat.mod_mod a m",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_nt_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="number_theory_modular",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_MODULAR,
        )
        return thm, tactics

    def generate_add_mod_identity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate modular addition homomorphism: (a + b) % m = (a % m + b % m) % m."""
        thm_name = f"nat_add_mod_hom_{idx}"
        statement = "∀ (a b m : Nat), (a + b) % m = (a % m + b % m) % m"
        tactics = [
            "intro a b m",
            "exact Nat.add_mod a b m",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_nt_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="number_theory_modular",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_MODULAR,
        )
        return thm, tactics

    def generate_mul_mod_identity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate modular multiplication homomorphism: (a * b) % m = ((a % m) * (b % m)) % m."""
        thm_name = f"nat_mul_mod_hom_{idx}"
        statement = "∀ (a b m : Nat), (a * b) % m = ((a % m) * (b % m)) % m"
        tactics = [
            "intro a b m",
            "exact Nat.mul_mod a b m",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_nt_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="number_theory_modular",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_MODULAR,
        )
        return thm, tactics

    def generate_congruence_symmetry(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate symmetry of modular equivalence."""
        thm_name = f"nat_mod_symm_{idx}"
        statement = "∀ (a b m : Nat), a % m = b % m → b % m = a % m"
        tactics = [
            "intro a b m h",
            "exact h.symm",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_nt_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="number_theory_modular",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_MODULAR,
        )
        return thm, tactics

    def generate_congruence_transitivity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate transitivity of modular equivalence."""
        thm_name = f"nat_mod_trans_{idx}"
        statement = "∀ (a b c m : Nat), a % m = b % m → b % m = c % m → a % m = c % m"
        tactics = [
            "intro a b c m hab hbc",
            "exact hab.trans hbc",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_nt_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="number_theory_modular",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_MODULAR,
        )
        return thm, tactics

    def generate_gcd_commutativity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate commutativity of greatest common divisor: gcd a b = gcd b a."""
        thm_name = f"nat_gcd_comm_{idx}"
        statement = "∀ (a b : Nat), Nat.gcd a b = Nat.gcd b a"
        tactics = [
            "intro a b",
            "exact Nat.gcd_comm a b",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_nt_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="number_theory_gcd",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_GCD,
        )
        return thm, tactics

    def generate_gcd_associativity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate associativity of greatest common divisor."""
        thm_name = f"nat_gcd_assoc_{idx}"
        statement = "∀ (a b c : Nat), Nat.gcd (Nat.gcd a b) c = Nat.gcd a (Nat.gcd b c)"
        tactics = [
            "intro a b c",
            "exact Nat.gcd_assoc a b c",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_nt_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="number_theory_gcd",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_GCD,
        )
        return thm, tactics

    def generate_gcd_zero_identity(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate identity gcd a 0 = a."""
        thm_name = f"nat_gcd_zero_{idx}"
        statement = "∀ (a : Nat), Nat.gcd a 0 = a"
        tactics = [
            "intro a",
            "exact Nat.gcd_zero_right a",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_nt_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="number_theory_gcd",
            parameters={"idx": str(idx)},
            eligible_premise_names=self.PREMISES_GCD,
        )
        return thm, tactics

    def generate_batch(self, total: int = 20) -> list[tuple[TheoremSpec, list[str]]]:
        """Generate a diverse batch of number theory and modular arithmetic theorems."""
        results: list[tuple[TheoremSpec, list[str]]] = []
        generators = [
            lambda i: self.generate_divisibility_transitivity(i),
            lambda i: self.generate_divisibility_linear_combination(i),
            lambda i: self.generate_mod_idempotence(i),
            lambda i: self.generate_add_mod_identity(i),
            lambda i: self.generate_mul_mod_identity(i),
            lambda i: self.generate_congruence_symmetry(i),
            lambda i: self.generate_congruence_transitivity(i),
            lambda i: self.generate_gcd_commutativity(i),
            lambda i: self.generate_gcd_associativity(i),
            lambda i: self.generate_gcd_zero_identity(i),
        ]
        for idx in range(total):
            gen = generators[idx % len(generators)]
            results.append(gen(idx))
        return results
