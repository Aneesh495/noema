"""Order theory and lattice theorem generator: posets, chains, and bounds."""

from __future__ import annotations

import random
from dataclasses import dataclass
from noema.contracts.theorems import TheoremSpec


class OrderTheoremGenerator:
    """Generates order-theoretic propositions with ground-truth Lean 4 proofs."""

    PREMISES_ORDER = [
        "Nat.le_refl",
        "Nat.le_trans",
        "Nat.le_antisymm",
        "Nat.le_add_right",
        "Nat.le_add_left",
        "Nat.add_le_add",
        "Nat.min_le_left",
        "Nat.min_le_right",
        "Nat.le_max_left",
        "Nat.le_max_right",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_transitivity_chain(self, chain_len: int = 3, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate a chain of inequalities x0 <= x1 <= ... <= xn -> x0 <= xn."""
        vars_list = [f"x{i}" for i in range(chain_len + 1)]
        vars_decl = " ".join(vars_list)

        # Hypotheses: x0 <= x1 -> x1 <= x2 -> ... -> x0 <= xn
        hyps = [f"{vars_list[i]} ≤ {vars_list[i+1]}" for i in range(chain_len)]
        hyp_str = " → ".join(hyps)
        conclusion = f"{vars_list[0]} ≤ {vars_list[-1]}"

        statement = f"∀ ({vars_decl} : Nat), {hyp_str} → {conclusion}"
        thm_name = f"nat_order_trans_chain_{chain_len}_{idx}"

        proof = [
            f"intro {' '.join(vars_list)}",
            f"intro {' '.join(f'h{i}' for i in range(chain_len))}",
            "omega",
        ]

        thm = TheoremSpec(
            theorem_id=f"thm_order_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="order_transitivity",
            parameters={"chain_len": str(chain_len)},
            eligible_premise_names=self.PREMISES_ORDER,
        )
        return thm, proof

    def generate_monotonicity_sum(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate monotonicity of addition: a <= b -> c <= d -> a + c <= b + d."""
        thm_name = f"nat_order_add_le_add_{idx}"
        statement = "∀ (a b c d : Nat), a ≤ b → c ≤ d → a + c ≤ b + d"
        proof = [
            "intro a b c d hab hcd",
            "omega",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_order_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="order_monotonicity",
            parameters={},
            eligible_premise_names=self.PREMISES_ORDER,
        )
        return thm, proof

    def generate_min_max_lattice(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate semilattice bounding inequalities: min a b <= a + b."""
        thm_name = f"nat_min_le_add_{idx}"
        statement = "∀ (a b : Nat), min a b ≤ a + b"
        proof = [
            "intro a b",
            "omega",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_order_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="order_lattice",
            parameters={},
            eligible_premise_names=self.PREMISES_ORDER,
        )
        return thm, proof

    def generate_trichotomy_decision(self, idx: int = 0) -> tuple[TheoremSpec, list[str]]:
        """Generate order trichotomy implications."""
        thm_name = f"nat_order_strict_le_{idx}"
        statement = "∀ (a b : Nat), a < b → a ≤ b"
        proof = [
            "intro a b h",
            "omega",
        ]
        thm = TheoremSpec(
            theorem_id=f"thm_order_{thm_name}",
            name=thm_name,
            module="Init.Data.Nat.Basic",
            statement=statement,
            family="order_trichotomy",
            parameters={},
            eligible_premise_names=self.PREMISES_ORDER,
        )
        return thm, proof

    def generate_order_batch(self, count: int = 100) -> list[tuple[TheoremSpec, list[str]]]:
        """Generate a balanced batch of order theory theorems."""
        results: list[tuple[TheoremSpec, list[str]]] = []
        for i in range(count):
            choice = i % 4
            if choice == 0:
                length = (i % 3) + 2
                thm, prf = self.generate_transitivity_chain(chain_len=length, idx=i)
            elif choice == 1:
                thm, prf = self.generate_monotonicity_sum(idx=i)
            elif choice == 2:
                thm, prf = self.generate_min_max_lattice(idx=i)
            else:
                thm, prf = self.generate_trichotomy_decision(idx=i)
            results.append((thm, prf))
        return results
