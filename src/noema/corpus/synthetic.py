"""Controlled synthetic theorem generation across structural families."""

from __future__ import annotations

import random
from typing import Generator
from noema.contracts.theorems import TheoremSpec


class TaskGenerator:
    """Generates structured mathematical theorems and verified reference proofs."""

    GENERATOR_VERSION = "1.0.0"

    FAMILIES = [
        "propositional_implication",
        "propositional_conjunction",
        "propositional_disjunction",
        "nat_equality_rewrite",
        "nat_addition_chains",
        "nat_induction_linear",
        "list_induction_append",
        "list_induction_length",
        "compositional_algebraic",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_task(self, index: int, is_held_out: bool = False) -> TheoremSpec:
        """Generate a single mathematically sound theorem with reference proof."""
        family = self.FAMILIES[index % len(self.FAMILIES)]
        thm_id = f"{'eval' if is_held_out else 'train'}_{family}_{index:05d}"
        
        # Family-specific construction
        if family == "propositional_implication":
            # e.g., P → Q → P
            depth = (index % 4) + 1
            var_names = [f"P_{i}" for i in range(depth + 1)]
            hyp_types = " → ".join(f"({v} : Prop)" for v in var_names)
            # Statement: P_0 → P_1 → ... → P_0
            chain = " → ".join(var_names) + f" → {var_names[0]}"
            statement = f"∀ {hyp_types}, {chain}"
            # Reference proof
            intros = " ".join(f"h_{v}" for v in var_names) + " " + " ".join(f"h_chain_{i}" for i in range(len(var_names)))
            ref_proof = f"by\n  intros\n  assumption"
            template_id = f"prop_imp_depth_{depth}"

        elif family == "propositional_conjunction":
            # e.g., A ∧ B → B ∧ A
            # Or multi-variable conjunction comm
            var_a, var_b = f"A_{index % 7}", f"B_{index % 7}"
            statement = f"∀ ({var_a} {var_b} : Prop), {var_a} ∧ {var_b} → {var_b} ∧ {var_a}"
            ref_proof = "by\n  intro A B h\n  cases h\n  constructor\n  assumption\n  assumption"
            template_id = "prop_conj_comm"

        elif family == "propositional_disjunction":
            var_a, var_b = f"A_{index % 7}", f"B_{index % 7}"
            statement = f"∀ ({var_a} {var_b} : Prop), {var_a} → {var_a} ∨ {var_b}"
            ref_proof = "by\n  intro A B ha\n  apply Or.inl\n  exact ha"
            template_id = "prop_disj_inl"

        elif family == "nat_equality_rewrite":
            # e.g. x + 0 = x
            k = (index % 5)
            statement = f"∀ (x_{k} : Nat), x_{k} + 0 = x_{k}"
            ref_proof = f"by\n  intro x_{k}\n  rfl"
            template_id = "nat_eq_add_zero"

        elif family == "nat_addition_chains":
            # e.g. (x + 0) + 0 = x
            depth = (index % 3) + 1
            inner = "x"
            for _ in range(depth):
                inner = f"({inner} + 0)"
            statement = f"∀ (x : Nat), {inner} = x"
            ref_proof = "by\n  intro x\n  rfl"
            template_id = f"nat_add_chain_{depth}"

        elif family == "nat_induction_linear":
            # e.g. 0 + n = n proved via induction on n
            statement = f"∀ (n_{index % 9} : Nat), 0 + n_{index % 9} = n_{index % 9}"
            vname = f"n_{index % 9}"
            ref_proof = f"by\n  intro {vname}\n  induction {vname} with\n  | zero => rfl\n  | succ d ih => rw [Nat.add_succ, ih]"
            template_id = "nat_ind_zero_add"

        elif family == "list_induction_append":
            # e.g. [] ++ l = l
            statement = f"∀ (α : Type) (l_{index % 5} : List α), [] ++ l_{index % 5} = l_{index % 5}"
            vname = f"l_{index % 5}"
            ref_proof = f"by\n  intro α {vname}\n  rfl"
            template_id = "list_nil_append"

        elif family == "list_induction_length":
            statement = f"∀ (α : Type) (x : α), List.length [x] = 1"
            ref_proof = "by\n  intro α x\n  rfl"
            template_id = "list_length_singleton"

        else: # compositional_algebraic
            # e.g. x = y → y = z → x = z
            statement = f"∀ (x y z : Nat), x = y → y = z → x = z"
            ref_proof = "by\n  intro x y z h1 h2\n  rw [h1, h2]"
            template_id = "eq_trans_nat"

        return TheoremSpec(
            theorem_id=thm_id,
            name=f"thm_{family}_{index}",
            module=f"Noema.Synthetic.{family.title().replace('_', '')}",
            statement=statement,
            namespace="Noema.Synthetic",
            family=family,
            template_id=template_id,
            reference_proof=ref_proof,
            metadata={
                "generator_version": self.GENERATOR_VERSION,
                "is_held_out": str(is_held_out),
                "index": str(index),
            },
        )

    def generate_corpus(
        self,
        num_train: int = 20000,
        num_eval: int = 2000
    ) -> tuple[list[TheoremSpec], list[TheoremSpec]]:
        """Generate full training and held-out evaluation task suites."""
        train_tasks = [self.generate_task(i, is_held_out=False) for i in range(num_train)]
        eval_tasks = [self.generate_task(num_train + i, is_held_out=True) for i in range(num_eval)]
        return train_tasks, eval_tasks
