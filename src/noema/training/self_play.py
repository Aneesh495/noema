"""Curriculum self-play, conjecture mutation, and regret matching for autonomous proof discovery."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence
from noema.contracts.theorems import TheoremSpec


@dataclass
class SelfPlayProofTask:
    """A generated self-play task along with difficulty calibration."""
    task_id: str
    theorem: TheoremSpec
    difficulty_score: float = 0.5
    generation_depth: int = 1
    mutation_history: list[str] = field(default_factory=list)


class ConjectureMutator:
    """Synthesizes new mathematical conjectures at the capability boundary of the prover."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def mutate_generalization(self, base_thm: TheoremSpec, idx: int = 0) -> TheoremSpec:
        """Generalize concrete instances into universally quantified variables."""
        stmt = base_thm.statement
        # Replace 0 or 1 with quantified variables if present
        new_stmt = stmt
        if "0" in stmt:
            new_stmt = f"∀ (z : Nat), {stmt.replace('0', 'z')}"
        elif "x" in stmt and "∀" not in stmt:
            new_stmt = f"∀ (x : Nat), {stmt}"

        return TheoremSpec(
            theorem_id=f"mut_gen_{base_thm.theorem_id}_{idx}",
            name=f"{base_thm.name}_gen_{idx}",
            module=base_thm.module,
            statement=new_stmt,
            family=f"{base_thm.family}_mutation",
            parameters={"mutation": "generalization"},
            eligible_premise_names=base_thm.eligible_premise_names,
        )

    def mutate_duality(self, base_thm: TheoremSpec, idx: int = 0) -> TheoremSpec:
        """Transpose operations under algebraic or lattice duality."""
        stmt = base_thm.statement
        # Swap ∧ and ∨ or + and *
        trans_table = str.maketrans({"∧": "∨", "∨": "∧"})
        dual_stmt = stmt.translate(trans_table)

        return TheoremSpec(
            theorem_id=f"mut_dual_{base_thm.theorem_id}_{idx}",
            name=f"{base_thm.name}_dual_{idx}",
            module=base_thm.module,
            statement=dual_stmt,
            family=f"{base_thm.family}_mutation",
            parameters={"mutation": "duality"},
            eligible_premise_names=base_thm.eligible_premise_names,
        )

    def mutate_composition_chain(self, base_thm: TheoremSpec, depth: int = 2, idx: int = 0) -> TheoremSpec:
        """Compose functions or associative operations deeper."""
        thm_name = f"{base_thm.name}_chain_{depth}_{idx}"
        return TheoremSpec(
            theorem_id=f"mut_chain_{base_thm.theorem_id}_{idx}",
            name=thm_name,
            module=base_thm.module,
            statement=base_thm.statement,
            family=f"{base_thm.family}_mutation",
            parameters={"mutation": "composition_chain", "depth": str(depth)},
            eligible_premise_names=base_thm.eligible_premise_names,
        )


class RegretMatchingPolicy:
    """Regret-matching multi-armed bandit policy for tactic selection during self-play."""

    def __init__(self, actions: Sequence[str]):
        self.actions = list(actions)
        self.cumulative_regret: dict[str, float] = {a: 0.0 for a in self.actions}

    def get_action_probabilities(self) -> dict[str, float]:
        """Compute action probabilities proportional to positive cumulative regret."""
        pos_regrets = {a: max(0.0, r) for a, r in self.cumulative_regret.items()}
        total = sum(pos_regrets.values())
        if total <= 1e-8:
            # Uniform fallback
            uniform_prob = 1.0 / len(self.actions)
            return {a: uniform_prob for a in self.actions}
        return {a: r / total for a, r in pos_regrets.items()}

    def select_action(self, rng: random.Random | None = None) -> str:
        """Sample an action according to regret matching distribution."""
        probs = self.get_action_probabilities()
        r = (rng or random).random()
        cumulative = 0.0
        for a, p in probs.items():
            cumulative += p
            if r <= cumulative:
                return a
        return self.actions[-1]

    def update_regret(self, chosen_action: str, actual_reward: float, counterfactual_rewards: dict[str, float]) -> None:
        """Update regret table: R(a) += counterfactual_reward(a) - actual_reward."""
        for a in self.actions:
            cf_r = counterfactual_rewards.get(a, actual_reward)
            self.cumulative_regret[a] += (cf_r - actual_reward)


class CurriculumSelfPlayEngine:
    """Orchestrates autonomous conjecture mutation, regret-guided exploration, and data generation."""

    def __init__(
        self,
        base_theorems: Sequence[TheoremSpec],
        mutator: ConjectureMutator | None = None,
        target_success_rate: float = 0.6,
    ):
        self.active_pool: list[SelfPlayProofTask] = [
            SelfPlayProofTask(
                task_id=f"task_{t.theorem_id}",
                theorem=t,
                difficulty_score=0.5,
            )
            for t in base_theorems
        ]
        self.mutator = mutator or ConjectureMutator()
        self.target_rate = target_success_rate
        self.solved_tasks: list[SelfPlayProofTask] = []
        self.failed_tasks: list[SelfPlayProofTask] = []

    def sample_task(self, rng: random.Random | None = None) -> SelfPlayProofTask:
        """Sample a task from the active frontier pool biased toward current capability boundary."""
        if not self.active_pool:
            raise ValueError("Active task pool is empty")
        r = rng or random
        # Pick task closest to target capability boundary
        return min(self.active_pool, key=lambda t: abs(t.difficulty_score - self.target_rate) + r.random() * 0.1)

    def record_attempt(
        self,
        task: SelfPlayProofTask,
        solved: bool,
        steps_taken: int,
    ) -> list[SelfPlayProofTask]:
        """Record the outcome of a proof attempt and synthesize new mutant tasks."""
        new_tasks: list[SelfPlayProofTask] = []

        if solved:
            self.solved_tasks.append(task)
            # If solved very quickly, escalate difficulty via mutation
            if steps_taken <= 2 and task.generation_depth < 4:
                mutant_thm = self.mutator.mutate_generalization(
                    task.theorem, idx=len(self.solved_tasks)
                )
                mutant_task = SelfPlayProofTask(
                    task_id=f"task_{mutant_thm.theorem_id}",
                    theorem=mutant_thm,
                    difficulty_score=min(1.0, task.difficulty_score + 0.15),
                    generation_depth=task.generation_depth + 1,
                    mutation_history=task.mutation_history + ["generalization"],
                )
                self.active_pool.append(mutant_task)
                new_tasks.append(mutant_task)
        else:
            self.failed_tasks.append(task)
            # Reduce difficulty score
            task.difficulty_score = max(0.1, task.difficulty_score - 0.1)

        return new_tasks
