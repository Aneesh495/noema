"""Contamination controls, scope isolation, and anti-leakage audits."""

from __future__ import annotations

import re
from typing import Sequence
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec


class ContaminationAuditor:
    """Audits environments and datasets to ensure zero contamination between training and evaluation."""

    @classmethod
    def audit_environment_scope(
        cls,
        target_theorem: TheoremSpec,
        env_spec: EnvironmentSpec,
    ) -> tuple[bool, list[str]]:
        """Verify that the target theorem and its descendants are excluded from the environment."""
        violations = []
        target_name = target_theorem.name
        
        # 1. Exact target name in eligible premises
        if target_name in env_spec.eligible_premises:
            violations.append(f"Target theorem '{target_name}' is present in eligible premises")

        # 2. Descendant declarations of target
        prefix = f"{target_name}."
        for p in env_spec.eligible_premises:
            if p.startswith(prefix):
                violations.append(f"Descendant declaration '{p}' of target is in eligible premises")

        # 3. Target namespace collision if target belongs to an evaluation suite
        if "eval" in target_theorem.theorem_id or "heldout" in target_theorem.theorem_id:
            for p in env_spec.eligible_premises:
                if p.startswith(target_theorem.module):
                    violations.append(f"Premise '{p}' from evaluation module '{target_theorem.module}' in environment")

        return (len(violations) == 0, violations)

    @classmethod
    def audit_splits(
        cls,
        train_tasks: Sequence[TheoremSpec],
        eval_tasks: Sequence[TheoremSpec],
    ) -> tuple[bool, dict[str, int]]:
        """Audit train and evaluation splits for duplicate statements, alias collisions, and ID leaks."""
        train_statements = {t.statement_hash(): t.theorem_id for t in train_tasks}
        train_names = {t.name for t in train_tasks}
        
        statement_collisions = 0
        name_collisions = 0

        for et in eval_tasks:
            shash = et.statement_hash()
            if shash in train_statements:
                statement_collisions += 1
            if et.name in train_names:
                name_collisions += 1

        stats = {
            "num_train": len(train_tasks),
            "num_eval": len(eval_tasks),
            "statement_collisions": statement_collisions,
            "name_collisions": name_collisions,
        }
        
        is_clean = (statement_collisions == 0 and name_collisions == 0)
        return is_clean, stats

    @classmethod
    def mask_reference_proofs_for_inference(cls, theorem: TheoremSpec) -> TheoremSpec:
        """Create a sanitized copy of a theorem with reference proof strictly erased."""
        copy_dict = theorem.model_dump()
        copy_dict["reference_proof"] = None
        return TheoremSpec(**copy_dict)
