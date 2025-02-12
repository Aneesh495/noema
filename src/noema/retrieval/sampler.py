"""Hard-negative premise sampling restricted to eligible environment declarations."""

from __future__ import annotations

import random
from noema.contracts.environment import EnvironmentSpec, PremiseDecl


class HardNegativeSampler:
    """Samples challenging negative premises based on lexical similarity and module co-occurrence."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def sample_negatives(
        self,
        positive_premise_name: str,
        env_spec: EnvironmentSpec,
        num_negatives: int = 4,
        target_name: str | None = None,
    ) -> list[str]:
        """Sample hard negative premises from eligible premises only."""
        eligible_keys = list(env_spec.eligible_premises.keys())
        if not eligible_keys:
            return []

        pos_decl = env_spec.eligible_premises.get(positive_premise_name)
        pos_module = pos_decl.module if pos_decl else ""

        # Pool of eligible candidates excluding positive and target
        pool = [
            k for k in eligible_keys
            if k != positive_premise_name and (target_name is None or k != target_name)
        ]
        if not pool:
            return []

        # Find module-sharing candidates (harder negatives)
        same_module = [k for k in pool if env_spec.eligible_premises[k].module == pos_module]
        diff_module = [k for k in pool if env_spec.eligible_premises[k].module != pos_module]

        chosen: list[str] = []
        # Prefer same module if available
        if same_module:
            k_same = min(len(same_module), num_negatives // 2 + 1)
            chosen.extend(self.rng.sample(same_module, k_same))

        remaining = num_negatives - len(chosen)
        if remaining > 0 and diff_module:
            k_diff = min(len(diff_module), remaining)
            chosen.extend(self.rng.sample(diff_module, k_diff))

        # If still short, sample from whatever is available in pool
        while len(chosen) < num_negatives and len(chosen) < len(pool):
            cand = self.rng.choice(pool)
            if cand not in chosen:
                chosen.append(cand)

        return chosen
