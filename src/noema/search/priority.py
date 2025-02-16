"""Search priority functions and heuristic exploration strategies."""

from __future__ import annotations

import math
import random
from enum import Enum
from noema.contracts.actions import TacticAction
from noema.contracts.states import ProofState


class SearchStrategy(str, Enum):
    NEURAL_GUIDED = "neural_guided"
    UNIFORM_BFS = "uniform_bfs"
    RANDOM_PRIORITY = "random_priority"
    LEXICAL_GUIDED = "lexical_guided"
    POLICY_ONLY = "policy_only"


class PriorityScorer:
    """Computes node priority queue score based on chosen search strategy."""

    def __init__(
        self,
        strategy: SearchStrategy = SearchStrategy.NEURAL_GUIDED,
        alpha_policy: float = 1.0,
        beta_value: float = 2.0,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.alpha_policy = alpha_policy
        self.beta_value = beta_value
        self.rng = random.Random(seed)

    def compute_score(
        self,
        state: ProofState,
        action: TacticAction | None = None,
        policy_prob: float = 1.0,
        value_estimate: float = 0.5,
        lexical_score: float = 0.0,
    ) -> float:
        """
        Compute queue priority. Lower values indicate higher search priority (min-heap convention).
        """
        depth_cost = float(state.depth)

        if self.strategy == SearchStrategy.UNIFORM_BFS:
            return depth_cost

        elif self.strategy == SearchStrategy.RANDOM_PRIORITY:
            return depth_cost + self.rng.random() * 10.0

        elif self.strategy == SearchStrategy.LEXICAL_GUIDED:
            # Lower score for higher lexical premise match
            return depth_cost - lexical_score

        elif self.strategy == SearchStrategy.POLICY_ONLY:
            # Policy log-prob guidance without value function
            log_prob = math.log(max(policy_prob, 1e-6))
            return depth_cost - self.alpha_policy * log_prob

        else: # NEURAL_GUIDED (policy + value)
            log_prob = math.log(max(policy_prob, 1e-6))
            # Higher value estimate -> lower priority score (expanded earlier)
            return depth_cost - self.alpha_policy * log_prob - self.beta_value * value_estimate
