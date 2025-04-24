"""Proximal Policy Optimization (PPO) and Generalized Advantage Estimation (GAE) for theorem proving."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PPOTrajectoryStep:
    """A single decision step in a theorem proving trajectory."""
    state_embedding: torch.Tensor
    action_token_ids: torch.Tensor
    old_log_prob: float
    value_pred: float
    reward: float
    is_terminal: bool
    subgoal_delta: int = 0


@dataclass
class PPOBatch:
    """Pre-computed tensor batch for PPO updates."""
    state_embeddings: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


def compute_generalized_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) and target returns.
    
    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t = sum_{l=0} (gamma * lam)^l * delta_{t+l}
    """
    n = len(rewards)
    advantages = [0.0] * n
    last_gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1:
            next_val = 0.0
            next_done = True
        else:
            next_val = values[t + 1]
            next_done = dones[t + 1]

        non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae

    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    val_tensor = torch.tensor(values, dtype=torch.float32)
    returns = adv_tensor + val_tensor

    # Standardize advantages
    if adv_tensor.numel() > 1:
        adv_mean = adv_tensor.mean()
        adv_std = adv_tensor.std(unbiased=False) + 1e-8
        adv_tensor = (adv_tensor - adv_mean) / adv_std

    return adv_tensor, returns


@dataclass
class PPOLossResult:
    """Decomposition of PPO objectives and diagnostics."""
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    kl_divergence: torch.Tensor
    clip_fraction: float
    approx_entropy: torch.Tensor


class PPOCriterion:
    """PPO clipped surrogate loss, clipped value function loss, and KL divergence regularizer."""

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        vf_coeff: float = 0.5,
        kl_coeff: float = 0.01,
        entropy_coeff: float = 0.005,
    ):
        self.clip_eps = clip_epsilon
        self.vf_coeff = vf_coeff
        self.kl_coeff = kl_coeff
        self.entropy_coeff = entropy_coeff

    def compute_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor | None,
        advantages: torch.Tensor,
        new_values: torch.Tensor,
        target_returns: torch.Tensor,
        old_values: torch.Tensor | None = None,
    ) -> PPOLossResult:
        """Compute PPO clipped policy and value objective."""
        # 1. Probability ratio r_t(theta) = exp(new_log_p - old_log_p)
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # 2. Clipped surrogate policy loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        with torch.no_grad():
            clipped = (ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)
            clip_fraction = clipped.float().mean().item()

        # 3. Value function loss with optional clipping
        if old_values is not None:
            val_clipped = old_values + torch.clamp(
                new_values - old_values, -self.clip_eps, self.clip_eps
            )
            vf_loss1 = (new_values - target_returns) ** 2
            vf_loss2 = (val_clipped - target_returns) ** 2
            value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
        else:
            value_loss = 0.5 * F.mse_loss(new_values, target_returns)

        # 4. KL divergence penalty against reference policy
        if ref_log_probs is not None:
            # KL(ref || new) approx ref_log - new_log
            kl_div = (ref_log_probs - new_log_probs).mean()
        else:
            # Approximate KL from old policy: (ratio - 1) - log(ratio)
            kl_div = ((ratio - 1.0) - log_ratio).mean()

        # 5. Approximate entropy
        approx_entropy = -new_log_probs.mean()

        # Total objective
        total_loss = (
            policy_loss
            + self.vf_coeff * value_loss
            + self.kl_coeff * kl_div
            - self.entropy_coeff * approx_entropy
        )

        return PPOLossResult(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            kl_divergence=kl_div,
            clip_fraction=clip_fraction,
            approx_entropy=approx_entropy,
        )


class PPOReplayBuffer:
    """In-memory rollout buffer storing theorem proving trajectory steps."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.steps: list[PPOTrajectoryStep] = []

    def push(self, step: PPOTrajectoryStep) -> None:
        if len(self.steps) >= self.capacity:
            self.steps.pop(0)
        self.steps.append(step)

    def is_empty(self) -> bool:
        return len(self.steps) == 0

    def clear(self) -> None:
        self.steps.clear()

    def create_batch(self, gamma: float = 0.99, lam: float = 0.95) -> PPOBatch:
        """Process collected trajectory steps into a PPOBatch."""
        if not self.steps:
            raise ValueError("Cannot create PPO batch from empty buffer")

        rewards = [s.reward for s in self.steps]
        values = [s.value_pred for s in self.steps]
        dones = [s.is_terminal for s in self.steps]

        advantages, returns = compute_generalized_advantages(
            rewards, values, dones, gamma=gamma, lam=lam
        )

        state_embeddings = torch.stack([s.state_embedding for s in self.steps])
        old_log_probs = torch.tensor([s.old_log_prob for s in self.steps], dtype=torch.float32)
        val_tensor = torch.tensor(values, dtype=torch.float32)

        return PPOBatch(
            state_embeddings=state_embeddings,
            old_log_probs=old_log_probs,
            returns=returns,
            advantages=advantages,
            values=val_tensor,
        )
