"""Unit tests for PPO and GAE reinforcement learning modules."""

import pytest
import torch
from noema.training.ppo import (
    PPOBatch,
    PPOCriterion,
    PPOReplayBuffer,
    PPOTrajectoryStep,
    compute_generalized_advantages,
)


def test_compute_generalized_advantages():
    rewards = [0.0, 0.0, 1.0]
    values = [0.2, 0.5, 0.8]
    dones = [False, False, True]

    adv, returns = compute_generalized_advantages(rewards, values, dones, gamma=0.99, lam=0.95)

    assert adv.shape == (3,)
    assert returns.shape == (3,)
    # Normalized advantages should have roughly zero mean
    assert abs(adv.mean().item()) < 1e-4


def test_ppo_criterion_clipped_surrogate():
    criterion = PPOCriterion(clip_epsilon=0.2, vf_coeff=0.5, kl_coeff=0.01)

    new_log_p = torch.tensor([-0.5, -0.6, -0.2], requires_grad=True)
    old_log_p = torch.tensor([-0.5, -0.5, -0.5])
    advantages = torch.tensor([1.0, -1.0, 0.5])
    new_values = torch.tensor([0.4, 0.2, 0.8], requires_grad=True)
    target_returns = torch.tensor([0.5, 0.0, 1.0])

    res = criterion.compute_loss(
        new_log_probs=new_log_p,
        old_log_probs=old_log_p,
        ref_log_probs=None,
        advantages=advantages,
        new_values=new_values,
        target_returns=target_returns,
    )

    assert res.total_loss.requires_grad
    assert res.policy_loss.item() != 0.0
    assert res.value_loss.item() >= 0.0
    assert 0.0 <= res.clip_fraction <= 1.0

    # Test backward pass
    res.total_loss.backward()
    assert new_log_p.grad is not None
    assert new_values.grad is not None


def test_ppo_replay_buffer_and_batch_creation():
    buffer = PPOReplayBuffer(capacity=5)
    dim = 16

    for i in range(3):
        step = PPOTrajectoryStep(
            state_embedding=torch.randn(dim),
            action_token_ids=torch.tensor([1, 2, 3]),
            old_log_prob=-0.5,
            value_pred=0.1 * i,
            reward=1.0 if i == 2 else 0.0,
            is_terminal=(i == 2),
        )
        buffer.push(step)

    assert len(buffer.steps) == 3
    batch = buffer.create_batch(gamma=0.99, lam=0.95)

    assert batch.state_embeddings.shape == (3, dim)
    assert batch.old_log_probs.shape == (3,)
    assert batch.advantages.shape == (3,)
    assert batch.returns.shape == (3,)

    buffer.clear()
    assert buffer.is_empty()
