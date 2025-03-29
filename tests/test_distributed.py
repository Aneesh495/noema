"""Tests for distributed optimization: gradient accumulation, loss scaling, and EMA."""

import pytest
import torch
import torch.nn as nn
from noema.training.distributed import (
    DynamicLossScaler,
    ExponentialMovingAverage,
    GradientAccumulator,
)


def test_gradient_accumulator_step_cadence():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    acc = GradientAccumulator(model, optimizer, accumulation_steps=3, max_grad_norm=1.0)

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)

    # Step 1: should not trigger optimizer step
    loss1 = nn.functional.mse_loss(model(x), target)
    did_step1, _, _ = acc.backward_and_step(loss1)
    assert not did_step1
    assert acc.current_step == 1

    # Step 2: should not trigger
    loss2 = nn.functional.mse_loss(model(x), target)
    did_step2, _, _ = acc.backward_and_step(loss2)
    assert not did_step2
    assert acc.current_step == 2

    # Step 3: accumulation boundary reached -> should trigger optimizer step
    loss3 = nn.functional.mse_loss(model(x), target)
    did_step3, _, norm = acc.backward_and_step(loss3)
    assert did_step3
    assert norm > 0.0
    assert acc.current_step == 3


def test_dynamic_loss_scaler_inf_handling():
    scaler = DynamicLossScaler(init_scale=1024.0, backoff_factor=0.5)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = nn.functional.mse_loss(model(torch.randn(2, 4)), torch.randn(2, 2))
    scaled = scaler.scale_loss(loss)
    scaled.backward()

    # Manually inject infinity into gradient
    list(model.parameters())[0].grad.data[0, 0] = float("inf")

    stepped, new_scale = scaler.step(optimizer, model.parameters())
    # Should detect inf, skip optimizer step, and halve scale
    assert not stepped
    assert new_scale == 512.0


def test_exponential_moving_average_smoothing():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    ema = ExponentialMovingAverage(model, decay=0.9)
    assert torch.allclose(ema.shadow["weight"], torch.tensor([[1.0, 1.0], [1.0, 1.0]]))

    # Modify model weights
    with torch.no_grad():
        model.weight.fill_(2.0)

    ema.update(model)
    # EMA = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    assert torch.allclose(ema.shadow["weight"], torch.tensor([[1.1, 1.1], [1.1, 1.1]]))

    # Apply to model
    backup = ema.apply_to(model)
    assert torch.allclose(model.weight, torch.tensor([[1.1, 1.1], [1.1, 1.1]]))

    # Restore from backup
    ema.restore_from(model, backup)
    assert torch.allclose(model.weight, torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
