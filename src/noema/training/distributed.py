"""Distributed optimization, gradient accumulation, and mixed-precision scaling."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from typing import Iterator


class GradientAccumulator:
    """Manages multi-step gradient accumulation for simulating large effective batch sizes."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = max(1, accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.current_step = 0
        self.accumulated_loss = 0.0

    def backward_and_step(self, loss: torch.Tensor) -> tuple[bool, float, float]:
        """Perform scaled backward pass and step optimizer on accumulation boundary.
        
        Returns (did_step, current_loss_value, grad_norm).
        """
        loss_val = float(loss.item())
        self.accumulated_loss += loss_val

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        self.current_step += 1
        did_step = False
        grad_norm = 0.0

        if self.current_step % self.accumulation_steps == 0:
            if self.max_grad_norm > 0.0:
                grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm))
            self.optimizer.step()
            self.optimizer.zero_grad()
            did_step = True
            self.accumulated_loss = 0.0

        return did_step, loss_val, grad_norm


class DynamicLossScaler:
    """Dynamic loss scaler for float16 mixed-precision gradient stability."""

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        min_scale: float = 1.0,
        max_scale: float = 16777216.0,
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.consecutive_good_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss prior to backward pass."""
        return loss * self.scale

    def step(self, optimizer: torch.optim.Optimizer, parameters: Iterator[nn.Parameter]) -> tuple[bool, float]:
        """Unscale gradients, check for inf/nan, and adjust scale factor.
        
        Returns (stepped_optimizer, current_scale).
        """
        # Unscale gradients
        inv_scale = 1.0 / self.scale
        has_nan_or_inf = False

        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(inv_scale)
                if torch.isnan(p.grad.data).any() or torch.isinf(p.grad.data).any():
                    has_nan_or_inf = True

        if has_nan_or_inf:
            # Back off scale and skip step
            self.scale = max(self.min_scale, self.scale * self.backoff_factor)
            self.consecutive_good_steps = 0
            optimizer.zero_grad()
            return False, self.scale

        # Valid step
        optimizer.step()
        optimizer.zero_grad()
        self.consecutive_good_steps += 1

        if self.consecutive_good_steps >= self.growth_interval:
            self.scale = min(self.max_scale, self.scale * self.growth_factor)
            self.consecutive_good_steps = 0

        return True, self.scale


class ExponentialMovingAverage:
    """Polyak exponential moving average (EMA) of model parameters for evaluation checkpoints."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with exponential decay."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Copy shadow weights into model, returning backup of original weights."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.clone().detach()
                param.data.copy_(self.shadow[name])
        return backup

    def restore_from(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore original parameters from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
