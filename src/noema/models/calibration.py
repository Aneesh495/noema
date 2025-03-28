"""Uncertainty quantification, Expected Calibration Error (ECE), and temperature scaling."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CalibrationMetrics:
    """Quantitative metrics evaluating prediction calibration and confidence alignment."""
    expected_calibration_error: float
    max_calibration_error: float
    brier_score: float
    optimal_temperature: float


class TemperatureCalibrator(nn.Module):
    """Post-hoc temperature scaling calibrator for classification logits.
    
    References:
    - Guo et al. (2017) On Calibration of Modern Neural Networks
    """

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([float(init_temperature)]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature parameter."""
        temp = torch.clamp(self.temperature, min=0.01, max=50.0)
        return logits / temp

    def fit(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        lr: float = 0.01,
        max_iters: int = 100,
    ) -> float:
        """Optimize temperature parameter using NLL loss on held-out validation data."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = criterion(scaled, val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())

    @staticmethod
    def compute_ece(
        probs: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 15,
    ) -> tuple[float, float, float]:
        """Compute Expected Calibration Error (ECE), Max Calibration Error (MCE), and Brier Score."""
        confidences, predictions = torch.max(probs, dim=-1)
        accuracies = predictions.eq(targets)

        bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
        ece = torch.zeros(1, device=probs.device)
        mce = torch.zeros(1, device=probs.device)

        n_samples = probs.shape[0]

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                acc_in_bin = accuracies[in_bin].float().mean()
                conf_in_bin = confidences[in_bin].mean()
                gap = torch.abs(acc_in_bin - conf_in_bin)
                ece += gap * (in_bin.sum().float() / n_samples)
                mce = torch.maximum(mce, gap)

        # Brier Score: 1/N * sum (p_i - y_i)^2
        one_hot = torch.zeros_like(probs).scatter_(-1, targets.unsqueeze(-1), 1.0)
        brier = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))

        return float(ece.item()), float(mce.item()), float(brier.item())


class ConformalTacticPredictor:
    """Split conformal prediction for tactical actions with formal coverage guarantees."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Desired error rate (1 - coverage)
        self.quantile_threshold: float = 1.0

    def calibrate(self, val_probs: torch.Tensor, val_targets: torch.Tensor) -> float:
        """Compute non-conformity quantile on calibration set."""
        n = val_probs.shape[0]
        # Non-conformity score: 1.0 - true_class_probability
        true_probs = val_probs[torch.arange(n), val_targets]
        scores = 1.0 - true_probs

        # Finite-sample adjusted quantile: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile_threshold = float(torch.quantile(scores, q_level, interpolation="higher").item())
        return self.quantile_threshold

    def predict_set(self, test_probs: torch.Tensor) -> list[list[int]]:
        """Generate prediction sets for candidate tactic choices ensuring coverage guarantee."""
        sets: list[list[int]] = []
        # Score for any class k is 1.0 - p_k <= threshold  <=>  p_k >= 1.0 - threshold
        min_p = max(0.0, 1.0 - self.quantile_threshold)

        for i in range(test_probs.shape[0]):
            cands = torch.nonzero(test_probs[i] >= min_p).squeeze(-1).tolist()
            if isinstance(cands, int):
                cands = [cands]
            elif not cands:
                # If empty set, include argmax as standard fallback
                cands = [int(torch.argmax(test_probs[i]).item())]
            sets.append(cands)

        return sets
