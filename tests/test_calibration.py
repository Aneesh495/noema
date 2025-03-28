"""Tests for uncertainty quantification, ECE calibration, and conformal prediction sets."""

import pytest
import torch
from noema.models.calibration import ConformalTacticPredictor, TemperatureCalibrator


def test_compute_ece_perfect_calibration():
    # Exactly confident with 100% accuracy -> ECE = 0.0
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1, 0, 1])

    ece, mce, brier = TemperatureCalibrator.compute_ece(probs, targets, num_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-5)
    assert mce == pytest.approx(0.0, abs=1e-5)
    assert brier == pytest.approx(0.0, abs=1e-5)


def test_temperature_scaling_optimization():
    torch.manual_seed(42)
    calibrator = TemperatureCalibrator(init_temperature=1.0)

    # Create overconfident synthetic logits
    logits = torch.randn(200, 5) * 5.0
    targets = torch.randint(0, 5, (200,))

    orig_probs = torch.softmax(logits, dim=-1)
    orig_ece, _, _ = TemperatureCalibrator.compute_ece(orig_probs, targets)

    # Fit temperature
    opt_temp = calibrator.fit(logits, targets, lr=0.05, max_iters=50)
    assert opt_temp > 0.0

    scaled_probs = torch.softmax(calibrator(logits), dim=-1)
    calibrated_ece, _, _ = TemperatureCalibrator.compute_ece(scaled_probs, targets)

    # Calibration should improve ECE on overconfident data
    assert calibrated_ece <= orig_ece + 1e-4


def test_conformal_prediction_coverage():
    torch.manual_seed(123)
    # Generate 500 calibration samples
    val_probs = torch.softmax(torch.randn(500, 10), dim=-1)
    val_targets = torch.randint(0, 10, (500,))

    predictor = ConformalTacticPredictor(alpha=0.10)  # 90% target coverage
    threshold = predictor.calibrate(val_probs, val_targets)
    assert 0.0 <= threshold <= 1.0

    # Test on 200 held-out samples
    test_probs = torch.softmax(torch.randn(200, 10), dim=-1)
    test_targets = torch.randint(0, 10, (200,))

    pred_sets = predictor.predict_set(test_probs)
    assert len(pred_sets) == 200

    # Calculate empirical coverage
    covered = 0
    for i, target in enumerate(test_targets.tolist()):
        if target in pred_sets[i]:
            covered += 1

    empirical_coverage = covered / 200.0
    # Expected around 90% (allow standard binomial sampling margin)
    assert empirical_coverage >= 0.80
