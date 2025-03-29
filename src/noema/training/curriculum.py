"""Automated curriculum learning, bandit scheduling, and competence pacing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence
from noema.contracts.theorems import TheoremSpec


@dataclass
class DifficultyBand:
    """A difficulty partition defined by syntactic proof depth and premise count."""
    band_id: str
    min_depth: int
    max_depth: int
    sampling_weight: float = 1.0
    theorems: list[TheoremSpec] = field(default_factory=list)
    success_history: list[bool] = field(default_factory=list)


class CurriculumScheduler:
    """Bandit-guided curriculum scheduler optimizing task presentation order."""

    def __init__(
        self,
        bands: Sequence[DifficultyBand] | None = None,
        target_success_rate: float = 0.5,
        exploration_rate: float = 0.15,
        window_size: int = 50,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.target_success_rate = target_success_rate
        self.exploration_rate = exploration_rate
        self.window_size = window_size

        if bands is not None:
            self.bands = list(bands)
        else:
            self.bands = [
                DifficultyBand(band_id="elementary", min_depth=1, max_depth=2, sampling_weight=1.0),
                DifficultyBand(band_id="intermediate", min_depth=3, max_depth=5, sampling_weight=0.5),
                DifficultyBand(band_id="advanced", min_depth=6, max_depth=15, sampling_weight=0.2),
            ]

    def register_theorem(self, theorem: TheoremSpec, proof_depth: int) -> None:
        """Assign a theorem to its appropriate difficulty band."""
        for band in self.bands:
            if band.min_depth <= proof_depth <= band.max_depth:
                band.theorems.append(theorem)
                return
        # If deeper than all, assign to hardest band
        self.bands[-1].theorems.append(theorem)

    def record_outcome(self, band_id: str, success: bool) -> None:
        """Record the proof search outcome and update band sampling weights via Exp3."""
        for band in self.bands:
            if band.band_id == band_id:
                band.success_history.append(success)
                if len(band.success_history) > self.window_size:
                    band.success_history.pop(0)
                break
        self._update_weights()

    def _update_weights(self) -> None:
        """Update sampling weights using distance to optimal learning zone (target rate)."""
        raw_weights = []
        for band in self.bands:
            if not band.theorems:
                raw_weights.append(0.0)
                continue

            if not band.success_history:
                # Prioritize unexplored bands
                rate = self.target_success_rate
            else:
                rate = sum(band.success_history) / len(band.success_history)

            # Zone of Proximal Development: highest reward when rate is close to target_success_rate
            # Gaussian reward curve centered at target_success_rate
            learning_value = math.exp(-((rate - self.target_success_rate) ** 2) / 0.08)
            raw_weights.append(learning_value)

        total = sum(raw_weights)
        if total <= 0:
            for band in self.bands:
                band.sampling_weight = 1.0 / len(self.bands)
            return

        # Blend with uniform exploration
        n = len(self.bands)
        for i, band in enumerate(self.bands):
            p_val = (1.0 - self.exploration_rate) * (raw_weights[i] / total) + (self.exploration_rate / n)
            band.sampling_weight = max(1e-4, p_val)

    def sample_theorem(self) -> tuple[TheoremSpec, str]:
        """Sample a theorem according to current bandit sampling weights."""
        valid_bands = [b for b in self.bands if b.theorems]
        if not valid_bands:
            raise ValueError("No theorems registered in curriculum scheduler")

        weights = [b.sampling_weight for b in valid_bands]
        chosen_band = self.rng.choices(valid_bands, weights=weights, k=1)[0]
        chosen_thm = self.rng.choice(chosen_band.theorems)
        return chosen_thm, chosen_band.band_id

    def get_progress_report(self) -> dict[str, dict[str, float]]:
        """Return curriculum status across all difficulty bands."""
        report = {}
        for b in self.bands:
            rate = (sum(b.success_history) / len(b.success_history)) if b.success_history else 0.0
            report[b.band_id] = {
                "theorem_count": float(len(b.theorems)),
                "success_rate": rate,
                "sampling_weight": b.sampling_weight,
            }
        return report
