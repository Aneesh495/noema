"""Tests for automated curriculum learning and difficulty bandit scheduling."""

import pytest
from noema.contracts.theorems import TheoremSpec
from noema.training.curriculum import CurriculumScheduler, DifficultyBand


def make_dummy_theorem(tid: str) -> TheoremSpec:
    return TheoremSpec(
        theorem_id=tid,
        name=f"thm_{tid}",
        module="Test.Basic",
        statement="∀ (x : Nat), x = x",
    )


def test_curriculum_registration_and_sampling():
    scheduler = CurriculumScheduler(seed=42)

    # Register 3 theorems at depth 1, 4, and 8
    t1 = make_dummy_theorem("t1")
    t2 = make_dummy_theorem("t2")
    t3 = make_dummy_theorem("t3")

    scheduler.register_theorem(t1, proof_depth=1)
    scheduler.register_theorem(t2, proof_depth=4)
    scheduler.register_theorem(t3, proof_depth=8)

    assert len(scheduler.bands[0].theorems) == 1
    assert len(scheduler.bands[1].theorems) == 1
    assert len(scheduler.bands[2].theorems) == 1

    # Sample a theorem
    sampled_thm, band_id = scheduler.sample_theorem()
    assert sampled_thm.theorem_id in {"t1", "t2", "t3"}
    assert band_id in {"elementary", "intermediate", "advanced"}


def test_curriculum_adaptive_weights_on_failures():
    scheduler = CurriculumScheduler(target_success_rate=0.5, seed=42)

    t1 = make_dummy_theorem("t1")
    t2 = make_dummy_theorem("t2")
    scheduler.register_theorem(t1, proof_depth=1)
    scheduler.register_theorem(t2, proof_depth=4)

    # Elementary band is 100% successful (too easy!)
    for _ in range(20):
        scheduler.record_outcome("elementary", success=True)

    # Intermediate band is 50% successful (optimal challenge!)
    for i in range(20):
        scheduler.record_outcome("intermediate", success=(i % 2 == 0))

    report = scheduler.get_progress_report()
    # The intermediate band should have higher sampling weight than the solved elementary band
    assert report["intermediate"]["sampling_weight"] > report["elementary"]["sampling_weight"]
    assert report["elementary"]["success_rate"] == 1.0
    assert report["intermediate"]["success_rate"] == 0.5
