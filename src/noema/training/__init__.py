"""Supervised training, joint loss, atomic checkpointing, and expert iteration."""

from noema.training.checkpoints import CheckpointManager, CheckpointMetadata
from noema.training.curriculum import CurriculumScheduler, DifficultyBand
from noema.training.distillation import (
    SelfImitationBuffer,
    SimplifiedTrace,
    TraceSimplifier,
)
from noema.training.distributed import (
    DynamicLossScaler,
    ExponentialMovingAverage,
    GradientAccumulator,
)
from noema.training.expert_iteration import (
    ExpertIterationEngine,
    ExpertIterationSummary,
)
from noema.training.loss import JointLoss
from noema.training.ppo import (
    PPOBatch,
    PPOCriterion,
    PPOLossResult,
    PPOReplayBuffer,
    PPOTrajectoryStep,
    compute_generalized_advantages,
)
from noema.training.self_play import (
    ConjectureMutator,
    CurriculumSelfPlayEngine,
    RegretMatchingPolicy,
    SelfPlayProofTask,
)
from noema.training.trainer import NoemaTrainer, TrainingConfig

__all__ = [
    "JointLoss",
    "CheckpointManager",
    "CheckpointMetadata",
    "NoemaTrainer",
    "TrainingConfig",
    "ExpertIterationEngine",
    "ExpertIterationSummary",
    "CurriculumScheduler",
    "DifficultyBand",
    "GradientAccumulator",
    "DynamicLossScaler",
    "ExponentialMovingAverage",
    "TraceSimplifier",
    "SelfImitationBuffer",
    "SimplifiedTrace",
    "PPOTrajectoryStep",
    "PPOBatch",
    "PPOCriterion",
    "PPOLossResult",
    "PPOReplayBuffer",
    "compute_generalized_advantages",
    "ConjectureMutator",
    "RegretMatchingPolicy",
    "CurriculumSelfPlayEngine",
    "SelfPlayProofTask",
]
