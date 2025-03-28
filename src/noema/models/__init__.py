"""Noema deep neural Transformer models, layers, and readout heads."""

from noema.models.attention import GroupedQueryAttention, KVCache
from noema.models.config import TransformerConfig
from noema.models.heads import (
    PolicyTacticHead,
    PremisePointerHead,
    ProgressValueHead,
)
from noema.models.layers import (
    MultiHeadAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFeedForward,
    TransformerBlock,
)
from noema.models.calibration import (
    CalibrationMetrics,
    ConformalTacticPredictor,
    TemperatureCalibrator,
)
from noema.models.transformer import NoemaTransformer

__all__ = [
    "TransformerConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "AdvancedRotaryEmbedding",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "KVCache",
    "SwiGLUFeedForward",
    "TransformerBlock",
    "PolicyTacticHead",
    "PremisePointerHead",
    "ProgressValueHead",
    "NoemaTransformer",
    "TemperatureCalibrator",
    "ConformalTacticPredictor",
    "CalibrationMetrics",
]
