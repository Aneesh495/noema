"""Configuration dataclasses for Noema Transformer models and depth ablations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TransformerConfig(BaseModel):
    """Specification of model hyper-parameters."""
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 24
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 512
    dropout: float = 0.1
    norm_eps: float = 1e-6
    rotary_dim: int = 64
    use_gradient_checkpointing: bool = False
    num_tactic_classes: int = 32

    @classmethod
    def flagship_24block(cls) -> TransformerConfig:
        """24-block flagship model (~44M parameters)."""
        return cls(
            d_model=384,
            n_layers=24,
            n_heads=6,
            d_ff=1536,
        )

    @classmethod
    def ablation_12block(cls) -> TransformerConfig:
        """12-block ablation model (~22M parameters)."""
        return cls(
            d_model=384,
            n_layers=12,
            n_heads=6,
            d_ff=1536,
        )

    @classmethod
    def ablation_4block(cls) -> TransformerConfig:
        """4-block shallow baseline model (~7.5M parameters)."""
        return cls(
            d_model=384,
            n_layers=4,
            n_heads=6,
            d_ff=1536,
        )
