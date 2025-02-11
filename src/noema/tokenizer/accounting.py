"""Explicit nonpadding token accounting and truncation tracking."""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field


class TokenAccountingSummary(BaseModel):
    """Auditable summary of token presentations during a training or evaluation run."""
    total_tokens_presented: int = 0
    nonpadding_tokens: int = 0
    padding_tokens: int = 0
    truncated_sequences: int = 0
    total_sequences: int = 0
    padding_fraction: float = 0.0
    truncation_fraction: float = 0.0


class TokenAccountant:
    """Accurate tracker guaranteeing padding tokens never enter completion or loss metrics."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        self.total_tokens_presented = 0
        self.nonpadding_tokens = 0
        self.padding_tokens = 0
        self.truncated_sequences = 0
        self.total_sequences = 0

    def record_batch(
        self,
        token_tensor: torch.Tensor,
        was_truncated: list[bool] | None = None
    ) -> int:
        """Record a batch of token tensors and return the nonpadding token count."""
        batch_size = token_tensor.size(0)
        total_in_batch = token_tensor.numel()
        
        nonpad_mask = (token_tensor != self.pad_token_id)
        nonpad_count = int(nonpad_mask.sum().item())
        pad_count = total_in_batch - nonpad_count

        self.total_sequences += batch_size
        self.total_tokens_presented += total_in_batch
        self.nonpadding_tokens += nonpad_count
        self.padding_tokens += pad_count

        if was_truncated:
            self.truncated_sequences += sum(1 for t in was_truncated if t)

        return nonpad_count

    def get_summary(self) -> TokenAccountingSummary:
        pad_frac = (
            float(self.padding_tokens) / self.total_tokens_presented
            if self.total_tokens_presented > 0
            else 0.0
        )
        trunc_frac = (
            float(self.truncated_sequences) / self.total_sequences
            if self.total_sequences > 0
            else 0.0
        )
        return TokenAccountingSummary(
            total_tokens_presented=self.total_tokens_presented,
            nonpadding_tokens=self.nonpadding_tokens,
            padding_tokens=self.padding_tokens,
            truncated_sequences=self.truncated_sequences,
            total_sequences=self.total_sequences,
            padding_fraction=pad_frac,
            truncation_fraction=trunc_frac,
        )

    def reset(self) -> None:
        self.total_tokens_presented = 0
        self.nonpadding_tokens = 0
        self.padding_tokens = 0
        self.truncated_sequences = 0
        self.total_sequences = 0
