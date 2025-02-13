"""Policy, Premise Pointer, and Progress/Value readout heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyTacticHead(nn.Module):
    """Generates tactical action predictions: tactic kind classification and token vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int, num_tactic_classes: int = 32):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # High-level tactic kind classification (e.g. rfl, intro, apply, rw)
        self.tactic_class_head = nn.Linear(d_model, num_tactic_classes)

        # Autoregressive next-token prediction head
        self.token_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooled_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [B, T, D] -> token_logits: [B, T, V]
        token_logits = self.token_head(hidden_states)

        # pooled_states: [B, D] -> class_logits: [B, C]
        class_logits = self.tactic_class_head(pooled_states)

        return class_logits, token_logits


class PremisePointerHead(nn.Module):
    """Scores candidate premises using attention dot-product between goal and premise vectors."""

    def __init__(self, d_model: int, projection_dim: int = 256):
        super().__init__()
        self.goal_proj = nn.Linear(d_model, projection_dim)
        self.premise_proj = nn.Linear(d_model, projection_dim)
        self.scale = 1.0 / (projection_dim ** 0.5)

    def forward(
        self,
        goal_vector: torch.Tensor,
        candidate_premise_vectors: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        goal_vector: [B, D]
        candidate_premise_vectors: [B, K, D]
        candidate_mask: [B, K] where 1 is eligible candidate, 0 is masked/ineligible
        """
        g_proj = self.goal_proj(goal_vector).unsqueeze(1) # [B, 1, P]
        p_proj = self.premise_proj(candidate_premise_vectors) # [B, K, P]

        # Dot-product similarity: [B, K]
        scores = torch.bmm(g_proj, p_proj.transpose(1, 2)).squeeze(1) * self.scale

        if candidate_mask is not None:
            # Mask out ineligible premise slots
            scores = scores.masked_fill(candidate_mask == 0, float("-inf"))

        return scores


class ProgressValueHead(nn.Module):
    """Estimates expected proof progress or solve likelihood v in [0, 1]."""

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pooled_states: torch.Tensor) -> torch.Tensor:
        # pooled_states: [B, D] -> value: [B, 1] in range (0, 1)
        logits = self.mlp(pooled_states)
        return torch.sigmoid(logits)
