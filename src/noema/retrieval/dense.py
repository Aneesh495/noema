"""Learned dense dual-encoder premise retriever."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from noema.contracts.environment import PremiseDecl
from noema.contracts.states import ProofGoal
from noema.tokenizer.math_tokenizer import MathTokenizer


class DenseDualEncoder(nn.Module):
    """Dual-encoder neural network mapping proof goals and premise declarations to a joint metric space."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.temperature = temperature

        # Token embedding shared or separate
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        # Goal encoder projection
        self.goal_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Premise encoder projection
        self.premise_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def encode_goal(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode proof goal sequence into a normalized d-dimensional vector."""
        emb = self.embedding(input_ids) # [B, T, H]
        # Mean pooling over nonpadding tokens
        mask = attention_mask.unsqueeze(-1).float() # [B, T, 1]
        sum_emb = (emb * mask).sum(dim=1) # [B, H]
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        pooled = sum_emb / lengths
        vec = self.goal_encoder(pooled)
        return F.normalize(vec, p=2, dim=-1)

    def encode_premise(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode premise declaration into a normalized d-dimensional vector."""
        emb = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        sum_emb = (emb * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        pooled = sum_emb / lengths
        vec = self.premise_encoder(pooled)
        return F.normalize(vec, p=2, dim=-1)

    def compute_similarity(
        self,
        goal_vectors: torch.Tensor,
        premise_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity matrix scaled by temperature."""
        # [B_goal, D] @ [B_prem, D].T -> [B_goal, B_prem]
        return torch.matmul(goal_vectors, premise_vectors.t()) / self.temperature

    def contrastive_loss(
        self,
        goal_vectors: torch.Tensor,
        positive_premise_vectors: torch.Tensor,
        negative_premise_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss over positive and negative premises."""
        batch_size = goal_vectors.size(0)
        # Dot product with positive premise: [B]
        pos_sim = (goal_vectors * positive_premise_vectors).sum(dim=-1, keepdim=True) / self.temperature

        if negative_premise_vectors is not None:
            # negative_premise_vectors: [B, K, D]
            # neg_sim: [B, K]
            neg_sim = torch.bmm(negative_premise_vectors, goal_vectors.unsqueeze(-1)).squeeze(-1) / self.temperature
            logits = torch.cat([pos_sim, neg_sim], dim=1) # [B, 1 + K]
        else:
            # In-batch negatives
            all_sim = self.compute_similarity(goal_vectors, positive_premise_vectors)
            logits = all_sim

        labels = torch.zeros(batch_size, dtype=torch.long, device=goal_vectors.device)
        return F.cross_entropy(logits, labels)
