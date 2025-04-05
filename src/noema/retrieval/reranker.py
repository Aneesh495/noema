"""Cross-encoder neural reranker with joint goal-premise cross-attention."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Sequence
from noema.contracts.environment import PremiseDecl
from noema.contracts.states import ProofGoal
from noema.tokenizer.math_tokenizer import MathTokenizer


@dataclass(frozen=True)
class ScoredPremise:
    """A candidate premise scored by cross-encoder joint relevance."""
    premise_name: str
    score: float
    statement: str
    rank: int


class PremiseCrossEncoder(nn.Module):
    """Deep cross-encoder model scoring joint (goal, premise) relevance."""

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute scalar relevance logit for concatenated goal-premise token sequence."""
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pooled representation (first token / CLS token representation)
        cls_rep = h[:, 0, :]
        logits = self.classifier(cls_rep).squeeze(-1)
        return logits


class PremiseReranker:
    """Manages cross-encoder scoring and reranking over candidate premises."""

    def __init__(
        self,
        model: PremiseCrossEncoder,
        tokenizer: MathTokenizer,
        device: torch.device | None = None,
        max_seq_len: int = 512,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def rerank(
        self,
        goal: ProofGoal,
        candidate_premises: Sequence[PremiseDecl],
        top_k: int = 10,
    ) -> list[ScoredPremise]:
        """Rerank candidate premises according to joint cross-attention relevance."""
        if not candidate_premises:
            return []

        self.model.eval()
        batch_tokens = []
        max_len = 0

        # Construct concatenated joint strings: "⊢ goal_target [SEP] premise_type"
        for p in candidate_premises:
            joint_text = f"⊢ {goal.target} [SEP] {p.name} : {p.type_str}"
            ids = self.tokenizer.encode(joint_text)
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len]
            batch_tokens.append(ids)
            max_len = max(max_len, len(ids))

        # Pad batch
        bsz = len(batch_tokens)
        input_ids = torch.full((bsz, max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros((bsz, max_len), dtype=torch.float32, device=self.device)

        for i, ids in enumerate(batch_tokens):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attn_mask[i, : len(ids)] = 1.0

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attn_mask)
            scores = torch.sigmoid(logits).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        scored: list[ScoredPremise] = []
        for i, p in enumerate(candidate_premises):
            scored.append(
                ScoredPremise(
                    premise_name=p.name,
                    score=float(scores[i]),
                    statement=p.type_str,
                    rank=0,
                )
            )

        # Sort by score descending
        scored.sort(key=lambda sp: sp.score, reverse=True)
        for rank, sp in enumerate(scored[:top_k], start=1):
            object.__setattr__(sp, "rank", rank)

        return scored[:top_k]
