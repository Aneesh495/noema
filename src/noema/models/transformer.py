"""Original 24-block flagship Transformer model built strictly from tensor/nn primitives."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from noema.models.config import TransformerConfig
from noema.models.heads import PolicyTacticHead, PremisePointerHead, ProgressValueHead
from noema.models.layers import RMSNorm, TransformerBlock


class NoemaTransformer(nn.Module):
    """The deep neural-network flagship: 24 distinct independently parameterized Transformer blocks."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=0,
        )

        # 24 independently instantiated Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                block_idx=i,
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                norm_eps=config.norm_eps,
            )
            for i in range(config.n_layers)
        ])

        # Final normalization before readout heads
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Readout heads
        self.policy_head = PolicyTacticHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            num_tactic_classes=config.num_tactic_classes,
        )
        self.pointer_head = PremisePointerHead(d_model=config.d_model)
        self.value_head = ProgressValueHead(d_model=config.d_model)

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def block_parameters(self) -> int:
        """Count parameters within the 24 Transformer blocks alone."""
        return sum(p.numel() for b in self.blocks for p in b.parameters() if p.requires_grad)

    def verify_distinct_blocks(self) -> tuple[bool, str]:
        """Verify that all blocks have distinct independent parameters (no weight sharing)."""
        data_ptrs = set()
        for i, block in enumerate(self.blocks):
            for name, param in block.named_parameters():
                ptr = param.data_ptr()
                if ptr in data_ptrs:
                    return False, f"Shared parameter tensor detected in block {i}: {name}"
                data_ptrs.add(ptr)

        num_params = self.num_parameters
        if self.config.n_layers == 24 and num_params < 40_000_000:
            return False, f"Flagship parameter count {num_params} is below the 40M threshold"

        return True, f"Verified {len(self.blocks)} distinct blocks with {num_params:,} parameters."

    def pool_representation(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean pool representation across nonpadding tokens."""
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        candidate_premise_vectors: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[float]]:
        # 1. Embed tokens
        x = self.token_embeddings(input_ids) # [B, T, D]

        activation_norms = []

        # 2. Sequential block execution
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to conserve memory
                x = checkpoint(block, x, attention_mask, is_causal, use_reentrant=False)
            else:
                x = block(x, attention_mask=attention_mask, is_causal=is_causal)
            activation_norms.append(float(x.detach().norm().item()))

        # 3. Final Pre-Norm
        x = self.final_norm(x)

        # 4. Pooling
        pooled = self.pool_representation(x, attention_mask=attention_mask)

        # 5. Readout predictions
        tactic_class_logits, token_logits = self.policy_head(x, pooled)
        value_pred = self.value_head(pooled)

        premise_scores = None
        if candidate_premise_vectors is not None:
            premise_scores = self.pointer_head(
                goal_vector=pooled,
                candidate_premise_vectors=candidate_premise_vectors,
                candidate_mask=candidate_mask,
            )

        return {
            "token_logits": token_logits,
            "tactic_class_logits": tactic_class_logits,
            "value": value_pred,
            "pooled": pooled,
            "hidden_states": x,
            "premise_scores": premise_scores,
            "activation_norms": activation_norms,
        }
