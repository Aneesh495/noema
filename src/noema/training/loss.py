"""Joint policy, premise pointer, and progress value loss with censored failure handling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLoss(nn.Module):
    """Multi-task objective combining tactic policy prediction, premise selection, and censored value estimation."""

    def __init__(
        self,
        lambda_policy: float = 1.0,
        lambda_pointer: float = 0.5,
        lambda_value: float = 0.5,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.lambda_policy = lambda_policy
        self.lambda_pointer = lambda_pointer
        self.lambda_value = lambda_value
        self.pad_token_id = pad_token_id

    def forward(
        self,
        tactic_class_logits: torch.Tensor,
        tactic_class_targets: torch.Tensor,
        value_pred: torch.Tensor,
        value_targets: torch.Tensor,
        token_logits: torch.Tensor | None = None,
        token_targets: torch.Tensor | None = None,
        premise_scores: torch.Tensor | None = None,
        premise_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute joint weighted loss components.
        Padding tokens in token_targets are ignored in cross-entropy.
        """
        # 1. Tactic class loss (cross entropy)
        class_loss = F.cross_entropy(tactic_class_logits, tactic_class_targets)

        # 2. Token-level policy loss if provided (ignoring pad_token_id)
        token_loss = torch.tensor(0.0, device=tactic_class_logits.device)
        if token_logits is not None and token_targets is not None:
            # Flatten [B, T, V] -> [B*T, V] and [B, T] -> [B*T]
            B, T, V = token_logits.shape
            token_loss = F.cross_entropy(
                token_logits.view(B * T, V),
                token_targets.view(B * T),
                ignore_index=self.pad_token_id,
            )

        policy_loss = class_loss + (0.5 * token_loss if token_logits is not None else 0.0)

        # 3. Value progress loss with censored outcome targets (smooth L1 or BCE)
        # value_targets in [0, 1] represent solve probability / progress under budget
        val_targets = value_targets.view_as(value_pred)
        value_loss = F.binary_cross_entropy(value_pred, val_targets)

        # 4. Premise pointer loss if provided
        pointer_loss = torch.tensor(0.0, device=tactic_class_logits.device)
        if premise_scores is not None and premise_targets is not None:
            # premise_scores: [B, K], premise_targets: [B]
            pointer_loss = F.cross_entropy(premise_scores, premise_targets)

        total_loss = (
            self.lambda_policy * policy_loss
            + self.lambda_value * value_loss
            + self.lambda_pointer * pointer_loss
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "class_loss": class_loss,
            "token_loss": token_loss,
            "value_loss": value_loss,
            "pointer_loss": pointer_loss,
        }
