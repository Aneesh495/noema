"""Supervised training loop with joint loss, per-block gradient tracking, and token accounting."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Generator
import torch
import torch.nn as nn
from pydantic import BaseModel
from noema.models.transformer import NoemaTransformer
from noema.tokenizer.accounting import TokenAccountant, TokenAccountingSummary
from noema.training.checkpoints import CheckpointManager
from noema.training.loss import JointLoss


class TrainingConfig(BaseModel):
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    total_steps: int = 500
    warmup_steps: int = 50
    seed: int = 42
    log_interval: int = 25
    save_interval: int = 100
    target_nonpadding_tokens: int = 15_000_000


class NoemaTrainer:
    """Trainer orchestrating joint policy, premise pointer, and censored value optimization."""

    def __init__(
        self,
        model: NoemaTransformer,
        config: TrainingConfig,
        checkpoint_dir: Path | str = "checkpoints",
        device: torch.device | str = "cpu",
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)

        self.loss_fn = JointLoss()
        self.accountant = TokenAccountant(pad_token_id=0)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        # Optimizer with weight decay separation
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name.lower() or "bias" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
        )

        self.step_count = 0
        self.best_loss = float("inf")
        self.history: list[dict[str, Any]] = []

    def get_lr(self, step: int) -> float:
        """Linear warmup followed by cosine decay."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * float(step + 1) / float(self.config.warmup_steps)
        progress = float(step - self.config.warmup_steps) / float(
            max(1, self.config.total_steps - self.config.warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))
        return self.config.min_learning_rate + (self.config.learning_rate - self.config.min_learning_rate) * float(cosine_decay)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        tactic_targets = batch["tactic_class_targets"].to(self.device)
        value_targets = batch["value_targets"].to(self.device)
        cand_prem = batch.get("candidate_premise_vectors")
        cand_mask = batch.get("candidate_mask")
        if cand_prem is not None:
            cand_prem = cand_prem.to(self.device)
        if cand_mask is not None:
            cand_mask = cand_mask.to(self.device)

        # Record nonpadding tokens
        nonpad_in_batch = self.accountant.record_batch(input_ids)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidate_premise_vectors=cand_prem,
            candidate_mask=cand_mask,
        )

        losses = self.loss_fn(
            tactic_class_logits=outputs["tactic_class_logits"],
            tactic_class_targets=tactic_targets,
            value_pred=outputs["value"],
            value_targets=value_targets,
            premise_scores=outputs.get("premise_scores"),
            premise_targets=batch.get("premise_targets"),
        )

        total_loss = losses["total_loss"]
        total_loss.backward()

        # Track per-block gradients to verify active parameter updates across all 24 blocks
        block_grads = []
        for b_idx, block in enumerate(self.model.blocks):
            g_norms = [p.grad.norm().item() for p in block.parameters() if p.grad is not None]
            avg_g = sum(g_norms) / len(g_norms) if g_norms else 0.0
            block_grads.append(avg_g)

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        # Update learning rate
        current_lr = self.get_lr(self.step_count)
        for pg in self.optimizer.param_groups:
            pg["lr"] = current_lr

        self.optimizer.step()
        self.step_count += 1

        record = {
            "step": self.step_count,
            "total_loss": float(total_loss.item()),
            "policy_loss": float(losses["policy_loss"].item()),
            "value_loss": float(losses["value_loss"].item()),
            "grad_norm": float(grad_norm.item()),
            "lr": current_lr,
            "nonpadding_tokens": self.accountant.nonpadding_tokens,
            "min_block_grad": min(block_grads) if block_grads else 0.0,
            "max_block_grad": max(block_grads) if block_grads else 0.0,
        }
        self.history.append(record)

        if total_loss.item() < self.best_loss:
            self.best_loss = total_loss.item()
            self.checkpoint_manager.save_checkpoint(
                "best.pt",
                self.model,
                self.optimizer,
                step=self.step_count,
                nonpadding_tokens=self.accountant.nonpadding_tokens,
                best_loss=self.best_loss,
                seed=self.config.seed,
            )

        if self.step_count % self.config.save_interval == 0:
            self.checkpoint_manager.save_checkpoint(
                f"checkpoint_step_{self.step_count}.pt",
                self.model,
                self.optimizer,
                step=self.step_count,
                nonpadding_tokens=self.accountant.nonpadding_tokens,
                best_loss=self.best_loss,
                seed=self.config.seed,
            )

        return record
