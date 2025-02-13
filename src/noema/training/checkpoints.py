"""Atomic model, optimizer, scheduler, and RNG checkpointing with interruption recovery."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from pydantic import BaseModel


class CheckpointMetadata(BaseModel):
    step: int
    epoch: int
    nonpadding_tokens: int
    best_loss: float
    seed: int
    timestamp_utc: str
    model_config_dict: dict[str, Any]


class CheckpointManager:
    """Manages atomic saving and deterministic resumption of training states."""

    def __init__(self, checkpoint_dir: Path | str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        filename: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        step: int = 0,
        epoch: int = 0,
        nonpadding_tokens: int = 0,
        best_loss: float = float("inf"),
        seed: int = 42,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically save checkpoint to disk."""
        target_path = self.checkpoint_dir / filename

        # Collect RNG states
        rng_state = {
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        meta = {
            "step": step,
            "epoch": epoch,
            "nonpadding_tokens": nonpadding_tokens,
            "best_loss": best_loss,
            "seed": seed,
            "extra": extra_meta or {},
        }

        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "rng_state": rng_state,
            "metadata": meta,
        }

        # Write to temporary file in same directory for atomic replace
        with tempfile.NamedTemporaryFile("wb", dir=self.checkpoint_dir, delete=False) as f:
            temp_path = Path(f.name)
            torch.save(payload, temp_path)

        # Atomic replacement
        os.replace(temp_path, target_path)
        return target_path

    def load_checkpoint(
        self,
        filename: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Load and restore model, optimizer, scheduler, and RNG state from checkpoint."""
        target_path = self.checkpoint_dir / filename
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {target_path}")

        payload = torch.load(target_path, map_location=device, weights_only=False)

        # Restore model
        model.load_state_dict(payload["model_state_dict"])

        # Restore optimizer if provided
        if optimizer and payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        # Restore scheduler if provided
        if scheduler and payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        # Restore RNG states
        rng = payload.get("rng_state")
        if rng:
            if "python_rng" in rng:
                random.setstate(rng["python_rng"])
            if "torch_rng" in rng and rng["torch_rng"] is not None:
                torch.set_rng_state(rng["torch_rng"])
            if "torch_cuda_rng" in rng and rng["torch_cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda_rng"])

        return payload.get("metadata", {})
