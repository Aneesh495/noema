"""Batch collation and attention mask construction for proof states and actions."""

from __future__ import annotations

import torch
from noema.contracts.actions import TacticAction
from noema.contracts.states import ProofGoal
from noema.tokenizer.math_tokenizer import MathTokenizer


class BatchCollator:
    """Collates proof goals and tactical actions into padded PyTorch tensors with explicit attention masks."""

    def __init__(self, tokenizer: MathTokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def collate_goals(self, goals: list[ProofGoal]) -> dict[str, torch.Tensor | list[bool]]:
        """Collate a list of ProofGoal objects into padded input tensors."""
        texts = []
        for g in goals:
            # Structure: <goal> <hyp> h1 <hyp> h2 <target> tgt
            hyps_str = " ".join(f"<hyp> {h.canonical_repr()}" for h in g.hypotheses)
            full_str = f"<goal> {hyps_str} <target> {g.target}"
            texts.append(full_str)

        return self.collate_texts(texts)

    def collate_texts(self, texts: list[str]) -> dict[str, torch.Tensor | list[bool]]:
        """Encode and pad an arbitrary list of strings."""
        batch_ids = []
        truncations = []

        for txt in texts:
            encoded = self.tokenizer.encode(txt, add_bos=True, add_eos=True)
            was_trunc = len(encoded) > self.max_seq_len
            truncations.append(was_trunc)
            if was_trunc:
                encoded = encoded[:self.max_seq_len - 1] + [self.tokenizer.eos_token_id]
            batch_ids.append(encoded)

        max_len = max(len(b) for b in batch_ids) if batch_ids else 0
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []

        for b in batch_ids:
            pad_len = max_len - len(b)
            padded = b + [pad_id] * pad_len
            mask = [1] * len(b) + [0] * pad_len
            input_ids.append(padded)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "was_truncated": truncations,
        }
