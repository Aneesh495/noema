"""Unit tests for MathTokenizer, TokenAccountant, and BatchCollator."""

import tempfile
import torch
import pytest
from noema.contracts.states import LocalHypothesis, ProofGoal
from noema.tokenizer import (
    MathTokenizer,
    TokenAccountant,
    BatchCollator,
)


def test_math_tokenizer_special_symbols():
    tok = MathTokenizer()
    text = "∀ (x y : Nat), x + y = y + x := by rfl"
    tokens = tok.tokenize_text(text)
    
    assert "∀" in tokens
    assert ":=" in tokens
    assert "=" in tokens
    assert "+" in tokens
    assert "rfl" in tokens


def test_tokenizer_training_and_serialization():
    train_corpus = [
        "∀ (n : Nat), n + 0 = n",
        "∀ (p q : Prop), p ∧ q → q ∧ p",
        "theorem test : 1 + 1 = 2 := by rfl",
    ]
    tok = MathTokenizer()
    tok.train_from_corpus(train_corpus, max_vocab_size=200)
    
    encoded = tok.encode("∀ (n : Nat), n + 0 = n")
    assert encoded[0] == tok.bos_token_id
    assert encoded[-1] == tok.eos_token_id
    
    decoded = tok.decode(encoded)
    assert "∀" in decoded
    assert "Nat" in decoded

    # Save and load
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        tok.save(f.name)
        loaded = MathTokenizer.load(f.name)
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.encode("n + 0 = n") == tok.encode("n + 0 = n")


def test_token_accountant_guarantees():
    accountant = TokenAccountant(pad_token_id=0)
    
    # Batch with 2 sequences: [1, 2, 3, 0, 0] (3 nonpad) and [4, 5, 0, 0, 0] (2 nonpad)
    batch = torch.tensor([
        [1, 2, 3, 0, 0],
        [4, 5, 0, 0, 0],
    ], dtype=torch.long)
    
    nonpad = accountant.record_batch(batch, was_truncated=[False, True])
    assert nonpad == 5
    
    summary = accountant.get_summary()
    assert summary.nonpadding_tokens == 5
    assert summary.padding_tokens == 5
    assert summary.total_tokens_presented == 10
    assert summary.truncated_sequences == 1
    assert summary.total_sequences == 2
    assert summary.padding_fraction == 0.5
    assert summary.truncation_fraction == 0.5


def test_batch_collator_attention_masks():
    tok = MathTokenizer()
    train_corpus = ["n : Nat ⊢ n = n", "p : Prop q : Prop ⊢ p ∧ q"]
    tok.train_from_corpus(train_corpus)
    
    collator = BatchCollator(tokenizer=tok, max_seq_len=64)
    
    g1 = ProofGoal(
        goal_id="g1",
        target="n = n",
        hypotheses=[LocalHypothesis(name="n", type_str="Nat")],
        mvar_id="?m.1",
    )
    g2 = ProofGoal(
        goal_id="g2",
        target="p ∧ q",
        hypotheses=[
            LocalHypothesis(name="p", type_str="Prop"),
            LocalHypothesis(name="q", type_str="Prop"),
        ],
        mvar_id="?m.2",
    )
    
    batch = collator.collate_goals([g1, g2])
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert input_ids.shape == attention_mask.shape
    # Check that where mask is 0, input_ids is pad_token_id
    pad_positions = (attention_mask == False)
    if pad_positions.any():
        assert (input_ids[pad_positions] == tok.pad_token_id).all()
