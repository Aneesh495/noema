"""Tests for cross-encoder premise reranking with joint attention."""

import pytest
import torch
from noema.contracts.environment import PremiseDecl
from noema.contracts.states import ProofGoal
from noema.retrieval.reranker import PremiseCrossEncoder, PremiseReranker
from noema.tokenizer.math_tokenizer import MathTokenizer


@pytest.fixture
def tokenizer() -> MathTokenizer:
    tok = MathTokenizer()
    tok.train_from_corpus([
        "∀ (x y : Nat), x + y = y + x",
        "Nat.add_comm : ∀ (n m : Nat), n + m = m + n",
        "Nat.zero_add : ∀ (n : Nat), 0 + n = n",
        "Nat.mul_comm : ∀ (n m : Nat), n * m = m * n",
    ])
    return tok


def test_cross_encoder_forward_pass():
    model = PremiseCrossEncoder(
        vocab_size=100,
        d_model=32,
        num_layers=2,
        num_heads=2,
        d_ff=64,
        max_seq_len=64,
    )
    model.eval()

    input_ids = torch.randint(0, 100, (2, 16))
    attn_mask = torch.ones(2, 16)
    logits = model(input_ids, attention_mask=attn_mask)

    assert logits.shape == (2,)


def test_premise_reranker_ranking(tokenizer):
    model = PremiseCrossEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_layers=2,
        num_heads=2,
        d_ff=64,
        max_seq_len=128,
    )
    reranker = PremiseReranker(model=model, tokenizer=tokenizer)

    goal = ProofGoal(
        goal_id="g1",
        mvar_id="?m1",
        target="x + y = y + x",
        hypotheses=[],
    )

    candidates = [
        PremiseDecl(
            name="Nat.add_comm",
            module="Init.Data.Nat.Basic",
            type_str="∀ (n m : Nat), n + m = m + n",
        ),
        PremiseDecl(
            name="Nat.zero_add",
            module="Init.Data.Nat.Basic",
            type_str="∀ (n : Nat), 0 + n = n",
        ),
    ]

    results = reranker.rerank(goal, candidates, top_k=2)
    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert 0.0 <= results[0].score <= 1.0
    assert results[0].score >= results[1].score


def test_premise_reranker_empty_candidates(tokenizer):
    model = PremiseCrossEncoder(vocab_size=tokenizer.vocab_size, d_model=16, num_layers=1, num_heads=2)
    reranker = PremiseReranker(model=model, tokenizer=tokenizer)
    goal = ProofGoal(goal_id="g1", mvar_id="?m1", target="x = x", hypotheses=[])
    res = reranker.rerank(goal, [])
    assert res == []
