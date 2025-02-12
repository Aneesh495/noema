"""Unit tests for lexical and learned premise retrieval."""

import torch
import pytest
from noema.contracts.environment import EnvironmentSpec, PremiseDecl
from noema.contracts.states import LocalHypothesis, ProofGoal
from noema.retrieval import (
    LexicalRetriever,
    DenseDualEncoder,
    HardNegativeSampler,
    PremiseIndex,
)
from noema.tokenizer import MathTokenizer, BatchCollator


def test_lexical_retriever_bm25():
    p1 = PremiseDecl(name="Nat.add_comm", type_str="∀ (n m : Nat), n + m = m + n", module="Init.Data.Nat.Basic")
    p2 = PremiseDecl(name="Nat.mul_comm", type_str="∀ (n m : Nat), n * m = m * n", module="Init.Data.Nat.Basic")
    p3 = PremiseDecl(name="List.nil_append", type_str="∀ {α : Type} (l : List α), [] ++ l = l", module="Init.Data.List.Basic")
    
    retriever = LexicalRetriever()
    retriever.index_premises({"Nat.add_comm": p1, "Nat.mul_comm": p2, "List.nil_append": p3})
    
    # Query for addition
    g = ProofGoal(
        goal_id="g1",
        target="x + y = y + x",
        hypotheses=[LocalHypothesis(name="x", type_str="Nat"), LocalHypothesis(name="y", type_str="Nat")],
        mvar_id="?m.1",
    )
    results = retriever.retrieve(g, top_k=2)
    assert len(results) > 0
    top_name = results[0][0]
    assert top_name == "Nat.add_comm"


def test_dense_dual_encoder_forward_and_loss():
    vocab_size = 100
    model = DenseDualEncoder(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    
    # Goal batch
    g_ids = torch.randint(1, vocab_size, (4, 10))
    g_mask = torch.ones(4, 10, dtype=torch.bool)
    g_vec = model.encode_goal(g_ids, g_mask)
    assert g_vec.shape == (4, 32)
    # Check normalization: norm should be approximately 1.0
    norms = torch.norm(g_vec, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    # Positive premise batch
    p_ids = torch.randint(1, vocab_size, (4, 8))
    p_mask = torch.ones(4, 8, dtype=torch.bool)
    p_vec = model.encode_premise(p_ids, p_mask)
    assert p_vec.shape == (4, 32)

    # In-batch contrastive loss
    loss = model.contrastive_loss(g_vec, p_vec)
    assert loss.item() > 0.0
    loss.backward()


def test_hard_negative_sampler_scope():
    p1 = PremiseDecl(name="Nat.add_comm", type_str="...", module="Init.Data.Nat")
    p2 = PremiseDecl(name="Nat.add_assoc", type_str="...", module="Init.Data.Nat")
    p3 = PremiseDecl(name="Nat.mul_comm", type_str="...", module="Init.Data.Nat")
    p4 = PremiseDecl(name="List.append_nil", type_str="...", module="Init.Data.List")
    
    env = EnvironmentSpec(
        toolchain="lean4",
        eligible_premises={
            "Nat.add_comm": p1,
            "Nat.add_assoc": p2,
            "Nat.mul_comm": p3,
            "List.append_nil": p4,
        },
    )
    
    sampler = HardNegativeSampler(seed=42)
    negs = sampler.sample_negatives("Nat.add_comm", env, num_negatives=2, target_name="my_target")
    assert len(negs) == 2
    assert "Nat.add_comm" not in negs
    assert "my_target" not in negs
    assert all(n in env.eligible_premises for n in negs)


def test_premise_index_and_recall_evaluation():
    p1 = PremiseDecl(name="Nat.add_comm", type_str="∀ (n m : Nat), n + m = m + n", module="Init.Data.Nat.Basic")
    p2 = PremiseDecl(name="Nat.add_assoc", type_str="∀ (n m k : Nat), n + m + k = n + (m + k)", module="Init.Data.Nat.Basic")
    env = EnvironmentSpec(
        toolchain="lean4",
        eligible_premises={"Nat.add_comm": p1, "Nat.add_assoc": p2},
    )
    
    index = PremiseIndex(env_spec=env)
    g = ProofGoal(goal_id="g1", target="a + b = b + a", mvar_id="?m.1")
    
    results = index.retrieve(g, top_k=2)
    assert len(results) == 2
    
    queries = [
        (g, ["Nat.add_comm"]),
        (ProofGoal(goal_id="g2", target="a + b + c = a + (b + c)", mvar_id="?m.2"), ["Nat.add_assoc"]),
    ]
    recall_stats = PremiseIndex.evaluate_recall(queries, index.retrieve, k_values=(1, 2))
    assert recall_stats["recall@1"] >= 0.5
    assert recall_stats["recall@2"] == 1.0
