"""Unit tests for Noema contracts."""

import pytest
from noema.contracts import (
    EnvironmentSpec,
    PremiseDecl,
    TheoremSpec,
    LocalHypothesis,
    ProofGoal,
    ProofState,
    TacticKind,
    TacticAction,
    ActionGrammar,
    HyperNode,
    HyperEdge,
    ProofHypergraph,
    VerificationStatus,
    ProofCertificate,
)


def test_environment_spec_and_eligibility():
    p1 = PremiseDecl(name="Nat.add_comm", type_str="∀ (n m : Nat), n + m = m + n", module="Init.Data.Nat.Basic")
    p2 = PremiseDecl(name="Nat.add_assoc", type_str="∀ (n m k : Nat), n + m + k = n + (m + k)", module="Init.Data.Nat.Basic")
    
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
        eligible_premises={"Nat.add_comm": p1, "Nat.add_assoc": p2},
    )
    
    assert env.is_premise_eligible("Nat.add_comm")
    assert not env.is_premise_eligible("Nat.add_comm", target_name="Nat.add_comm")
    assert not env.is_premise_eligible("Nat.mul_comm")
    assert len(env.environment_hash()) == 64


def test_theorem_spec_and_hashing():
    th = TheoremSpec(
        theorem_id="th_001",
        name="test_thm",
        module="Test.Basic",
        statement="∀ (n : Nat), n + 0 = n",
    )
    decl = th.make_full_declaration("by rfl")
    assert "theorem test_thm : ∀ (n : Nat), n + 0 = n := by rfl" in decl
    assert len(th.statement_hash()) == 64


def test_proof_state_and_mvar_coupling():
    h1 = LocalHypothesis(name="n", type_str="Nat")
    g1 = ProofGoal(goal_id="g1", target="n + 0 = n", hypotheses=[h1], mvar_id="?m.1", coupled_mvars=["?m.2"])
    g2 = ProofGoal(goal_id="g2", target="n + 0 = n", hypotheses=[h1], mvar_id="?m.2", coupled_mvars=["?m.1"])
    
    s1 = ProofState.compute_state_id([g1], {})
    s2 = ProofState.compute_state_id([g2], {})
    # Identical text but different metavariable couplings must have distinct state identities
    assert s1 != s2


def test_action_grammar_and_tampering():
    valid, msg = ActionGrammar.is_valid_tactic_syntax("apply Nat.add_comm")
    assert valid
    
    valid, msg = ActionGrammar.is_valid_tactic_syntax("rfl")
    assert valid
    
    valid, msg = ActionGrammar.is_valid_tactic_syntax("sorry")
    assert not valid
    assert "Prohibited" in msg

    valid, msg = ActionGrammar.is_valid_tactic_syntax("intro x; rfl")
    assert not valid
    assert "Semicolon" in msg


def test_hypergraph_propagation():
    h1 = LocalHypothesis(name="x", type_str="Nat")
    g_root = ProofGoal(goal_id="g0", target="x = x", hypotheses=[h1], mvar_id="?m.0")
    node0 = HyperNode(node_id="n0", state_id="s0", goal=g_root)
    
    hg = ProofHypergraph(root_node_id="n0")
    hg.add_node(node0)
    
    # Apply rfl: closes the goal with 0 subgoals
    action = TacticAction.from_string("rfl", goal_id="g0", mvar_id="?m.0")
    edge = HyperEdge(
        edge_id="e0",
        source_node_id="n0",
        target_node_ids=[], # 0 subgoals
        action=action,
    )
    hg.add_edge(edge)
    
    assert not hg.nodes["n0"].is_proved
    proved = hg.update_proved_status("n0")
    assert proved
    assert hg.nodes["n0"].is_proved


def test_proof_certificate_tamper_detection():
    cert = ProofCertificate(
        theorem_id="t1",
        theorem_name="test_refl",
        statement="x = x",
        statement_hash="hash_stmt",
        environment_hash="hash_env",
        tactic_script="intro x\nrfl\n",
        status=VerificationStatus.VERIFIED,
    )
    cert.seal()
    assert cert.verify_integrity()
    
    # Tamper with the tactic script
    cert.tactic_script = "intro x\nsorry\n"
    assert not cert.verify_integrity()
