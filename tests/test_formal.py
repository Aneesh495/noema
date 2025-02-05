"""Unit and integration tests for Lean 4 formal bridge and independent checker."""

import pytest
from noema.contracts import (
    EnvironmentSpec,
    PremiseDecl,
    TheoremSpec,
    TacticAction,
    VerificationStatus,
)
from noema.formal import (
    GoalParser,
    FormalBridge,
    IndependentChecker,
    TransitionStatus,
)


def test_goal_parser_single_goal():
    output = """
case intro
n : Nat
m : Nat
⊢ n + m = m + n
"""
    goals = GoalParser.parse_goals(output)
    assert len(goals) == 1
    g = goals[0]
    assert g.target == "n + m = m + n"
    assert len(g.hypotheses) == 2
    assert g.hypotheses[0].name == "n"
    assert g.hypotheses[0].type_str == "Nat"
    assert g.hypotheses[1].name == "m"


def test_goal_parser_multi_goal():
    output = """
case left
P : Prop
Q : Prop
hP : P
⊢ P

case right
P : Prop
Q : Prop
hQ : Q
⊢ Q
"""
    goals = GoalParser.parse_goals(output)
    assert len(goals) == 2
    assert goals[0].target == "P"
    assert goals[1].target == "Q"


def test_real_lean_bridge_transitions():
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )
    bridge = FormalBridge(env_spec=env)
    
    th = TheoremSpec(
        theorem_id="t_refl",
        name="test_refl_prop",
        module="Test.Basic",
        statement="∀ (x : Nat), x = x",
    )
    
    init_state, _ = bridge.get_initial_state(th)
    assert init_state.has_open_goals()
    
    # 1. Step: intro x
    action1 = TacticAction.from_string("intro x")
    res1 = bridge.step(th, init_state, action1, tactic_history=[])
    assert res1.is_valid_transition
    assert res1.status == TransitionStatus.SUCCESS
    assert res1.after_state is not None
    assert res1.after_state.has_open_goals()
    
    # 2. Step: rfl (should close the goal!)
    action2 = TacticAction.from_string("rfl")
    res2 = bridge.step(th, res1.after_state, action2, tactic_history=["intro x"])
    assert res2.is_valid_transition
    assert res2.status == TransitionStatus.PROVED
    assert res2.is_proved()


def test_independent_checker_real_proof():
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
        allowed_axioms=["propext", "Classical.choice", "Quot.sound"],
    )
    checker = IndependentChecker(env_spec=env)
    
    th = TheoremSpec(
        theorem_id="t_add_zero",
        name="test_add_zero",
        module="Test.Basic",
        statement="∀ (x : Nat), x + 0 = x",
    )
    
    # Valid proof
    cert, audit = checker.verify_proof(th, tactic_script="intro x\nrfl")
    assert cert.status == VerificationStatus.VERIFIED
    assert cert.verify_integrity()
    assert audit.is_valid
    assert audit.no_sorry
    assert audit.axioms_permitted


def test_independent_checker_rejects_sorry():
    env = EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
    )
    checker = IndependentChecker(env_spec=env)
    
    th = TheoremSpec(
        theorem_id="t_cheat",
        name="test_cheat_thm",
        module="Test.Basic",
        statement="∀ (x : Nat), x = 0",
    )
    
    cert, audit = checker.verify_proof(th, tactic_script="intro x\nsorry")
    assert cert.status == VerificationStatus.REJECTED
    assert not audit.is_valid
    assert not audit.no_sorry
    assert "sorry" in cert.kernel_diagnostics.lower()
