"""Tests for the persistent Lean 4 REPL interactive session manager."""

import pytest
from noema.contracts.actions import TacticAction
from noema.contracts.environment import EnvironmentSpec, PremiseDecl
from noema.contracts.states import ProofState
from noema.contracts.theorems import TheoremSpec
from noema.formal.parser import GoalParser
from noema.formal.repl import LeanReplError, PersistentLeanSession
from noema.formal.types import TransitionStatus


@pytest.fixture
def sample_env() -> EnvironmentSpec:
    return EnvironmentSpec(
        toolchain="leanprover/lean4:v4.8.0",
        imported_modules=["Init.Data.Nat.Basic"],
        available_premises=[
            PremiseDecl(
                name="Nat.add_comm",
                module="Init.Data.Nat.Basic",
                type_str="∀ (n m : Nat), n + m = m + n",
            )
        ],
    )


def test_persistent_lean_session_lifecycle(sample_env):
    session = PersistentLeanSession(sample_env)
    assert not session.is_alive
    session.start()
    assert session.is_alive
    session.stop()
    assert not session.is_alive


def test_persistent_lean_session_invalid_syntax(sample_env):
    session = PersistentLeanSession(sample_env)
    session.start()
    try:
        thm = TheoremSpec(
            theorem_id="test_thm",
            name="test_thm",
            module="Test.Basic",
            statement="∀ (x : Nat), x = x",
        )
        initial_state = GoalParser.make_proof_state(
            "x : Nat\n⊢ x = x",
            env_hash=sample_env.environment_hash(),
        )
        action = TacticAction.from_string("bogus_unknown_tactic_xyz")
        res = session.execute_tactic(thm, initial_state, action, accumulated_history=[])
        assert res.status == TransitionStatus.SYNTAX_ERROR
        assert not res.is_valid_transition
    finally:
        session.stop()


def test_persistent_lean_session_ineligible_premise(sample_env):
    session = PersistentLeanSession(sample_env)
    session.start()
    try:
        thm = TheoremSpec(
            theorem_id="test_thm",
            name="test_thm",
            module="Test.Basic",
            statement="∀ (x : Nat), x = x",
        )
        initial_state = GoalParser.make_proof_state(
            "x : Nat\n⊢ x = x",
            env_hash=sample_env.environment_hash(),
        )
        action = TacticAction.from_string("rw [Bogus.lemma]")
        res = session.execute_tactic(thm, initial_state, action, accumulated_history=[])
        assert res.status == TransitionStatus.INVALID_TACTIC
        assert "Ineligible premise" in res.diagnostics
    finally:
        session.stop()


def test_persistent_lean_session_real_step(sample_env):
    session = PersistentLeanSession(sample_env)
    session.start()
    try:
        thm = TheoremSpec(
            theorem_id="demo_refl",
            name="demo_refl",
            module="Test.Basic",
            statement="∀ (x : Nat), x = x",
        )
        initial_state = GoalParser.make_proof_state(
            "x : Nat\n⊢ x = x",
            env_hash=sample_env.environment_hash(),
        )
        # Step 1: intro x
        res1 = session.execute_tactic(
            thm,
            initial_state,
            TacticAction.from_string("intro x"),
            accumulated_history=[],
        )
        assert res1.is_valid_transition

        # Step 2: rfl -> proves it
        res2 = session.execute_tactic(
            thm,
            res1.after_state or initial_state,
            TacticAction.from_string("rfl"),
            accumulated_history=["intro x"],
        )
        assert res2.is_valid_transition
        assert res2.status == TransitionStatus.PROVED
    finally:
        session.stop()
