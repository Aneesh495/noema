"""Unit tests for interactive proof shell."""

import pytest
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.states import LocalHypothesis, ProofGoal, ProofState
from noema.contracts.theorems import TheoremSpec
from noema.formal.checker import IndependentChecker
from noema.shell import InteractiveProofShell, ShellCommandType


@pytest.fixture
def sample_theorem() -> TheoremSpec:
    return TheoremSpec(
        theorem_id="thm_shell_test",
        name="shell_id_thm",
        module="Init.Core",
        statement="∀ (P : Prop), P → P",
        family="logic",
        eligible_premise_names=["id", "rfl"],
    )


@pytest.fixture
def sample_state() -> ProofState:
    goal = ProofGoal(
        goal_id="g1",
        target="P → P",
        hypotheses=[LocalHypothesis(name="P", type_str="Prop")],
        mvar_id="?m.1",
    )
    return ProofState(state_id="s0", goals=[goal], depth=0)


def test_shell_command_parsing(sample_theorem, sample_state):
    shell = InteractiveProofShell(sample_theorem, sample_state)

    cmd, arg = shell.parse_command("intro hp")
    assert cmd == ShellCommandType.TACTIC
    assert arg == "intro hp"

    cmd, arg = shell.parse_command("suggest 3")
    assert cmd == ShellCommandType.SUGGEST
    assert arg == "3"

    cmd, arg = shell.parse_command("undo")
    assert cmd == ShellCommandType.UNDO

    cmd, arg = shell.parse_command("redo")
    assert cmd == ShellCommandType.REDO

    cmd, arg = shell.parse_command("quit")
    assert cmd == ShellCommandType.QUIT


def test_shell_status_formatting(sample_theorem, sample_state):
    shell = InteractiveProofShell(sample_theorem, sample_state)
    status_str = shell.format_status()
    assert "shell_id_thm" in status_str
    assert "Active Goals: 1" in status_str
    assert "⊢ P → P" in status_str


def test_shell_suggestion_and_retrieval(sample_theorem, sample_state):
    def dummy_suggester(state, k):
        return [("intro hp", 0.95), ("exact hp", 0.8)]

    def dummy_retriever(query, k):
        return ["Nat.add_assoc", "Nat.add_comm"]

    shell = InteractiveProofShell(
        sample_theorem,
        sample_state,
        policy_suggester=dummy_suggester,
        premise_retriever=dummy_retriever,
    )
    hints = shell.suggest_tactics(top_k=2)
    assert len(hints) == 2
    assert hints[0][0] == "intro hp"

    premises = shell.retrieve_premises("add", top_k=2)
    assert len(premises) == 2
    assert "Nat.add_assoc" in premises


def test_shell_execute_tactic_and_undo_redo(sample_theorem, sample_state):
    checker = IndependentChecker(env_spec=EnvironmentSpec(toolchain="leanprover/lean4:v4.8.0"))
    shell = InteractiveProofShell(sample_theorem, sample_state, checker=checker)

    # First step
    ok, msg = shell.execute_tactic("intro P hp")
    assert ok

    # Second step completes the proof
    ok2, msg2 = shell.execute_tactic("exact hp")
    assert ok2
    assert shell.is_solved
    assert "SOLVED" in shell.format_status()

    # Test undo
    undone, u_msg = shell.undo()
    assert undone
    assert not shell.is_solved
    assert len(shell.applied_tactics) == 1

    # Test redo
    redone, r_msg = shell.redo()
    assert redone
    assert shell.is_solved
    assert len(shell.applied_tactics) == 2


def test_shell_export_proof_script(sample_theorem, sample_state):
    shell = InteractiveProofShell(sample_theorem, sample_state)
    shell.applied_tactics = ["intro P hp", "exact hp"]
    exported = shell.export_proof_script()
    assert "theorem shell_id_thm : ∀ (P : Prop), P → P := by" in exported
    assert "exact hp" in exported
