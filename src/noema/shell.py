"""Interactive terminal proof shell with neural step completions and premise suggestions."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence
from noema.contracts.actions import TacticAction
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.states import LocalHypothesis, ProofGoal, ProofState
from noema.contracts.theorems import TheoremSpec
from noema.formal.checker import IndependentChecker


class ShellCommandType(Enum):
    TACTIC = "tactic"
    SUGGEST = "suggest"
    RETRIEVE = "retrieve"
    UNDO = "undo"
    REDO = "redo"
    STATUS = "status"
    EXPORT = "export"
    HELP = "help"
    QUIT = "quit"


@dataclass
class ShellHistoryEntry:
    """Historical step for interactive undo/redo navigation."""
    tactic: str
    previous_state: ProofState
    next_state: ProofState


class InteractiveProofShell:
    """Stateful interactive REPL shell for human-in-the-loop theorem proving."""

    def __init__(
        self,
        theorem: TheoremSpec,
        initial_state: ProofState,
        checker: IndependentChecker | None = None,
        policy_suggester: Callable[[ProofState, int], list[tuple[str, float]]] | None = None,
        premise_retriever: Callable[[str, int], list[str]] | None = None,
    ):
        self.theorem = theorem
        self.current_state = initial_state
        self.checker = checker or IndependentChecker(
            env_spec=EnvironmentSpec(toolchain="leanprover/lean4:v4.8.0")
        )
        self.policy_suggester = policy_suggester
        self.premise_retriever = premise_retriever
        self.history: list[ShellHistoryEntry] = []
        self.redo_stack: list[ShellHistoryEntry] = []
        self.applied_tactics: list[str] = []

    @property
    def is_solved(self) -> bool:
        return len(self.current_state.goals) == 0

    def parse_command(self, raw_input: str) -> tuple[ShellCommandType, str]:
        """Parse raw command line into command type and argument."""
        s = raw_input.strip()
        if not s:
            return ShellCommandType.STATUS, ""
        parts = s.split(maxsplit=1)
        cmd_str = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd_str in ["q", "quit", "exit"]:
            return ShellCommandType.QUIT, ""
        if cmd_str in ["h", "help", "?"]:
            return ShellCommandType.HELP, ""
        if cmd_str in ["u", "undo"]:
            return ShellCommandType.UNDO, ""
        if cmd_str in ["r", "redo"]:
            return ShellCommandType.REDO, ""
        if cmd_str in ["s", "suggest", "hint"]:
            return ShellCommandType.SUGGEST, arg
        if cmd_str in ["find", "retrieve", "premise"]:
            return ShellCommandType.RETRIEVE, arg
        if cmd_str in ["export", "save"]:
            return ShellCommandType.EXPORT, arg
        if cmd_str in ["status", "goal", "goals"]:
            return ShellCommandType.STATUS, ""

        # Default action is executing a tactic
        if cmd_str == "tactic":
            return ShellCommandType.TACTIC, arg
        return ShellCommandType.TACTIC, s

    def execute_tactic(self, tactic_str: str) -> tuple[bool, str]:
        """Execute a tactic, test via fresh checker, and transition state."""
        candidate_tactics = self.applied_tactics + [tactic_str]
        full_script = "\n".join(candidate_tactics)

        cert, audit = self.checker.verify_proof(self.theorem, full_script)
        if audit.is_valid:
            # Solved the full theorem
            prev = self.current_state
            self.applied_tactics.append(tactic_str)
            new_state = ProofState(
                state_id=f"state_{len(self.applied_tactics)}",
                goals=[],
                depth=self.current_state.depth + 1,
            )
            entry = ShellHistoryEntry(
                tactic=tactic_str,
                previous_state=prev,
                next_state=new_state,
            )
            self.history.append(entry)
            self.redo_stack.clear()
            self.current_state = new_state
            return True, "Theorem proved successfully! No goals remaining."

        # If checker reported unsolved goals, advance partial state
        if "unsolved goals" in cert.kernel_diagnostics or "error" not in cert.kernel_diagnostics.lower():
            prev = self.current_state
            self.applied_tactics.append(tactic_str)
            # Simulated partial goal progression
            remaining_goals = self.current_state.goals[1:] if len(self.current_state.goals) > 1 else self.current_state.goals
            new_state = ProofState(
                state_id=f"state_{len(self.applied_tactics)}",
                goals=remaining_goals,
                depth=self.current_state.depth + 1,
            )
            entry = ShellHistoryEntry(
                tactic=tactic_str,
                previous_state=prev,
                next_state=new_state,
            )
            self.history.append(entry)
            self.redo_stack.clear()
            self.current_state = new_state
            return True, f"Tactic '{tactic_str}' accepted. {len(new_state.goals)} goals remaining."

        return False, f"Tactic rejected by Lean kernel: {cert.kernel_diagnostics}"

    def suggest_tactics(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Query the neural model policy for candidate tactic suggestions."""
        if self.policy_suggester is None:
            return [("intro", 0.8), ("simp", 0.5), ("exact", 0.3)]
        return self.policy_suggester(self.current_state, top_k)

    def retrieve_premises(self, query: str, top_k: int = 5) -> list[str]:
        """Query premise retriever index for relevant lemmas."""
        if self.premise_retriever is None:
            return self.theorem.eligible_premise_names[:top_k]
        return self.premise_retriever(query, top_k)

    def undo(self) -> tuple[bool, str]:
        """Revert the most recent tactic step."""
        if not self.history:
            return False, "Undo stack is empty."
        entry = self.history.pop()
        self.redo_stack.append(entry)
        self.applied_tactics.pop()
        self.current_state = entry.previous_state
        return True, f"Undid tactic '{entry.tactic}'."

    def redo(self) -> tuple[bool, str]:
        """Reapply the most recently undone tactic step."""
        if not self.redo_stack:
            return False, "Redo stack is empty."
        entry = self.redo_stack.pop()
        self.history.append(entry)
        self.applied_tactics.append(entry.tactic)
        self.current_state = entry.next_state
        return True, f"Redid tactic '{entry.tactic}'."

    def export_proof_script(self, target_path: str | None = None) -> str:
        """Export complete compilable Lean 4 source code for the proof."""
        proof_lines = "\n  ".join(self.applied_tactics)
        proof_body = f"by\n  {proof_lines}" if self.applied_tactics else "by sorry"
        code = self.theorem.make_full_declaration(proof_body=proof_body)
        if target_path:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(code)
        return code

    def format_status(self) -> str:
        """Render terminal representation of the current proof state."""
        lines = [f"=== Proof Shell: {self.theorem.name} ==="]
        if self.is_solved:
            lines.append("Status: SOLVED (QED)")
            lines.append(f"Steps: {len(self.applied_tactics)}")
            return "\n".join(lines)

        lines.append(f"Active Goals: {len(self.current_state.goals)}")
        for idx, g in enumerate(self.current_state.goals):
            lines.append(f"\nGoal [{idx + 1}/{len(self.current_state.goals)}]:")
            for h in g.hypotheses:
                lines.append(f"  {h.canonical_repr()}")
            lines.append(f"  ⊢ {g.target}")

        if self.applied_tactics:
            lines.append("\nApplied Tactics:")
            for i, t in enumerate(self.applied_tactics):
                lines.append(f"  {i + 1}. {t}")

        return "\n".join(lines)
