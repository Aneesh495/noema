"""Lean 4 formal verification bridge and transition execution."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from noema.contracts.actions import ActionGrammar, TacticAction
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.states import ProofGoal, ProofState
from noema.contracts.theorems import TheoremSpec
from noema.formal.parser import GoalParser
from noema.formal.types import TransitionResult, TransitionStatus


class FormalBridge:
    """Communicates with the Lean 4 formal system to execute tactic transitions and retrieve goals."""

    def __init__(
        self,
        env_spec: EnvironmentSpec,
        lean_cmd: str | None = None,
        default_timeout_sec: float = 5.0,
    ):
        self.env_spec = env_spec
        self.lean_cmd = lean_cmd or shutil.which("lean") or "lean"
        self.default_timeout_sec = default_timeout_sec
        self.env_hash = env_spec.environment_hash()

    def get_initial_state(self, theorem: TheoremSpec) -> tuple[ProofState, str]:
        """Obtain initial ProofState for a theorem before any tactics are applied."""
        code = self._build_probe_file(theorem, tactic_script="")
        output, err, code_res = self._run_lean_code(code, timeout_sec=self.default_timeout_sec)
        
        # In Lean 4, 'by' with empty or sorry outputs the goal
        state = GoalParser.make_proof_state(
            output + "\n" + err,
            env_hash=self.env_hash,
            depth=0,
            parent_state_id=None,
        )
        return state, output + "\n" + err

    def step(
        self,
        theorem: TheoremSpec,
        current_state: ProofState,
        action: TacticAction,
        tactic_history: list[str],
        timeout_sec: float | None = None,
    ) -> TransitionResult:
        """Execute a single tactic transition and return resulting state."""
        start_t = time.perf_counter()
        t_sec = timeout_sec or self.default_timeout_sec

        # 1. Grammar and syntactic sanity validation
        valid, msg = ActionGrammar.is_valid_tactic_syntax(action.raw_tactic)
        if not valid:
            elapsed = (time.perf_counter() - start_t) * 1000.0
            return TransitionResult(
                status=TransitionStatus.SYNTAX_ERROR,
                before_state_id=current_state.state_id,
                action=action,
                diagnostics=msg,
                execution_time_ms=elapsed,
                is_valid_transition=False,
                state_changed=False,
            )

        # 2. Check premise eligibility
        for prem in action.premises:
            if not self.env_spec.is_premise_eligible(prem, target_name=theorem.name):
                elapsed = (time.perf_counter() - start_t) * 1000.0
                return TransitionResult(
                    status=TransitionStatus.INVALID_TACTIC,
                    before_state_id=current_state.state_id,
                    action=action,
                    diagnostics=f"Ineligible premise: {prem}",
                    execution_time_ms=elapsed,
                    is_valid_transition=False,
                    state_changed=False,
                )

        # 3. Construct Lean 4 source file with accumulated tactics + new action
        proposed_script = tactic_history + [action.raw_tactic]
        code = self._build_probe_file(theorem, tactic_script="\n  ".join(proposed_script))

        stdout, stderr, ret_code = self._run_lean_code(code, timeout_sec=t_sec)
        elapsed = (time.perf_counter() - start_t) * 1000.0
        combined = (stdout + "\n" + stderr).strip()

        # 4. Analyze execution outcome
        if ret_code == -9 or "timed out" in combined.lower():
            return TransitionResult(
                status=TransitionStatus.TIMEOUT,
                before_state_id=current_state.state_id,
                action=action,
                diagnostics="Lean execution timed out",
                execution_time_ms=elapsed,
                is_valid_transition=False,
                state_changed=False,
            )

        has_unsolved_goals = ("unsolved goals" in combined.lower())
        fatal_error_lines = [
            l for l in combined.splitlines()
            if "error:" in l and "unsolved goals" not in l
        ]

        if fatal_error_lines or (ret_code != 0 and not has_unsolved_goals):
            return TransitionResult(
                status=TransitionStatus.INVALID_TACTIC,
                before_state_id=current_state.state_id,
                action=action,
                diagnostics=combined,
                execution_time_ms=elapsed,
                is_valid_transition=False,
                state_changed=False,
            )

        # 5. Goal extraction
        new_state = GoalParser.make_proof_state(
            combined,
            env_hash=self.env_hash,
            depth=current_state.depth + 1,
            parent_state_id=current_state.state_id,
        )

        state_changed = (new_state.state_id != current_state.state_id)
        is_proved = (ret_code == 0 and not new_state.has_open_goals())

        return TransitionResult(
            status=TransitionStatus.PROVED if is_proved else TransitionStatus.SUCCESS,
            before_state_id=current_state.state_id,
            action=action,
            after_state=new_state,
            open_goals=new_state.get_open_goals(),
            diagnostics=combined,
            execution_time_ms=elapsed,
            is_valid_transition=True,
            state_changed=state_changed,
        )

    def _build_probe_file(self, theorem: TheoremSpec, tactic_script: str) -> str:
        imports = "\n".join(f"import {m}" for m in self.env_spec.imported_modules)
        ns_open = f"namespace {theorem.namespace}\n" if theorem.namespace else ""
        ns_close = f"\nend {theorem.namespace}" if theorem.namespace else ""

        lines = [f"  {l.strip()}" for l in tactic_script.strip().splitlines() if l.strip()]
        indented = "\n".join(lines)
        body = (
            f"theorem {theorem.name} : {theorem.statement} := by\n"
            f"{indented}\n"
        )
        return f"{imports}\n\n{ns_open}{body}{ns_close}\n"

    def _run_lean_code(self, code: str, timeout_sec: float) -> tuple[str, str, int]:
        with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            cmd = [self.lean_cmd, temp_path]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            return "", "Execution timed out", -9
        except Exception as e:
            return "", str(e), 1
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
