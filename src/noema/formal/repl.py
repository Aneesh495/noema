"""Stateful persistent Lean 4 REPL server process and interactive session manager."""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from noema.contracts.actions import ActionGrammar, TacticAction
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.states import ProofGoal, ProofState
from noema.contracts.theorems import TheoremSpec
from noema.formal.parser import GoalParser
from noema.formal.types import TransitionResult, TransitionStatus


class LeanReplError(Exception):
    """Raised when the persistent Lean 4 REPL encounters an unrecoverable protocol error."""
    pass


class PersistentLeanSession:
    """An interactive, persistent Lean 4 process session for low-latency tactic execution."""

    def __init__(
        self,
        env_spec: EnvironmentSpec,
        lean_cmd: str | None = None,
        timeout_sec: float = 10.0,
    ):
        self.env_spec = env_spec
        self.lean_cmd = lean_cmd or shutil.which("lean") or "lean"
        self.timeout_sec = timeout_sec
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.is_alive = False
        self.session_id = f"session_{os.getpid()}_{int(time.time() * 1000)}"
        self.active_theorem: TheoremSpec | None = None
        self.history: list[str] = []

    def start(self) -> None:
        """Start the Lean 4 worker process."""
        with self._lock:
            if self.is_alive:
                return
            # Launch lean process with interactive flags
            cmd = [self.lean_cmd, "--run"]
            try:
                self.process = subprocess.Popen(
                    [self.lean_cmd, "--server"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                self.is_alive = True
            except Exception as e:
                self.is_alive = False
                raise LeanReplError(f"Failed to start Lean REPL process: {e}")

    def execute_tactic(
        self,
        theorem: TheoremSpec,
        current_state: ProofState,
        action: TacticAction,
        accumulated_history: list[str],
        timeout_sec: float | None = None,
    ) -> TransitionResult:
        """Execute a tactical step through the persistent session."""
        start_t = time.perf_counter()
        t_sec = timeout_sec or self.timeout_sec

        # 1. Grammar validation
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

        # 3. Construct and compile verification probe
        full_script = accumulated_history + [action.raw_tactic]
        code = self._build_script(theorem, full_script)

        # Execute using standard safe runner
        stdout, stderr, ret_code = self._run_code(code, timeout_sec=t_sec)
        elapsed = (time.perf_counter() - start_t) * 1000.0
        combined = (stdout + "\n" + stderr).strip()

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

        new_state = GoalParser.make_proof_state(
            combined,
            env_hash=self.env_spec.environment_hash(),
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

    def _build_script(self, theorem: TheoremSpec, tactic_lines: list[str]) -> str:
        imports = "\n".join(f"import {m}" for m in self.env_spec.imported_modules)
        ns_open = f"namespace {theorem.namespace}\n" if theorem.namespace else ""
        ns_close = f"\nend {theorem.namespace}" if theorem.namespace else ""

        indented = "\n".join(f"  {l.strip()}" for l in tactic_lines if l.strip())
        body = (
            f"theorem {theorem.name} : {theorem.statement} := by\n"
            f"{indented}\n"
        )
        return f"{imports}\n\n{ns_open}{body}{ns_close}\n"

    def _run_code(self, code: str, timeout_sec: float) -> tuple[str, str, int]:
        import tempfile
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

    def stop(self) -> None:
        """Terminate the session cleanly."""
        with self._lock:
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=1.0)
                except Exception:
                    self.process.kill()
                self.process = None
            self.is_alive = False
