# Walkthrough 1: Interactive Lean 4 Proof Session

Noema's terminal shell checks tactics with Lean and displays the goals that remain after each step. You can also use the same shell from Python.

## Prerequisites

Build the Lean package from the repository root:

```bash
cd lean
lake build
cd ..
```

Install Noema in an active virtual environment as described in the [quickstart](../../README.md#installation--setup).

## Launch the terminal shell

```bash
python3 -m noema.cli shell
```

The default theorem has one proposition binder. Enter `intro P h`, then `exact h`. Use `status` to inspect the result and `quit` to leave. The CLI uses built-in tactic hints when no model is attached.

Pass a different declaration with `--theorem`:

```bash
python3 -m noema.cli shell --theorem 'theorem and_comm_demo (P Q : Prop) : P ∧ Q → Q ∧ P := by'
```

## Run the same proof from Python

```python
from noema.contracts.states import ProofGoal, ProofState
from noema.contracts.theorems import TheoremSpec
from noema.shell import InteractiveProofShell

theorem = TheoremSpec(
    theorem_id="and_comm_demo",
    name="and_comm_demo",
    module="Init",
    statement="∀ (P Q : Prop), P ∧ Q → Q ∧ P",
    family="logic",
)
initial_state = ProofState(
    state_id="start",
    goals=[ProofGoal(goal_id="g0", target=theorem.statement, mvar_id="?m.0")],
)
shell = InteractiveProofShell(theorem, initial_state)

for tactic in ("intro P Q h", "constructor", "exact h.2", "exact h.1"):
    accepted, message = shell.execute_tactic(tactic)
    if not accepted:
        raise RuntimeError(message)
    print(message)

print(shell.format_status())
assert shell.is_solved
```

After `constructor`, Lean reports two goals, `Q` and `P`. The final status reads `SOLVED (QED)`. Each accepted step saves the previous and next proof states. Use `shell.undo()` and `shell.redo()` to move between them.

When Lean rejects a tactic, `execute_tactic()` returns `False` with the diagnostic and keeps the last accepted state and history.
