# Walkthrough 1: Interactive Proof Session with Lean 4 REPL

This guide details how to interact directly with the Lean 4 proof environment using Noema's persistent REPL protocol and interactive terminal shell.

## Prerequisites

Ensure the Lean 4 environment is built and accessible via Lake:

```bash
cd lean
lake build
cd ..
```

Verify that the Python virtual environment is active and Noema is installed in editable mode:

```bash
source .venv/bin/activate
python3 -c "import noema; print(noema.__name__)"
```

## Launching the Interactive Shell

Noema includes a dedicated proof shell with syntax awareness, neural tactic completion, and undo/redo capabilities:

```bash
python3 -m noema.shell
```

Alternatively, you can drive the REPL programmatically from Python using the `LeanREPLSession` client.

## Programmatic REPL Session Example

Create a script or execute in a Python session:

```python
import sys
from noema.formal.repl import LeanREPLSession

# Initialize persistent session using Lake environment
session = LeanREPLSession(project_dir="lean")
session.start()

# Define theorem declaration in Lean 4
theorem_code = """
theorem and_comm_demo (P Q : Prop) : P ∧ Q → Q ∧ P := by
"""

# Send declaration to initialize proof state
response = session.submit_command(theorem_code)
print(f"Session state ID: {response.env_id}")
print(f"Open goals: {response.goals}")
# Output: [ "case intro\nP Q : Prop\nh : P ∧ Q\n⊢ Q ∧ P" ]

# Execute first tactic: intro h
t1 = session.run_tactic(response.env_id, "intro h")
print(f"Tactic 1 status: {t1.status}, Goals: {t1.goals}")

# Execute second tactic: constructor
t2 = session.run_tactic(t1.env_id, "constructor")
print(f"Tactic 2 status: {t2.status}, Goals: {t2.goals}")
# Two subgoals created: ⊢ Q and ⊢ P

# Resolve first branch
t3 = session.run_tactic(t2.env_id, "exact h.2")
print(f"Tactic 3 status: {t3.status}, Remaining goals: {t3.goals}")

# Resolve second branch and close proof
t4 = session.run_tactic(t3.env_id, "exact h.1")
print(f"Tactic 4 status: {t4.status}, Closed: {t4.is_closed}")

session.close()
```

## State Snapshotting and Rollback

The REPL protocol treats proof search states as immutable environment IDs:
- Every successful tactic application yields a distinct `env_id`.
- Backtracking is achieved by branching from a parent `env_id` without restarting the process.
- If a speculative tactic fails or diverges, the search engine discards the erroneous child `env_id` and continues from the parent.

## Diagnostic Handling

When an invalid tactic is evaluated, the REPL captures the diagnostic message without crashing:

```python
err_resp = session.run_tactic(t2.env_id, "ring")
if err_resp.has_error:
    print(f"Kernel error message: {err_resp.error_message}")
    # Search resumes from t2.env_id safely
```
