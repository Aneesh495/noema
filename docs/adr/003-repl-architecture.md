# ADR-003: Persistent Lean 4 Process and REPL Architecture

## Context
Invoking the `lean` CLI as a fresh subprocess for every single tactic step incurs process spawn overhead, environment re-initialization, and module reloading penalties of 200 to 500 milliseconds per step. During high-throughput tree search exploring thousands of candidate tactics per minute, subprocess overhead dominates the entire system runtime. However, a long-lived interactive process must maintain strict isolation and memory bounds to avoid state corruption or gradual memory leaks.

## Decision
Noema adopts a dual-mode verification architecture:
1. Search-Time REPL Session (`PersistentLeanSession`):
   - Maintains a persistent communication channel with a managed Lean 4 process.
   - Communicates via newline-delimited JSON commands over standard input and output.
   - Implements state checkpoints, fast tactic transitions, timeout watchers, and crash recovery.
2. Final Verification Checker (`IndependentChecker`):
   - When a proof candidate trajectory reaches QED, the complete proof script is submitted to a pristine, fresh `lean` process.
   - The fresh process independently compiles the full `.lean` file and runs `#print axioms` to ensure that no cached state, unsound environment mutations, or unassigned metavariables permitted invalid proofs.

## Consequences
- Reduces tactic transition latency during search from 350 ms down to under 15 ms per step.
- Guarantees complete kernel soundness by requiring every accepted proof certificate to pass independent verification in an isolated process.
