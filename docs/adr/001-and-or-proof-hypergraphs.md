# ADR-001: AND/OR Proof Hypergraphs vs Simple Search Trees

## Context
Interactive theorem proving tactics frequently split a single proof obligation into multiple concurrent subgoals (for example, applying `constructor` on `P ∧ Q` produces two obligations: `⊢ P` and `⊢ Q`). Standard tree search algorithms assume an action transitions from state $s$ to a single successor state $s'$. Under tactic execution, however, an action generates an AND-branch: all child subgoals must be solved simultaneously to witness a proof of the parent. Treating multi-goal states as monolithic nodes in a simple tree leads to exponential state space explosion and prevents independent progress on decoupled subgoals.

## Decision
Noema represents proof search state spaces as directed AND/OR hypergraphs.
- State nodes correspond to individual localized goals or conjuncts.
- Hyperedges represent tactic applications, connecting a source goal node to a tuple of target child subgoals $(g_1, g_2, \dots, g_k)$.
- When $k = 0$, the hyperedge represents a terminal base-case proof (QED), such as `rfl`, `exact h`, or `assumption`.
- Solvability propagates upward via generalized bottom-up fixed-point iteration: a node is proved if at least one incoming hyperedge has all target subgoals proved.

## Consequences
- Decouples independent subgoals so that tactics applied to one branch do not invalidate search progress on orthogonal sibling goals.
- Requires hypergraph-aware search heuristics (such as hyperpath cost accumulation and PUCT generalized across hyperedges).
- Cycle detection must track hyperedge dependencies rather than simple ancestor paths to avoid infinite loops across mutually recursive goals.
