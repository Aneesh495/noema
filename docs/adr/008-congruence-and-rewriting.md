# ADR-008: Ground Congruence Closure and Equational Term Rewriting

## Context
Equational reasoning accounts for a substantial fraction of proof steps in Lean 4 (such as simplifying algebraic expressions, evaluating terms, or applying identity lemmas). In pure neural autoregressive decoding, generating each step of an equational chain requires multiple token generations and search steps, frequently hallucinating invalid algebraic identities. Symbolic solvers excel at equational logic and can normalize terms deterministically in microseconds.

## Decision
Noema integrates deterministic equational reasoning directly into the search engine:
1. First-Order Term Rewriting (`TermRewriter`):
   - Directs algebraic identities into confluent rewrite systems using Lexicographic Path Order (LPO) symbol precedence.
   - Computes critical pairs and executes Knuth-Bendix completion to ensure confluence.
   - Evaluates subterms to unique normal forms.
2. Ground Congruence Closure (`CongruenceClosure`):
   - Implements Nelson-Oppen congruence closure on ground terms via union-find and signature tables.
   - Propagates functional congruence ($a = b \implies f(a) = f(b)$) automatically.
   - Generates explainable proof traces using Lowest Common Ancestor (LCA) paths in the disjoint set forest.

## Consequences
- Offloads deterministic algebraic and equational reasoning from the neural policy, freeing search budgets for non-trivial inductive leaps.
- Generates verified Lean 4 `rfl`, `simp`, and `congr` tactic scripts directly from symbolic normal forms.
