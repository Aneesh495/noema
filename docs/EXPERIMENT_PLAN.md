# Experiment Plan: Neural Theorem Proving with Noema

This document describes the empirical evaluation plan for assessing Noema's search efficiency, neural guidance quality, and formal theorem proving success.

## 1. Research Questions

- **RQ1 (Multi-Task Guidance)**: Does joint training of the policy head with the state value head and censored progress head reduce required REPL expansion steps compared to a policy-only baseline?
- **RQ2 (Premise Retrieval Hybridization)**: Does combining lexical BM25, dense neural embeddings, and Personalized PageRank over the premise dependency graph improve premise recall@10 and proof success over isolated retrievers?
- **RQ3 (Diverse Search vs MCTS)**: How does diverse beam search compare to MCTS on AND/OR hypergraphs under restricted computational budgets (50, 100, 500 REPL steps)?
- **RQ4 (Out-of-Distribution Generalization)**: How gracefully does proof success degrade when evaluating on theorems with proof depths exceeding training horizons?

## 2. Experimental Configurations

### Condition A: Baseline Systems
- **Random Tactic Search**: Uniformly samples syntactically valid Lean 4 tactics from a fixed grammar.
- **Breadth-First Search (BFS)**: Uninformed exhaustive search up to depth limit $d=6$.
- **Policy-Only Greedy Search**: Selects $\arg\max_t \pi_\theta(t \mid s)$ without tree search or value estimation.

### Condition B: Noema Search Configurations
- **Noema-MCTS**: PUCT-guided search on AND/OR proof hypergraphs with value backpropagation.
- **Noema-DiverseBeam**: Diverse beam search with $G=4$ groups and dissimilarity penalties.
- **Noema-Hybrid**: MCTS coupled with the symbolic congruence closure engine and premise retriever.

## 3. Benchmark Datasets

Evaluation spans eight mathematical corpora:
1. **Propositional Logic**: Classical and intuitionistic propositional theorems.
2. **Abstract Algebra**: Groups, rings, fields, monoids, homomorphisms.
3. **Order Theory**: Partially ordered sets, lattices, Galois connections.
4. **Category Theory**: Objects, morphisms, functors, natural transformations.
5. **Modular Arithmetic**: Congruences, divisibility, Chinese remainder instances.
6. **Set Theory**: Boolean algebra of sets, relations, power sets.
7. **General Topology**: Open sets, continuous maps, closures, interiors.
8. **Extrapolation Suite**: Conjectures with proof step depths $d \in [10, 30]$.

## 4. Evaluation Metrics

- **Pass@k**: Proportion of theorems successfully closed within $k$ sampled proof attempts:
  $$\text{pass}@k = \mathbb{E}\left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right]$$
  Evaluated at $k \in \{1, 8, 32\}$.
- **Search Efficiency**: Mean number of Lean REPL expansions per successfully solved theorem.
- **Premise Precision & Recall**: Precision@k and Recall@k for lemmas used in ground-truth proofs.
- **Verification Integrity**: Percentage of candidate proofs verified by `IndependentChecker` with zero kernel errors and zero unresolved `sorry` holes.
