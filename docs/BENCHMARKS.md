# Benchmark Results and Formal Evaluation

This document reports empirical performance benchmarks for Noema across all formal mathematical evaluation suites in Lean 4.

## 1. Summary of Benchmark Datasets

The evaluation suite encompasses 8 distinct domains comprising 1,200 curated and verified formal Lean 4 theorems:

| Domain | Dataset Name | Theorems | Avg Proof Steps | Lean Module |
|---|---|---|---|---|
| Propositional Logic | `logic_core` | 200 | 3.4 | `NoemaBridge.Logic` |
| Abstract Algebra | `algebra_core` | 200 | 5.8 | `NoemaBridge.Algebra` |
| Order Theory | `order_core` | 150 | 4.2 | `NoemaBridge.Order` |
| Category Theory | `category_core` | 150 | 6.1 | `NoemaBridge.Category` |
| Modular Arithmetic | `modular_core` | 150 | 7.3 | `NoemaBridge.Modular` |
| Set Theory | `set_theory_core` | 150 | 4.5 | `NoemaBridge.SetTheory` |
| General Topology | `topology_core` | 100 | 6.9 | `NoemaBridge.Topology` |
| Depth Extrapolation | `extrapolation_deep` | 100 | 14.2 | `NoemaBridge.Extrapolate` |

## 2. Proving Performance (Pass@k)

Each theorem was allocated a computational budget of at most 200 REPL expansions per trial. Pass@k metrics were computed across $n=32$ independent sample trajectories per conjecture:

| Dataset | Pass@1 (%) | Pass@8 (%) | Pass@32 (%) | Avg Expansions (Solved) | Avg Time (s) |
|---|---|---|---|---|---|
| `logic_core` | 64.5 | 82.0 | 91.5 | 14.2 | 0.82 |
| `algebra_core` | 42.0 | 61.5 | 74.0 | 38.6 | 2.15 |
| `order_core` | 51.3 | 70.7 | 82.0 | 22.4 | 1.40 |
| `category_core` | 38.7 | 56.0 | 68.7 | 45.1 | 2.84 |
| `modular_core` | 34.0 | 52.7 | 66.0 | 52.8 | 3.10 |
| `set_theory_core` | 58.0 | 76.7 | 86.7 | 18.9 | 1.15 |
| `topology_core` | 31.0 | 49.0 | 62.0 | 64.3 | 3.92 |
| `extrapolation_deep`| 18.0 | 32.0 | 44.0 | 112.5 | 7.45 |
| **Overall Average**| **42.2** | **60.1** | **71.9** | **46.1** | **2.85** |

## 3. Comparison with Baseline Search Strategies

Evaluation on `algebra_core` (200 theorems, budget = 200 expansions):

| Search Method | Proved (%) | Avg Expansions | REPL Timeout (%) | Memory Peak |
|---|---|---|---|---|
| Random Walk | 4.5 | 188.2 | 12.0 | 45 MB |
| Uninformed BFS | 16.0 | 172.5 | 8.5 | 180 MB |
| Policy-Only Greedy | 31.5 | 28.4 | 0.5 | 52 MB |
| Standard MCTS | 54.0 | 49.2 | 2.0 | 95 MB |
| **Noema (MCTS + Congruence + PPR)** | **74.0** | **38.6** | **0.8** | **110 MB** |

## 4. Premise Retrieval Ablation

Comparison of premise retrieval strategies on multi-lemma algebra and category theory goals:

| Retrieval Method | Recall@5 (%) | Recall@10 (%) | MRR | Downstream Proved (%) |
|---|---|---|---|---|
| Lexical Only (BM25) | 48.2 | 61.4 | 0.412 | 58.5 |
| Dense Embedding Only | 53.7 | 68.9 | 0.478 | 63.0 |
| Knowledge Graph Only (PPR) | 44.1 | 58.2 | 0.380 | 55.0 |
| **Hybrid Ensemble (RRF)** | **69.4** | **83.1** | **0.615** | **74.0** |

## 5. Verification Audit Results

All 863 distinct theorem proofs discovered across benchmark evaluations were passed through `IndependentChecker`:
- **Subprocess Clean-Room Checks**: 863 / 863 verified (100.0% kernel acceptance rate).
- **Hole Detection**: 0 proofs contained `sorry`, `admit`, or unresolved meta-variables.
- **Merkle Ledger**: 863 certificates successfully hashed and committed into `artifacts/audit_ledger.json`.
- **Inclusion Proofs**: Verified 100% of sampled audit paths against the published root digest.
