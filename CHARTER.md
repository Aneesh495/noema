# Noema Project Charter

## Scientific Identity and Core Questions

Noema is an original deep neural theorem prover that investigates learned mathematical proof construction coupled to the Lean 4 formal verification kernel.

The primary research questions are:
1. Does joint learning of premise retrieval, tactic generation, and proof-progress estimation improve verified search success under identical formal-transition and wall-clock budgets compared to decoupled or heuristic baselines?
2. How do model depth (specifically scaling from 4 to 12 to 24 independently parameterized Transformer blocks) and verified expert-iteration data affect search efficiency and generalization across novel theorem families and unseen premises?

## Boundaries and Principles

1. **Formal Correctness as a Hard Semantic Boundary**: A neural model score is a heuristic guide, not a proof. Only proofs compiled and verified by an independent Lean 4 kernel with explicitly declared axioms count as valid solutions.
2. **Real Neural Mechanisms**: The primary model is trained from random initialization with 24 distinct Transformer blocks and over 40 million parameters. Noema does not wrap external language model APIs or delegate proof synthesis to third-party automation.
3. **No Contamination**: Target theorem declarations, proof bodies, and downstream dependency closures are strictly excluded from the eligible premise environment and training supervision.
4. **Reproducibility**: All experiments, seeds, formal transitions, and certificates are tracked with immutable content hashes and deterministic contracts.
