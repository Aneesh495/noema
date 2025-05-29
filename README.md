# Noema: Deep Neural Search for Checked Mathematical Proofs

Noema is an original deep neural theorem prover combining trained premise retrieval, a 24-block policy and value Transformer model, proof-hypergraph search with coupled metavariables, and independent proof checking in Lean 4.

## System Highlights

- **Flagship Neural Guidance**: 24 stacked Transformer blocks (60,128,673 trainable parameters) with RoPE embeddings, RMSNorm, SwiGLU activations, and multi-task heads for tactic policy prediction, state value scoring, and censored proof progress estimation.
- **Hybrid Premise Retrieval**: Unifies BM25 lexical matching, dense semantic embeddings, and Personalized PageRank (PPR) over formal premise dependency graphs using Reciprocal Rank Fusion.
- **Proof-Hypergraph Search**: Monte Carlo Tree Search (MCTS) with PUCT exploration adapted for AND/OR hypergraphs, tracking disjunctive goals and conjunctive multi-goal tactics.
- **Symbolic Decision Engines**: Integrated explainable congruence closure, Knuth-Bendix term rewriting completion, and propositional semantic tableau solvers.
- **Formal Verification Environment**: Persistent Lean 4.8.0 REPL protocol over bidirectional NDJSON pipes with state snapshots and environment caching.
- **Cryptographic Audit Ledger**: Strict distrust of neural suggestions. All candidate proofs are re-checked from scratch by isolated subprocesses (`lake env lean`) and sealed into an append-only SHA-256 Merkle audit tree with verifiable inclusion proofs.

---

## Architecture Overview

```
+-------------------------------------------------------------------------+
|                           User / CLI Interface                          |
|             Interactive Shell (REPL), Benchmark Runner, CLI             |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         Proof Search Subsystem                          |
|   MCTS on AND/OR Hypergraphs | Diverse Beam Search | Tableau Prover     |
|   Congruence Closure Engine  | Term Rewriting Completion (Knuth-Bendix) |
+------------------+----------------------------------+-------------------+
                   |                                  |
                   v                                  v
+------------------------------------+  +---------------------------------+
|     Neural Guidance Subsystem      |  |   Lean 4 Formal Environment     |
| 24-Block Transformer (60.1M params)|  | Persistent Subprocess REPL      |
| - Tactic Policy Head               |  | Lake 5.0.0 Build Toolchain      |
| - State Value Head                 |  | Lean 4.8.0 Kernel & Elaborator  |
| - Censored Progress Head           |  +----------------+----------------+
| Hybrid Premise Retriever (BM25+PPR)|                   |
+------------------------------------+                   v
                                        +---------------------------------+
                                        |    Cryptographic Audit Store    |
                                        | Fresh Process Subprocess Checker|
                                        | Sealed Proof Certificates       |
                                        | Append-Only Merkle Tree Ledger  |
+-------------------------------------------------------------------------+
```

Detailed architectural diagrams are available in [`docs/diagrams/`](docs/diagrams/):
- [01: System Overview](docs/diagrams/01-system-overview.mmd)
- [02: AND/OR Proof Hypergraphs](docs/diagrams/02-and-or-hypergraph.mmd)
- [03: 24-Block Transformer Architecture](docs/diagrams/03-transformer-24-block.mmd)
- [04: Hybrid Premise Retrieval](docs/diagrams/04-premise-retrieval.mmd)
- [05: Persistent Lean 4 REPL Protocol](docs/diagrams/05-repl-protocol.mmd)
- [06: Explainable Congruence Closure](docs/diagrams/06-congruence-closure.mmd)
- [07: Cryptographic Merkle Audit Tree](docs/diagrams/07-merkle-audit-tree.mmd)
- [08: Curriculum Self-Play Pipeline](docs/diagrams/08-curriculum-self-play.mmd)
- [09: Diverse Beam Search](docs/diagrams/09-diverse-beam-search.mmd)
- [10: PPO Policy & Value Training](docs/diagrams/10-ppo-training-pipeline.mmd)

---

## Formal Evaluation Results

Evaluated across 1,200 curated formal theorems across 8 mathematical domains under a budget of 200 REPL expansions per trial:

| Mathematical Domain | Dataset | Pass@1 (%) | Pass@8 (%) | Pass@32 (%) | Solved Expansions | Verification |
|---|---|---|---|---|---|---|
| Propositional Logic | `logic_core` | 64.5 | 82.0 | 91.5 | 14.2 | 100% Kernel Verified |
| Abstract Algebra | `algebra_core` | 42.0 | 61.5 | 74.0 | 38.6 | 100% Kernel Verified |
| Order Theory | `order_core` | 51.3 | 70.7 | 82.0 | 22.4 | 100% Kernel Verified |
| Category Theory | `category_core` | 38.7 | 56.0 | 68.7 | 45.1 | 100% Kernel Verified |
| Modular Arithmetic | `modular_core` | 34.0 | 52.7 | 66.0 | 52.8 | 100% Kernel Verified |
| Set Theory | `set_theory_core` | 58.0 | 76.7 | 86.7 | 18.9 | 100% Kernel Verified |
| General Topology | `topology_core` | 31.0 | 49.0 | 62.0 | 64.3 | 100% Kernel Verified |
| Depth Extrapolation | `extrapolation_deep` | 18.0 | 32.0 | 44.0 | 112.5 | 100% Kernel Verified |
| **Overall Aggregate** | **All 8 Domains** | **42.2** | **60.1** | **71.9** | **46.1** | **863 / 863 Clean (0 sorry)** |

Detailed experiment configurations and ablations are documented in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Installation & Setup

### Prerequisites
- Python 3.12+
- PyTorch 2.2+ (CPU, Apple Silicon MPS, or NVIDIA CUDA)
- Lean 4 toolchain `leanprover/lean4:v4.8.0` and Lake 5.0.0 (installed via `elan`)

### Environment Initialization

```bash
# Clone repository
git clone https://github.com/Aneesh495/noema.git
cd noema

# Setup Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

# Build Lean 4 formal bridge package
cd lean
lake build
cd ..
```

---

## Quickstart & CLI Reference

Verify the local toolchain and environment:

```bash
python3 -m noema.cli doctor
```

Execute the bounded end-to-end trained model demonstration:

```bash
python3 -m noema.cli demo
```

Launch the interactive proof shell:

```bash
python3 -m noema.cli shell --theorem "theorem demo (P Q : Prop) : P ∧ Q → Q ∧ P := by"
```

Generate synthetic mathematical corpora across domains:

```bash
python3 -m noema.cli data --output data/ --num-train 20000 --num-eval 2000
```

Run latency, throughput, and memory profiling:

```bash
python3 -m noema.cli bench
```

Generate source-bound acceptance evidence:

```bash
python3 -m noema.cli acceptance
```

Validate an evidence bundle independently without retraining:

```bash
python3 -m noema.cli verify --evidence runs/evidence/
```

Run test suite:

```bash
pytest
```

---

## Technical Documentation & Guides

- [System Architecture](docs/ARCHITECTURE.md): Comprehensive design of models, search, and retrieval
- [Mathematical Derivations](docs/MATHEMATICAL_DERIVATIONS.md): PUCT bounds, GAE, SwiGLU, and survival objectives
- [Reproducibility Guide](docs/REPRODUCIBILITY.md): Exact hardware environments, seeds, and execution steps
- [Experiment Plan](docs/EXPERIMENT_PLAN.md): Research questions, baseline conditions, and evaluation protocol
- [Benchmark Results](docs/BENCHMARKS.md): Empirical pass@k rates, search statistics, and ablation tables
- [Walkthrough 1: Interactive REPL Session](docs/walkthroughs/01-repl-interactive-session.md)
- [Walkthrough 2: Training the 24-Block Model](docs/walkthroughs/02-training-24-block-model.md)
- [Walkthrough 3: Proof Search with MCTS & Hypergraphs](docs/walkthroughs/03-mcts-hypergraph-proving.md)
- [Walkthrough 4: Cryptographic Auditing & Merkle Ledger](docs/walkthroughs/04-certificate-merkle-auditing.md)
- [Architectural Decision Records (ADR 001 - 008)](docs/adr/): Formal record of key engineering choices
