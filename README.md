# Noema: Deep Neural Search for Checked Mathematical Proofs

Noema is an original deep neural theorem prover combining trained premise retrieval, a 24-block policy and value Transformer model, proof-hypergraph search with coupled metavariables, and independent proof checking in Lean 4.

## Overview

Automated theorem proving requires searching through vast state spaces of mathematical goals and tactical choices. Noema investigates whether joint training of tactic prediction, eligible premise retrieval, and proof progress estimation enables more efficient proof search under constrained computational and formal transition budgets.

### Architecture

```
+-------------------------------------------------------------+
|                      Formal Environment                     |
|  (Lean 4 Kernel, Pinned Mathlib Declarations, Frozen Scope) |
+------------------------------+------------------------------+
                               |
                               v
                     +--------------------+
                     |  Current Proof Goal |
                     +---------+----------+
                               |
         +---------------------+---------------------+
         |                                           |
         v                                           v
+------------------+                       +-------------------+
| Lexical / Dense  |                       | Goal Tokenization |
| Premise Retriever|                       | & Hypothesis View |
+--------+---------+                       +---------+---------+
         |                                           |
         +---------------------+---------------------+
                               |
                               v
               +-------------------------------+
               |   Noema 24-Block Transformer  |
               |  (Joint Policy, Pointer, Val) |
               +---------------+---------------+
                               |
         +---------------------+---------------------+
         |                     |                     |
         v                     v                     v
+-----------------+   +-----------------+   +-----------------+
| Tactic Proposal |   | Premise Pointer |   | Value / Progress|
| Distribution    |   | Selection       |   | Likelihood      |
+--------+--------+   +--------+--------+   +--------+--------+
         |                     |                     |
         +---------------------+---------------------+
                               |
                               v
               +-------------------------------+
               |     Proof-Hypergraph Search   |
               |  (Metavariable AND/OR Engine) |
               +---------------+---------------+
                               |
                               v
               +-------------------------------+
               |    Lean 4 Bridge & Checker    |
               | (Fresh Process Kernel Replay) |
               +-------------------------------+
```

## Setup and Quick Start

### Prerequisites
- Python 3.12+
- PyTorch 2.2+ (CPU or Apple Silicon MPS / CUDA)
- Lean 4 (version 4.8.0) and Lake (managed via elan)

### Quick Commands

```bash
# Verify environment and dependencies
make doctor

# Run test suite
make test

# Run bounded end-to-end demo
make demo
```
