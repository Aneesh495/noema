# Noema Architecture

Noema is an automated neural theorem prover for Lean 4. The system couples learned tactic policy distributions, dense and knowledge-graph premise retrieval, and Monte Carlo tree search over AND/OR proof hypergraphs with independent clean-room verification.

## 1. System Decomposition

The architecture is divided into five cooperating subsystems:

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

## 2. Neural Architecture: 24-Block Transformer

The flagship neural model is an autoregressive Transformer comprising 24 stacked blocks, parameterized as follows:

- **Layer count**: 24 Transformer blocks
- **Model dimension ($d_{\text{model}}$)**: 512
- **Feedforward dimension ($d_{\text{ff}}$)**: 2048 using SwiGLU non-linear gating
- **Attention heads**: 8 heads with head dimension 64
- **Position encoding**: Rotary Position Embeddings (RoPE)
- **Normalization**: Pre-layer RMSNorm with learned scale parameters
- **Vocabulary**: 16,384 tokens covering Lean 4 syntax, identifiers, and unicode mathematical symbols
- **Parameter total**: 60,128,673 trainable weights without weight tying

### Prediction Heads

1. **Tactic Policy Head**: Autoregressive token generation producing candidate tactics conditioned on serialized goal state and retrieved premise tokens.
2. **State Value Head**: Two-layer MLP with tanh and sigmoid activations projecting sequence representations to a scalar proving probability $V(s) \in [0, 1]$.
3. **Censored Progress Head**: Predicts expected remaining proof distance with a discrete survival distribution, accounting for right-censored search timeouts.

## 3. Hybrid Premise Retrieval

Premise selection combines three orthogonal ranking strategies:

1. **Lexical Matcher**: BM25 scoring over tokenized theorem signatures, definitions, and local hypothesis identifiers.
2. **Dense Semantic Embedder**: Cosine similarity in a 512-dimensional shared embedding space fine-tuned on goal-premise co-occurrence.
3. **Knowledge Graph Centrality**: Personalized PageRank (PPR) evaluated on the formal premise dependency DAG, propagating relevance scores from local hypotheses to prerequisite lemmas.

Rankings are unified through Reciprocal Rank Fusion (RRF):

$$RRF(p) = \sum_{m \in \{\text{BM25}, \text{Dense}, \text{PPR}\}} \frac{w_m}{k + \text{rank}_m(p)}$$

## 4. Search Engines

### 4.1. AND/OR Proof Hypergraphs and MCTS

Formal proofs require decomposing a conjecture into multiple subgoals. The search state is structured as an AND/OR hypergraph:
- **OR-Nodes**: Represent open goals. Solved if any outgoing hyperedge is closed.
- **AND-Hyperedges**: Represent multi-goal tactics. Solved if and only if all resulting subgoals are closed.

MCTS explores this hypergraph using an adapted Predictor Upper Confidence bound for Trees (PUCT) variant that propagates conservative lower bounds through conjunctive hyperedges.

### 4.2. Algorithmic Decision Procedures

For specialized sub-problems, Noema invokes fast symbolic procedures:
- **Congruence Closure**: Computes the reflexive, symmetric, transitive, and congruent closure of ground equality hypotheses, generating explainable Lean 4 tactic chains.
- **Term Rewriting**: Knuth-Bendix completion and directed term reduction for monoids, groups, and lattices.
- **Semantic Tableau**: Analytic tableau calculus for propositional resolution and countermodel generation.

## 5. Formal REPL and Independent Verification

Noema communicates with Lean 4 via a persistent bidirectional NDJSON protocol over stdin/stdout, maintaining incremental environment states without restart overhead.

Crucially, Noema enforces an untrusted model boundary:
- All proof candidates found in REPL sessions are re-verified by an independent `IndependentChecker`.
- The checker executes a fresh, isolated `lake env lean` process on the serialized proof script from scratch.
- Verified proofs are recorded in an append-only Merkle ledger with SHA-256 digests.
