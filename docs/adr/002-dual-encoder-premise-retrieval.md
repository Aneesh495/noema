# ADR-002: Dual-Encoder and Lexical Hybrid Premise Retrieval

## Context
Formal mathematical libraries such as Mathlib contain tens of thousands of lemmas. Passing the entire corpus directly into the context window of an autoregressive model exceeds compute budgets and degrades attention quality. Pure lexical search (such as BM25) fails when theorems use synonyms or abstract notation (for instance, searching for group cancellation when the lemma is named `mul_left_cancel`). Conversely, pure dense retrieval struggles on exact identifier matching where single character alterations change mathematical meaning (such as `Nat.add_comm` versus `Nat.mul_comm`).

## Decision
Noema implements a two-stage hybrid retrieval architecture with cross-encoder reranking:
1. Candidate Generation:
   - Dense Dual-Encoder: Maps goal states and candidate premises into a shared 256-dimensional embedding space using cosine similarity.
   - Lexical Inverted Index: BM25 score computed over mathematical tokens and identifier fragments.
   - Reciprocal Rank Fusion (RRF): Combines ranks with $RRF(d) = \sum \frac{1}{60 + r_i(d)}$.
2. Structural Knowledge Filtering:
   - Personalized PageRank over the premise dependency DAG prunes circular premises and rewards foundational centrality.
3. Cross-Encoder Reranker:
   - Full cross-attention over joint pairs `[CLS] goal [SEP] candidate_premise [SEP]` scores the top-64 candidates with calibrated temperature scaling.

## Consequences
- Achieves high recall on conceptual premises while retaining exact-match precision on named identifiers.
- Guarantees zero lookahead or circular premise leakage by topologically auditing the dependency graph prior to score accumulation.
