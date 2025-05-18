# ADR-007: Syntactic and Normalized Contamination Auditing

## Context
A recurring challenge in machine learning for mathematics is data leakage between training corpora and evaluation benchmarks. If evaluation theorems or trivial renamings appear in the training split, evaluation accuracy is artificially inflated and fails to measure genuine generalization. Simple string equality checks are insufficient because alpha-equivalence (renaming variables $x$ to $y$) or whitespace variation can mask identical theorems.

## Decision
Noema implements multi-stage contamination auditing:
1. Exact Syntactic Matching: Checks exact statement hashes across train, validation, and test sets.
2. Alpha-Normalized Equivalence: Strips whitespace, normalizes variable identifiers to canonical de Bruijn indices ($v_0, v_1, \dots$), and maps proposition identifiers canonically.
3. n-Gram Jaccard Substring Overlap: Computes token 3-gram and 5-gram Jaccard similarity to flag near-duplicate theorem statements.
4. Family and Domain Segregation: Enforces strict isolation between synthetic arithmetic, order theory, inductive data types, and topology splits.

## Consequences
- Guarantees that evaluation benchmarks reflect genuine out-of-distribution reasoning rather than memorization.
- Automatically rejects training batches containing theorems that match evaluation splits under alpha-normalization.
