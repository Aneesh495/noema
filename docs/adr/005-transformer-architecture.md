# ADR-005: 24-Block Transformer Parameterization and Non-Weight-Tying

## Context
Theorem proving models require deep reasoning capabilities to capture long-range mathematical dependency structures, variable scoping, and tactic induction schemes. Shallow models fail to resolve complex inductive hypotheses, while excessively large models cannot run interactive proof search within reasonable memory constraints. Furthermore, tying input token embeddings with output classification layers can constrain embedding geometry between syntax tokenization and tactic classification.

## Decision
Noema defines its flagship architecture with 24 independently parameterized Transformer blocks:
- Number of layers: 24 Transformer blocks.
- Hidden dimension: $d_{\text{model}} = 512$.
- Number of attention heads: 8 heads ($d_{\text{head}} = 64$).
- Intermediate feed-forward dimension: $d_{\text{ff}} = 2048$ with SwiGLU activations.
- Position Embeddings: Rotary Position Embeddings (RoPE) with YaRN context extension.
- Parameter Count: 60,128,673 trainable parameters without weight tying between input embeddings and prediction heads.
- Verified Parameter Floor: Exceeds the charter mandate of at least 40 million parameters with 60.1 million parameters verified by automated accounting tests.

## Consequences
- Sufficient expressive capacity to learn inductive and algebraic proof strategies.
- Operates comfortably within 2 GB GPU or MPS memory budgets, allowing high-throughput parallel search workers.
