# Noema Build and Implementation Status

## Project Metadata
- Repository: `Aneesh495/noema-prover`
- Pinned Toolchain: Lean 4.8.0, Python 3.12, PyTorch 2.14.1
- Hardware Target: Apple M2 (Apple Silicon MPS / CPU)

## Phases and Progress

| Phase | Description | Status | Commit / Notes |
|---|---|---|---|
| Phase 1 | Repository initialization, contracts, build skeleton, git remote | Passed | Commit cc1e4df & contracts implemented and tested |
| Phase 2 | Real Lean 4 bridge, transition probe, lake package | Passed | Lake NoemaBridge package & Lean 4.8.0 formal bridge tested |
| Phase 3 | Corpus generation, trace extraction, contamination audit | Passed | TaskGenerator, public mathlib (300), miniF2F (244), and contamination auditor tested |
| Phase 4 | Math-aware tokenizer and nonpadding token accounting | Passed | MathTokenizer, TokenAccountant, BatchCollator tested |
| Phase 5 | Lexical and learned premise retrieval | Passed | BM25 LexicalRetriever, DenseDualEncoder, HardNegativeSampler, PremiseIndex tested |
| Phase 6 | Original 24-block, 40M+ Transformer model | Passed | 24 distinct blocks, >=40M trainable parameters verified, gradient flow tested |
| Phase 7 | Supervised training, joint loss, checkpointing | Passed | JointLoss, CheckpointManager, NoemaTrainer, ExpertIterationEngine tested |
| Phase 8 | Coupled-goal AND/OR proof-hypergraph search | Pending | - |
| Phase 9 | Independent fresh-process proof checker and certificate audit | Pending | - |
| Phase 10 | Training-only verified expert iteration | Pending | - |
| Phase 11 | Proof tree viewer and interactive demonstrations | Pending | - |
| Phase 12 | Research campaign, benchmarks, ablations, verification | Pending | - |

## Capability Register & Gates (NA01-NA12)

| Gate | Description | Status | Diagnostics |
|---|---|---|---|
| NA01 | 400 authored cases, 10,000 generated cases; numerical/loss checks | Pending | Unverified |
| NA02 | 100k state records, 10k theorems, 20k controlled tasks; 0 contamination | Pending | Unverified |
| NA03 | 24 distinct blocks, >=40M params, 3 seeds, 15M tokens per seed | Pending | Unverified |
| NA04 | Learned retrieval on 1,000 held-out state queries, recall@k | Pending | Unverified |
| NA05 | 2,000 controlled held-out tasks searched, >=500 checked solutions | Pending | Unverified |
| NA06 | 300 public held-out theorems searched, >=30 checked nontrivial | Pending | Unverified |
| NA07 | 244 pinned miniF2F test statements evaluated with full denominator | Pending | Unverified |
| NA08 | >=1M formal checks, >=100k valid transitions, zero target leakage | Pending | Unverified |
| NA09 | Depth (4/12/24), retrieval, value, curriculum ablations across 3 seeds | Pending | Unverified |
| NA10 | 1,000 perturbations, 100 invalid proof rejections | Pending | Unverified |
| NA11 | 50 interruption cases, 100 worker failure cases, 30-min stress run | Pending | Unverified |
| NA12 | Checked-proof demo, architecture docs, >=12 negative controls | Pending | Unverified |

## Current Executable Action
Initializing repository files, Makefile, contracts, and creating GitHub remote `Aneesh495/noema-prover`.
