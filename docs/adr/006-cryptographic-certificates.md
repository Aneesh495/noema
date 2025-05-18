# ADR-006: Cryptographic Merkle Trees and Tamper-Evident Proof Certificates

## Context
Neural theorem provers must provide verifiable evidence that claims of proven theorems are valid and uncorrupted. Storing proof scripts as loose text files creates risks of accidental modification, incomplete tracking of environment dependencies, or undetectable silent failures. Furthermore, auditing benchmark results requires proving that an individual theorem was included in a specific benchmark run without re-evaluating the entire test suite.

## Decision
Noema implements cryptographic proof certificates audited via Merkle trees:
1. Proof Certificates (`ProofCertificate`):
   - Cryptographically hashes the formal theorem statement: $H(\text{module} \parallel \text{name} \parallel \text{statement})$.
   - Records the exact Lean 4 toolchain version, Lake configuration hash, and allowed axiom set.
   - Enforces zero holes (`sorry`, `admit`, `unsafe`).
2. Merkle Audit Ledger (`MerkleAuditTree`):
   - Computes SHA-256 leaf hashes for every generated proof certificate.
   - Builds a balanced binary Merkle tree yielding an immutable root hash for the acceptance bundle.
   - Produces compact $O(\log N)$ Merkle inclusion proofs (`MerkleInclusionProof`) for individual theorem verification.

## Consequences
- Independent evaluators can verify individual theorems against the published Merkle root in milliseconds.
- Any modification to a tactic script, environment specification, or theorem statement invalidates the cryptographic inclusion proof.
