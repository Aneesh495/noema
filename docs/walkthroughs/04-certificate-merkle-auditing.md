# Walkthrough 4: Cryptographic Auditing and Proof Certificates

This walkthrough covers how Noema certifies discovered proofs and records them in an immutable, tamper-evident cryptographic Merkle ledger.

## Philosophy of Independent Verification

Neural networks can produce hallucinatory outputs, and interactive REPL environments can accumulate stale global state. Noema enforces strict verification boundaries:
- Proofs discovered during search are treated as untrusted candidates.
- A fresh, isolated subprocess (`lake env lean`) evaluates the complete generated script from scratch in an empty environment.
- Any theorem containing `sorry`, `admit`, or non-zero kernel exit codes is immediately rejected.

## Generating a Proof Certificate

Once a proof passes independent checking, Noema packages it into a cryptographically sealed `ProofCertificate`:

```python
from noema.formal.checker import IndependentChecker
from noema.artifacts.certificates import CertificateManager

checker = IndependentChecker(project_dir="lean")
cert_manager = CertificateManager(storage_dir="artifacts/certificates")

theorem_name = "Logic.and_comm_formal"
proof_script = """
import NoemaBridge.Logic

theorem and_comm_formal (P Q : Prop) : P ∧ Q → Q ∧ P := by
  intro h
  constructor
  · exact h.2
  · exact h.1
"""

# Independent verification in isolated process
verification_result = checker.verify_proof(proof_script)
assert verification_result.success, "Verification failed in clean-room environment."

# Create cryptographically signed certificate
certificate = cert_manager.create_certificate(
    theorem_name=theorem_name,
    proof_script=proof_script,
    verification_hash=verification_result.environment_digest,
    execution_time_ms=verification_result.duration_ms,
)

print(f"Certificate ID: {certificate.cert_id}")
print(f"Certificate Hash: {certificate.sha256_hash}")
```

## Recording into the Merkle Audit Ledger

Certificates are appended to an append-only cryptographic Merkle tree:

```python
from noema.artifacts.merkle import MerkleLedger

ledger = MerkleLedger(storage_path="artifacts/audit_ledger.json")

# Append newly certified proof
leaf_index = ledger.append_certificate(certificate)
print(f"Appended at leaf index: {leaf_index}")
print(f"Current Merkle Root: {ledger.get_root_hash()}")

# Generate inclusion audit proof
audit_proof = ledger.generate_inclusion_proof(leaf_index)
print(f"Audit path length: {len(audit_proof.sibling_hashes)}")

# Verify inclusion proof independently
is_valid = ledger.verify_inclusion(certificate.sha256_hash, audit_proof)
print(f"Audit proof valid: {is_valid}")
```

## Negative Control: Tampering Detection

To confirm that tampering is reliably caught, modify a single character in the certified script:

```python
tampered_hash = cert_manager.compute_hash(
    theorem_name=theorem_name,
    proof_script=proof_script + " ", # Injected space
    environment_digest=verification_result.environment_digest,
)

is_tampered_valid = ledger.verify_inclusion(tampered_hash, audit_proof)
print(f"Tampered proof accepted: {is_tampered_valid}")
# Expected output: False
```
