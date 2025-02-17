"""Certificate management, replay file generation, and poisoned-cache rejection."""

from __future__ import annotations

import json
from pathlib import Path
from noema.artifacts.axiom_auditor import AxiomAuditor
from noema.contracts.certificates import CertificateAudit, ProofCertificate
from noema.contracts.theorems import TheoremSpec


class CertificateManager:
    """Manages serialization, validation, and fresh-process replay of proof certificates."""

    def __init__(self, artifacts_dir: Path | str = "artifacts/proofs"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.auditor = AxiomAuditor()

    def save_certificate(self, certificate: ProofCertificate) -> Path:
        """Seal and write proof certificate and standalone Lean replay script."""
        certificate.seal()
        out_json = self.artifacts_dir / f"{certificate.theorem_id}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            f.write(certificate.model_dump_json(indent=2))

        # Also emit clean standalone replay file
        out_lean = self.artifacts_dir / f"{certificate.theorem_id}_replay.lean"
        with open(out_lean, "w", encoding="utf-8") as f:
            lines = [f"  {l.strip()}" for l in certificate.tactic_script.strip().splitlines() if l.strip()]
            indented = "\n".join(lines)
            f.write(
                f"-- Standalone Lean 4 Replay Certificate for {certificate.theorem_name}\n"
                f"-- Theorem ID: {certificate.theorem_id}\n"
                f"-- Certificate Hash: {certificate.certificate_hash}\n\n"
                f"theorem {certificate.theorem_name} : {certificate.statement} := by\n"
                f"{indented}\n\n"
                f"#print axioms {certificate.theorem_name}\n"
            )

        return out_json

    def load_and_verify(self, path: Path | str) -> tuple[ProofCertificate, CertificateAudit]:
        """Load certificate and strictly reject poisoned or tampered files."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        cert = ProofCertificate(**data)

        # Audit certificate integrity and axioms
        audit = self.auditor.audit_certificate(cert)
        return cert, audit
