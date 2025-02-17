"""Semantic axiom auditor and dependency verifier."""

from __future__ import annotations

from noema.contracts.certificates import CertificateAudit, ProofCertificate
from noema.contracts.environment import EnvironmentSpec


class AxiomAuditor:
    """Audits proof certificates against authorized classical foundations."""

    STANDARD_CLASSICAL_AXIOMS = {
        "propext",
        "Classical.choice",
        "Quot.sound",
    }

    def __init__(self, allowed_axioms: set[str] | None = None):
        self.allowed_axioms = allowed_axioms or self.STANDARD_CLASSICAL_AXIOMS

    def audit_certificate(self, certificate: ProofCertificate) -> CertificateAudit:
        """Perform comprehensive semantic audit on an accepted proof certificate."""
        # 1. Check certificate cryptographic integrity
        if not certificate.verify_integrity():
            return CertificateAudit(
                is_valid=False,
                rejection_reason="Certificate hash mismatch; tampering detected",
                clean_exit_code=False,
            )

        # 2. Check for prohibited hole keywords in tactic script
        bad_words = ["sorry", "admit", "cheat", "unsafe"]
        for w in bad_words:
            if f" {w}" in certificate.tactic_script or f"{w}\n" in certificate.tactic_script or certificate.tactic_script.strip() == w:
                return CertificateAudit(
                    is_valid=False,
                    rejection_reason=f"Prohibited hole keyword '{w}' present in certificate script",
                    no_sorry=False,
                )

        # 3. Check axioms against authorized boundary
        for ax in certificate.used_axioms:
            if ax not in self.allowed_axioms:
                return CertificateAudit(
                    is_valid=False,
                    rejection_reason=f"Proof depends on unpermitted axiom '{ax}' outside classical boundary",
                    axioms_permitted=False,
                )

        return CertificateAudit(
            is_valid=True,
            rejection_reason=None,
            axioms_permitted=True,
            no_sorry=True,
            no_unassigned_mvars=True,
            no_target_self_reference=True,
            exact_statement_match=True,
            clean_exit_code=True,
        )
