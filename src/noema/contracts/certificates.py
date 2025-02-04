"""Proof certificate and independent verification contracts."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"
    TIMEOUT = "timeout"


class ProofCertificate(BaseModel):
    """Tamper-evident proof certificate for independently verified theorems."""
    theorem_id: str
    theorem_name: str
    statement: str
    statement_hash: str
    environment_hash: str
    tactic_script: str
    proof_term: str | None = None
    used_axioms: list[str] = Field(default_factory=list)
    used_premises: list[str] = Field(default_factory=list)
    search_transitions: int = 0
    search_time_sec: float = 0.0
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    checker_version: str = "lean4:v4.8.0"
    kernel_diagnostics: str = ""
    timestamp_utc: str = ""
    certificate_hash: str = ""

    def compute_hash(self) -> str:
        data = {
            "theorem_id": self.theorem_id,
            "statement_hash": self.statement_hash,
            "environment_hash": self.environment_hash,
            "tactic_script": self.tactic_script.strip(),
            "proof_term": self.proof_term or "",
            "used_axioms": sorted(self.used_axioms),
            "used_premises": sorted(self.used_premises),
            "status": self.status.value,
            "checker_version": self.checker_version,
        }
        raw = json.dumps(data, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def seal(self) -> None:
        """Compute and set the immutable certificate hash."""
        self.certificate_hash = self.compute_hash()

    def verify_integrity(self) -> bool:
        """Verify that the certificate hash matches its content."""
        if not self.certificate_hash:
            return False
        return self.compute_hash() == self.certificate_hash


class CertificateAudit(BaseModel):
    """Result of an independent audit on a candidate certificate."""
    is_valid: bool
    rejection_reason: str | None = None
    axioms_permitted: bool = True
    no_sorry: bool = True
    no_unassigned_mvars: bool = True
    no_target_self_reference: bool = True
    exact_statement_match: bool = True
    clean_exit_code: bool = True
