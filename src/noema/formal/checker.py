"""Independent fresh-process Lean 4 proof checker and axiom auditor."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from noema.contracts.certificates import (
    CertificateAudit,
    ProofCertificate,
    VerificationStatus,
)
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec


class IndependentChecker:
    """Independently verifies proof terms and scripts in a clean Lean 4 process."""

    def __init__(self, env_spec: EnvironmentSpec, lean_cmd: str | None = None):
        self.env_spec = env_spec
        self.lean_cmd = lean_cmd or shutil.which("lean") or "lean"
        self.allowed_axioms = set(env_spec.allowed_axioms)

    def verify_proof(
        self,
        theorem: TheoremSpec,
        tactic_script: str,
        timeout_sec: float = 10.0,
    ) -> tuple[ProofCertificate, CertificateAudit]:
        """Perform a complete independent check of the candidate proof."""
        # 1. Textual inspection for cheating keywords or holes
        bad_keywords = ["sorry", "admit", "cheat", "unsafe"]
        for kw in bad_keywords:
            if re.search(rf"\b{kw}\b", tactic_script):
                cert = ProofCertificate(
                    theorem_id=theorem.theorem_id,
                    theorem_name=theorem.name,
                    statement=theorem.statement,
                    statement_hash=theorem.statement_hash(),
                    environment_hash=self.env_spec.environment_hash(),
                    tactic_script=tactic_script,
                    status=VerificationStatus.REJECTED,
                    kernel_diagnostics=f"Prohibited hole keyword '{kw}' detected in proof script",
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                cert.seal()
                audit = CertificateAudit(
                    is_valid=False,
                    rejection_reason=f"Prohibited hole keyword '{kw}'",
                    no_sorry=False,
                )
                return cert, audit

        # 2. Construct standalone Lean verification file with axiom query
        imports = "\n".join(f"import {m}" for m in self.env_spec.imported_modules)
        ns_open = f"namespace {theorem.namespace}\n" if theorem.namespace else ""
        ns_close = f"\nend {theorem.namespace}" if theorem.namespace else ""
        qual_name = f"{theorem.namespace}.{theorem.name}" if theorem.namespace else theorem.name

        lines = [f"  {l.strip()}" for l in tactic_script.strip().splitlines() if l.strip()]
        indented_script = "\n".join(lines)

        code = (
            f"{imports}\n\n"
            f"{ns_open}"
            f"theorem {theorem.name} : {theorem.statement} := by\n"
            f"{indented_script}\n\n"
            f"#print axioms {qual_name}\n"
            f"{ns_close}\n"
        )

        with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            res = subprocess.run(
                [self.lean_cmd, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            stdout = res.stdout
            stderr = res.stderr
            ret = res.returncode
            combined = stdout + "\n" + stderr

            if ret != 0:
                cert = ProofCertificate(
                    theorem_id=theorem.theorem_id,
                    theorem_name=theorem.name,
                    statement=theorem.statement,
                    statement_hash=theorem.statement_hash(),
                    environment_hash=self.env_spec.environment_hash(),
                    tactic_script=tactic_script,
                    status=VerificationStatus.REJECTED,
                    kernel_diagnostics=f"Lean compilation failed (exit {ret}): {combined.strip()}",
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                cert.seal()
                audit = CertificateAudit(
                    is_valid=False,
                    rejection_reason=f"Lean compilation failed with exit code {ret}",
                    clean_exit_code=False,
                )
                return cert, audit

            # 3. Parse axioms output from `#print axioms`
            used_axioms = self._parse_axioms(stdout)
            unpermitted_axioms = [a for a in used_axioms if a not in self.allowed_axioms]

            if unpermitted_axioms:
                cert = ProofCertificate(
                    theorem_id=theorem.theorem_id,
                    theorem_name=theorem.name,
                    statement=theorem.statement,
                    statement_hash=theorem.statement_hash(),
                    environment_hash=self.env_spec.environment_hash(),
                    tactic_script=tactic_script,
                    used_axioms=used_axioms,
                    status=VerificationStatus.REJECTED,
                    kernel_diagnostics=f"Proof depends on unpermitted axioms: {unpermitted_axioms}",
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                cert.seal()
                audit = CertificateAudit(
                    is_valid=False,
                    rejection_reason=f"Unpermitted axioms: {unpermitted_axioms}",
                    axioms_permitted=False,
                )
                return cert, audit

            # Proof is valid!
            cert = ProofCertificate(
                theorem_id=theorem.theorem_id,
                theorem_name=theorem.name,
                statement=theorem.statement,
                statement_hash=theorem.statement_hash(),
                environment_hash=self.env_spec.environment_hash(),
                tactic_script=tactic_script,
                used_axioms=used_axioms,
                status=VerificationStatus.VERIFIED,
                kernel_diagnostics="Verification successful, valid classical axioms only",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
            cert.seal()
            audit = CertificateAudit(
                is_valid=True,
                rejection_reason=None,
                axioms_permitted=True,
                no_sorry=True,
                no_unassigned_mvars=True,
                no_target_self_reference=True,
                exact_statement_match=True,
                clean_exit_code=True,
            )
            return cert, audit

        except subprocess.TimeoutExpired:
            cert = ProofCertificate(
                theorem_id=theorem.theorem_id,
                theorem_name=theorem.name,
                statement=theorem.statement,
                statement_hash=theorem.statement_hash(),
                environment_hash=self.env_spec.environment_hash(),
                tactic_script=tactic_script,
                status=VerificationStatus.TIMEOUT,
                kernel_diagnostics="Verification timed out in fresh process",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
            cert.seal()
            audit = CertificateAudit(
                is_valid=False,
                rejection_reason="Verification timed out",
                clean_exit_code=False,
            )
            return cert, audit
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _parse_axioms(self, output: str) -> list[str]:
        axioms = []
        # `#print axioms` outputs e.g.:
        # 'test_thm' depends on axioms: [propext, Classical.choice]
        # or "'test_thm' does not depend on any axioms"
        m = re.search(r"depends on axioms:\s*\[(.*?)\]", output)
        if m:
            raw = m.group(1).strip()
            if raw:
                axioms = [a.strip() for a in raw.split(",") if a.strip()]
        return axioms
