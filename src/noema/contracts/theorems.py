"""Theorem and task specification contracts."""

from __future__ import annotations

import hashlib
import json
from pydantic import BaseModel, Field


class TheoremSpec(BaseModel):
    """Specification of a mathematical theorem to be proved."""
    theorem_id: str
    name: str
    module: str
    statement: str
    namespace: str = ""
    family: str = Field("synthetic_arithmetic", description="Family category for split and contamination tracking")
    template_id: str | None = None
    parameters: dict[str, str] = Field(default_factory=dict, description="Generator parameters or variable types")
    reference_proof: str | None = Field(None, description="Compilable reference proof; hidden from inference worker")
    eligible_premise_names: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    def statement_hash(self) -> str:
        s = f"{self.module}::{self.name}::{self.statement.strip()}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def make_full_declaration(self, proof_body: str = "by sorry") -> str:
        """Construct full Lean 4 declaration code."""
        ns_open = f"namespace {self.namespace}\n\n" if self.namespace else ""
        ns_close = f"\n\nend {self.namespace}" if self.namespace else ""
        body = f"theorem {self.name} : {self.statement} := {proof_body}"
        return f"{ns_open}{body}{ns_close}"
