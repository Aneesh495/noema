"""Formal environment and premise scope contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from pydantic import BaseModel, Field


class PremiseDecl(BaseModel):
    """A formal premise (theorem, lemma, definition, axiom) available in the environment."""
    name: str
    type_str: str
    module: str
    is_axiom: bool = False
    source_line: int | None = None
    docstring: str | None = None

    def content_hash(self) -> str:
        s = f"{self.name}::{self.type_str}::{self.module}::{self.is_axiom}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class EnvironmentSpec(BaseModel):
    """Specification of a pinned formal verification environment."""
    toolchain: str = Field(..., description="Lean 4 toolchain string, e.g. leanprover/lean4:v4.8.0")
    lakefile_path: str = "lean/lakefile.lean"
    imported_modules: list[str] = Field(default_factory=list)
    declaration_prefix: str = Field("", description="Only declarations prior to this prefix are eligible")
    allowed_axioms: list[str] = Field(
        default_factory=lambda: ["propext", "Classical.choice", "Quot.sound"],
        description="Standard allowed classical Lean 4 axioms"
    )
    eligible_premises: dict[str, PremiseDecl] = Field(
        default_factory=dict,
        description="Map of eligible premise name to premise declaration"
    )

    def environment_hash(self) -> str:
        data = {
            "toolchain": self.toolchain,
            "imports": sorted(self.imported_modules),
            "prefix": self.declaration_prefix,
            "allowed_axioms": sorted(self.allowed_axioms),
            "num_premises": len(self.eligible_premises),
        }
        raw = json.dumps(data, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def is_premise_eligible(self, name: str, target_name: str | None = None) -> bool:
        """Verify whether a premise is eligible for use in proof search."""
        if target_name and name == target_name:
            return False
        if target_name and name.startswith(f"{target_name}."):
            return False
        return name in self.eligible_premises
