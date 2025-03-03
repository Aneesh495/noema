"""Tactic elaboration, proof term analysis, and syntactic verification utilities for Lean 4."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence
from noema.contracts.actions import ActionGrammar, TacticAction
from noema.contracts.theorems import TheoremSpec


@dataclass(frozen=True)
class ProofTermSummary:
    """Summary of an elaborated proof term and its constituent components."""
    theorem_name: str
    tactic_count: int
    premises_used: list[str]
    has_sorry: bool
    has_admit: bool
    is_axiom_free: bool
    syntactic_depth: int
    complexity_score: float
    raw_script: str


@dataclass
class TacticASTNode:
    """Node in an elaborated tactic AST."""
    tactic_text: str
    combinator: str | None = None
    children: list[TacticASTNode] = field(default_factory=list)
    premises: list[str] = field(default_factory=list)
    is_terminal: bool = False


class ElaborationError(Exception):
    """Raised when proof script elaboration or term parsing fails."""
    pass


class TacticElaborator:
    """Elaborates, decomposes, and analyzes tactical proof scripts into canonical structures."""

    # Disallowed escape keywords that undermine formal soundness
    _FORBIDDEN_TOKENS = {
        "sorry",
        "admit",
        "stop",
        "unsafe",
        "#exit",
        "#check",
        "#eval",
    }

    _PREMISE_EXTRACTOR = re.compile(r"\[([^\]]+)\]")
    _IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_.]*\b")

    @classmethod
    def decompose_script(cls, script_text: str) -> list[str]:
        """Decompose a multiline or semicolon-separated proof script into atomic tactic steps.
        
        Handles:
        - Semicolon sequencing: 'intro x; rfl' -> ['intro x', 'rfl']
        - Combinator '<;>': 'cases h <;> rfl'
        - Multi-line blocks and indentation stripping
        - Comment removal (-- line comments and /- -/ block comments)
        """
        cleaned = cls._strip_comments(script_text)
        raw_lines = cleaned.splitlines()
        atomic_steps: list[str] = []

        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Check for semicolon separation
            if ";" in stripped and "<;>" not in stripped:
                parts = [p.strip() for p in stripped.split(";") if p.strip()]
                atomic_steps.extend(parts)
            else:
                atomic_steps.append(stripped)

        return atomic_steps

    @classmethod
    def _strip_comments(cls, text: str) -> str:
        """Strip single-line and multi-line comments from Lean source text."""
        # Multi-line /- ... -/
        text = re.sub(r"/-(?:[^-]|-(?!/))*-/", "", text, flags=re.DOTALL)
        # Single-line -- ...
        lines = [re.sub(r"--.*$", "", line) for line in text.splitlines()]
        return "\n".join(lines)

    @classmethod
    def parse_tactic_tree(cls, script_text: str) -> list[TacticASTNode]:
        """Parse tactical steps into a list of structured AST nodes."""
        steps = cls.decompose_script(script_text)
        nodes: list[TacticASTNode] = []

        for step in steps:
            premises = cls.extract_premises(step)
            is_term = any(
                step.startswith(k)
                for k in ("rfl", "decide", "omega", "ring", "exact", "assumption", "trivial", "done")
            )
            node = TacticASTNode(
                tactic_text=step,
                premises=premises,
                is_terminal=is_term,
            )
            nodes.append(node)

        return nodes

    @classmethod
    def extract_premises(cls, tactic_text: str) -> list[str]:
        """Extract explicit theorem or lemma names referenced in a tactic invocation."""
        premises: set[str] = set()
        
        # 1. Bracketed lists in rw [a, b], simp only [a, b]
        bracket_matches = cls._PREMISE_EXTRACTOR.findall(tactic_text)
        for bm in bracket_matches:
            for item in bm.split(","):
                clean = item.strip().lstrip("←").strip()
                if clean and not clean.startswith("←") and cls._is_valid_identifier(clean):
                    premises.add(clean)

        # 2. Argument to apply, exact, have, refine
        words = tactic_text.strip().split()
        if len(words) >= 2 and words[0] in ("apply", "exact", "refine"):
            cand = words[1].strip()
            if cls._is_valid_identifier(cand):
                premises.add(cand)

        return sorted(list(premises))

    @classmethod
    def _is_valid_identifier(cls, token: str) -> bool:
        if not token:
            return False
        # Filter common keywords
        keywords = {
            "fun", "let", "in", "with", "case", "by", "if", "then", "else",
            "at", "only", "using", "intro", "revert", "clear"
        }
        if token in keywords:
            return False
        return bool(cls._IDENTIFIER_PATTERN.fullmatch(token))

    @classmethod
    def audit_proof_soundness(cls, script_text: str) -> tuple[bool, str]:
        """Verify that a proof script contains no forbidden sound-violating constructs.
        
        Returns (is_sound, reason).
        """
        cleaned = cls._strip_comments(script_text)
        tokens = set(re.findall(r"\b[A-Za-z_#][A-Za-z0-9_.]*\b", cleaned))

        for forbidden in cls._FORBIDDEN_TOKENS:
            if forbidden in tokens:
                return False, f"Proof contains forbidden sound-violating token: '{forbidden}'"

        return True, "Sound proof script"

    @classmethod
    def summarize_proof(
        cls,
        theorem: TheoremSpec,
        tactic_steps: Sequence[str],
    ) -> ProofTermSummary:
        """Analyze and summarize an entire completed proof trace."""
        full_text = "\n".join(tactic_steps)
        decomposed = cls.decompose_script(full_text)
        has_sorry = any("sorry" in step for step in decomposed)
        has_admit = any("admit" in step for step in decomposed)

        all_premises: set[str] = set()
        for step in decomposed:
            for p in cls.extract_premises(step):
                all_premises.add(p)

        is_sound, _ = cls.audit_proof_soundness(full_text)
        depth = len(decomposed)
        
        # Complexity heuristic based on length and premise diversity
        complexity = float(depth * 1.5 + len(all_premises) * 2.0)

        return ProofTermSummary(
            theorem_name=theorem.name,
            tactic_count=depth,
            premises_used=sorted(list(all_premises)),
            has_sorry=has_sorry,
            has_admit=has_admit,
            is_axiom_free=is_sound and not has_sorry and not has_admit,
            syntactic_depth=depth,
            complexity_score=complexity,
            raw_script=full_text,
        )

    @classmethod
    def format_lean_theorem(
        cls,
        theorem: TheoremSpec,
        tactic_steps: Sequence[str],
        indent_spaces: int = 2,
    ) -> str:
        """Format a complete compilable Lean 4 theorem definition with proper indentation."""
        indent = " " * indent_spaces
        lines: list[str] = []

        if theorem.namespace:
            lines.append(f"namespace {theorem.namespace}")

        lines.append(f"theorem {theorem.name} : {theorem.statement} := by")
        for step in tactic_steps:
            stripped = step.strip()
            if stripped:
                lines.append(f"{indent}{stripped}")

        if theorem.namespace:
            lines.append(f"end {theorem.namespace}")

        return "\n".join(lines) + "\n"
