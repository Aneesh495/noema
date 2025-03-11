"""Tactical action grammar and action representation contracts."""

from __future__ import annotations

import re
from enum import Enum
from pydantic import BaseModel, Field


class TacticKind(str, Enum):
    INTRO = "intro"
    INTROS = "intros"
    APPLY = "apply"
    EXACT = "exact"
    REWRITE = "rewrite"
    REWRITE_REV = "rewrite_rev"
    CASES = "cases"
    INDUCTION = "induction"
    CONSTRUCTOR = "constructor"
    RFL = "rfl"
    ASSUMPTION = "assumption"
    SPLIT = "split"
    HAVE = "have"
    TRIVIAL = "trivial"
    DECIDE = "decide"
    CUSTOM = "custom"


class TacticAction(BaseModel):
    """Structured representation of a tactical proof action."""
    action_id: str
    kind: TacticKind
    raw_tactic: str
    target_goal_id: str | None = None
    target_mvar_id: str | None = None
    premises: list[str] = Field(default_factory=list, description="Referenced premise identifiers")
    parameters: list[str] = Field(default_factory=list, description="Tactical arguments e.g. variable names")
    confidence: float = 1.0

    @classmethod
    def from_string(cls, raw: str, goal_id: str | None = None, mvar_id: str | None = None) -> TacticAction:
        clean = raw.strip()
        tokens = clean.split()
        if not tokens:
            return cls(
                action_id="empty",
                kind=TacticKind.CUSTOM,
                raw_tactic="",
                target_goal_id=goal_id,
                target_mvar_id=mvar_id,
            )

        cmd = tokens[0].lower()
        args = tokens[1:]
        premises = []
        parameters = []

        if cmd in ("rfl", "reflexivity"):
            kind = TacticKind.RFL
        elif cmd in ("intro", "intros"):
            kind = TacticKind.INTROS if cmd == "intros" else TacticKind.INTRO
            parameters = args
        elif cmd == "apply":
            kind = TacticKind.APPLY
            if args:
                premises = [args[0]]
                parameters = args[1:]
        elif cmd == "exact":
            kind = TacticKind.EXACT
            if args:
                premises = [args[0]]
                parameters = args[1:]
        elif cmd in ("rw", "rewrite"):
            if args and args[0] in ("<-", "←"):
                kind = TacticKind.REWRITE_REV
                if len(args) > 1:
                    raw_p = args[1].strip("[]")
                    premises = [raw_p] if raw_p else []
                    parameters = args[2:]
            else:
                kind = TacticKind.REWRITE
                if args:
                    raw_p = args[0].strip("[]")
                    premises = [raw_p] if raw_p else []
                    parameters = args[1:]
        elif cmd == "have":
            kind = TacticKind.HAVE
            if ":=" in clean:
                rhs = clean.split(":=", 1)[1].strip()
                if rhs:
                    raw_p = rhs.split()[0].strip("[]")
                    if raw_p:
                        premises = [raw_p]
            parameters = args
        elif cmd == "cases":
            kind = TacticKind.CASES
            parameters = args
        elif cmd == "induction":
            kind = TacticKind.INDUCTION
            parameters = args
        elif cmd == "constructor":
            kind = TacticKind.CONSTRUCTOR
        elif cmd == "assumption":
            kind = TacticKind.ASSUMPTION
        elif cmd == "split":
            kind = TacticKind.SPLIT
        elif cmd == "trivial":
            kind = TacticKind.TRIVIAL
        elif cmd == "decide":
            kind = TacticKind.DECIDE
        else:
            kind = TacticKind.CUSTOM
            parameters = args

        action_id = f"{kind.value}_{'-'.join(tokens[:3])}"
        return cls(
            action_id=action_id,
            kind=kind,
            raw_tactic=clean,
            target_goal_id=goal_id,
            target_mvar_id=mvar_id,
            premises=premises,
            parameters=parameters,
        )


class ActionGrammar:
    """Grammar validator ensuring actions match the supported formal dialect."""

    SUPPORTED_COMMANDS = {
        "rfl", "reflexivity", "intro", "intros", "apply", "exact", "rw", "rewrite",
        "cases", "induction", "constructor", "assumption", "split", "trivial",
        "decide", "have"
    }

    @classmethod
    def is_valid_tactic_syntax(cls, tactic_str: str) -> tuple[bool, str]:
        s = tactic_str.strip()
        if not s:
            return False, "Empty tactic string"
        if ";" in s or ";;" in s:
            return False, "Semicolon combinators disabled in primary search"
        
        # Check for harmful or hole-introducing keywords
        bad_patterns = [r"\bsorry\b", r"\badmit\b", r"\bcheat\b", r"\bunsafe\b"]
        for pat in bad_patterns:
            if re.search(pat, s):
                return False, f"Prohibited tactic pattern: {pat}"

        # Check command
        head = s.split()[0]
        if head.startswith("<"):
            return False, "Invalid syntax header"
        if head not in cls.SUPPORTED_COMMANDS:
            return False, f"Unsupported command '{head}' outside bounded grammar"

        return True, "Valid"
