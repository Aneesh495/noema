"""Lean 4 tactic syntax tree builder, tactical combinators, and proof script optimizers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence


class TacticNode(ABC):
    """Abstract base class for nodes in a structured Lean 4 tactic tree."""

    @abstractmethod
    def to_lean_str(self, indent_level: int = 0) -> str:
        """Render node into formatted Lean 4 tactic source code."""
        pass

    @abstractmethod
    def complexity(self) -> int:
        """Measure size or operational complexity of the tactic construct."""
        pass

    @abstractmethod
    def extract_premises(self) -> list[str]:
        """Extract all premise and lemma names referenced in this tactic node."""
        pass

    def contains_unsafe_keywords(self) -> bool:
        """Inspect whether this tactic node contains unsound cheating keywords."""
        code = self.to_lean_str().lower()
        unsafe = ["sorry", "admit", "cheat", "unsafe"]
        return any(re.search(rf"\b{u}\b", code) for u in unsafe)


@dataclass(frozen=True)
class AtomicTactic(TacticNode):
    """An individual atomic Lean 4 tactic step (e.g. 'intro x', 'rw [Nat.add_comm]')."""
    name: str
    args: tuple[str, ...] = ()

    def to_lean_str(self, indent_level: int = 0) -> str:
        pad = "  " * indent_level
        if not self.args:
            return f"{pad}{self.name}"
        args_str = " ".join(self.args)
        return f"{pad}{self.name} {args_str}"

    def complexity(self) -> int:
        return 1

    def extract_premises(self) -> list[str]:
        premises: list[str] = []
        for arg in self.args:
            # Check for bracketed rewrite rules or exact identifiers
            cleaned = arg.strip("[](),")
            for token in cleaned.split():
                if token and (token[0].isupper() or "." in token):
                    premises.append(token)
        return premises


@dataclass(frozen=True)
class SeqTactic(TacticNode):
    """Sequential composition of tactics."""
    steps: tuple[TacticNode, ...]

    def to_lean_str(self, indent_level: int = 0) -> str:
        return "\n".join(s.to_lean_str(indent_level) for s in self.steps)

    def complexity(self) -> int:
        return sum(s.complexity() for s in self.steps)

    def extract_premises(self) -> list[str]:
        res: list[str] = []
        for s in self.steps:
            res.extend(s.extract_premises())
        return res


@dataclass(frozen=True)
class RepeatTactic(TacticNode):
    """Repetition tactical combinator: repeat t."""
    body: TacticNode

    def to_lean_str(self, indent_level: int = 0) -> str:
        pad = "  " * indent_level
        inner = self.body.to_lean_str(indent_level + 1)
        return f"{pad}repeat\n{inner}"

    def complexity(self) -> int:
        return 1 + self.body.complexity()

    def extract_premises(self) -> list[str]:
        return self.body.extract_premises()


@dataclass(frozen=True)
class TryTactic(TacticNode):
    """Soft-failure tactical combinator: try t."""
    body: TacticNode

    def to_lean_str(self, indent_level: int = 0) -> str:
        pad = "  " * indent_level
        inner = self.body.to_lean_str(indent_level + 1)
        return f"{pad}try\n{inner}"

    def complexity(self) -> int:
        return 1 + self.body.complexity()

    def extract_premises(self) -> list[str]:
        return self.body.extract_premises()


@dataclass(frozen=True)
class FirstTactic(TacticNode):
    """Disjunctive tactic search: first | t1 | t2 | ..."""
    branches: tuple[TacticNode, ...]

    def to_lean_str(self, indent_level: int = 0) -> str:
        pad = "  " * indent_level
        lines = [f"{pad}first"]
        for b in self.branches:
            b_str = b.to_lean_str(indent_level + 1).strip()
            lines.append(f"{pad}  | {b_str}")
        return "\n".join(lines)

    def complexity(self) -> int:
        return 1 + sum(b.complexity() for b in self.branches)

    def extract_premises(self) -> list[str]:
        res: list[str] = []
        for b in self.branches:
            res.extend(b.extract_premises())
        return res


@dataclass(frozen=True)
class AllGoalsTactic(TacticNode):
    """Apply tactic across all open subgoals: all_goals t."""
    body: TacticNode

    def to_lean_str(self, indent_level: int = 0) -> str:
        pad = "  " * indent_level
        inner = self.body.to_lean_str(indent_level + 1)
        return f"{pad}all_goals\n{inner}"

    def complexity(self) -> int:
        return 1 + self.body.complexity()

    def extract_premises(self) -> list[str]:
        return self.body.extract_premises()


@dataclass(frozen=True)
class FocusTactic(TacticNode):
    """Focus on single subgoal using bullet: · t."""
    body: TacticNode

    def to_lean_str(self, indent_level: int = 0) -> str:
        pad = "  " * indent_level
        inner = self.body.to_lean_str(indent_level + 1).lstrip()
        return f"{pad}· {inner}"

    def complexity(self) -> int:
        return self.body.complexity()

    def extract_premises(self) -> list[str]:
        return self.body.extract_premises()


class TacticTreeOptimizer:
    """Optimizes and canonicalizes structured tactic syntax trees."""

    @classmethod
    def optimize(cls, node: TacticNode) -> TacticNode:
        """Apply simplification rules recursively."""
        if isinstance(node, SeqTactic):
            # 1. Flatten nested SeqTactics
            flattened: list[TacticNode] = []
            for s in node.steps:
                opt_s = cls.optimize(s)
                if isinstance(opt_s, SeqTactic):
                    flattened.extend(opt_s.steps)
                else:
                    flattened.append(opt_s)

            # 2. Merge consecutive intros
            merged: list[TacticNode] = []
            idx = 0
            while idx < len(flattened):
                cur = flattened[idx]
                if (
                    isinstance(cur, AtomicTactic)
                    and cur.name == "intro"
                    and idx + 1 < len(flattened)
                    and isinstance(flattened[idx + 1], AtomicTactic)
                    and flattened[idx + 1].name == "intro"
                ):
                    next_intro = flattened[idx + 1]
                    combined_args = cur.args + next_intro.args
                    merged.append(AtomicTactic("intro", combined_args))
                    idx += 2
                else:
                    merged.append(cur)
                    idx += 1

            if len(merged) == 1:
                return merged[0]
            return SeqTactic(tuple(merged))

        if isinstance(node, RepeatTactic):
            opt_inner = cls.optimize(node.body)
            # repeat (repeat t) -> repeat t
            if isinstance(opt_inner, RepeatTactic):
                return opt_inner
            return RepeatTactic(opt_inner)

        if isinstance(node, TryTactic):
            opt_inner = cls.optimize(node.body)
            # try (try t) -> try t
            if isinstance(opt_inner, TryTactic):
                return opt_inner
            return TryTactic(opt_inner)

        return node
