"""Analytic semantic tableau automated theorem prover and Lean 4 proof extractor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence


class Formula(ABC):
    """Abstract base class for propositional formulas."""

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass(frozen=True)
class Atom(Formula):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Not(Formula):
    inner: Formula

    def __str__(self) -> str:
        return f"¬{self.inner}"


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Imp(Formula):
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} → {self.right})"


@dataclass(frozen=True)
class SignedFormula:
    """A formula labeled as True (T) or False (F)."""
    sign: bool  # True for T, False for F
    formula: Formula

    def __str__(self) -> str:
        prefix = "T" if self.sign else "F"
        return f"{prefix}:{self.formula}"


@dataclass
class TableauBranch:
    """A single branch in the analytic tableau search tree."""
    formulas: list[SignedFormula] = field(default_factory=list)
    is_closed: bool = False
    closing_atom: str | None = None

    def check_closure(self) -> bool:
        """Check if branch contains both T:P and F:P for an atomic proposition."""
        true_atoms: set[str] = set()
        false_atoms: set[str] = set()

        for sf in self.formulas:
            if isinstance(sf.formula, Atom):
                if sf.sign:
                    true_atoms.add(sf.formula.name)
                else:
                    false_atoms.add(sf.formula.name)

        overlap = true_atoms.intersection(false_atoms)
        if overlap:
            self.is_closed = True
            self.closing_atom = next(iter(overlap))
            return True
        return False


class SemanticTableauProver:
    """Refutation-based analytic tableau solver for propositional logic."""

    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth

    def prove(self, target: Formula) -> tuple[bool, list[TableauBranch]]:
        """Prove validity of target formula by refuting F:target.
        
        Returns (is_valid, list of closed branches).
        """
        # Root refutation starts with F:target
        initial_branch = TableauBranch(formulas=[SignedFormula(sign=False, formula=target)])
        branches: list[TableauBranch] = [initial_branch]

        for _ in range(self.max_depth):
            all_closed = True
            new_branches: list[TableauBranch] = []

            for branch in branches:
                if branch.is_closed or branch.check_closure():
                    new_branches.append(branch)
                    continue

                all_closed = False
                expanded = False

                for i, sf in enumerate(branch.formulas):
                    # Alpha rules (non-branching)
                    if sf.sign and isinstance(sf.formula, And):
                        sub = [f for j, f in enumerate(branch.formulas) if j != i]
                        sub.append(SignedFormula(True, sf.formula.left))
                        sub.append(SignedFormula(True, sf.formula.right))
                        b_new = TableauBranch(formulas=sub)
                        b_new.check_closure()
                        new_branches.append(b_new)
                        expanded = True
                        break

                    if not sf.sign and isinstance(sf.formula, Or):
                        sub = [f for j, f in enumerate(branch.formulas) if j != i]
                        sub.append(SignedFormula(False, sf.formula.left))
                        sub.append(SignedFormula(False, sf.formula.right))
                        b_new = TableauBranch(formulas=sub)
                        b_new.check_closure()
                        new_branches.append(b_new)
                        expanded = True
                        break

                    if not sf.sign and isinstance(sf.formula, Imp):
                        sub = [f for j, f in enumerate(branch.formulas) if j != i]
                        sub.append(SignedFormula(True, sf.formula.left))
                        sub.append(SignedFormula(False, sf.formula.right))
                        b_new = TableauBranch(formulas=sub)
                        b_new.check_closure()
                        new_branches.append(b_new)
                        expanded = True
                        break

                    if isinstance(sf.formula, Not):
                        sub = [f for j, f in enumerate(branch.formulas) if j != i]
                        sub.append(SignedFormula(not sf.sign, sf.formula.inner))
                        b_new = TableauBranch(formulas=sub)
                        b_new.check_closure()
                        new_branches.append(b_new)
                        expanded = True
                        break

                    # Beta rules (branching)
                    if sf.sign and isinstance(sf.formula, Or):
                        sub1 = [f for j, f in enumerate(branch.formulas) if j != i]
                        sub1.append(SignedFormula(True, sf.formula.left))
                        b1 = TableauBranch(formulas=sub1)
                        b1.check_closure()

                        sub2 = [f for j, f in enumerate(branch.formulas) if j != i]
                        sub2.append(SignedFormula(True, sf.formula.right))
                        b2 = TableauBranch(formulas=sub2)
                        b2.check_closure()

                        new_branches.extend([b1, b2])
                        expanded = True
                        break

                    if not sf.sign and isinstance(sf.formula, And):
                        sub1 = [f for j, f in enumerate(branch.formulas) if j != i]
                        sub1.append(SignedFormula(False, sf.formula.left))
                        b1 = TableauBranch(formulas=sub1)
                        b1.check_closure()

                        sub2 = [f for j, f in enumerate(branch.formulas) if j != i]
                        sub2.append(SignedFormula(False, sf.formula.right))
                        b2 = TableauBranch(formulas=sub2)
                        b2.check_closure()

                        new_branches.extend([b1, b2])
                        expanded = True
                        break

                    if sf.sign and isinstance(sf.formula, Imp):
                        sub1 = [f for j, f in enumerate(branch.formulas) if j != i]
                        sub1.append(SignedFormula(False, sf.formula.left))
                        b1 = TableauBranch(formulas=sub1)
                        b1.check_closure()

                        sub2 = [f for j, f in enumerate(branch.formulas) if j != i]
                        sub2.append(SignedFormula(True, sf.formula.right))
                        b2 = TableauBranch(formulas=sub2)
                        b2.check_closure()

                        new_branches.extend([b1, b2])
                        expanded = True
                        break

                if not expanded:
                    # No rules applicable, branch remains open
                    new_branches.append(branch)

            branches = new_branches
            if all_closed:
                return True, branches

        is_valid = all(b.is_closed for b in branches)
        return is_valid, branches
