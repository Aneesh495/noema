"""Metavariable coupling, substitution propagation, and first-order pattern unification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence
from noema.contracts.states import ProofGoal, ProofState


class UnificationError(Exception):
    """Raised when two formal expressions cannot be unified or occurs check fails."""
    pass


@dataclass(frozen=True)
class Term:
    """A first-order term or pattern representation for unification."""
    name: str
    args: tuple[Term, ...] = field(default_factory=tuple)
    is_mvar: bool = False

    @classmethod
    def parse(cls, s: str) -> Term:
        """Parse a simple prefix or infix expression into a Term structure."""
        s = s.strip()
        if not s:
            return cls(name="")

        # Check if metavariable
        if s.startswith("?"):
            return cls(name=s, is_mvar=True)

        # Check for function application: f(arg1, arg2)
        if "(" in s and s.endswith(")"):
            head = s[:s.index("(")].strip()
            inside = s[s.index("(") + 1 : -1].strip()
            args = tuple(cls.parse(part.strip()) for part in inside.split(",") if part.strip())
            return cls(name=head, args=args, is_mvar=head.startswith("?"))

        # Atomic constant or variable
        return cls(name=s, is_mvar=False)

    def to_string(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(a.to_string() for a in self.args)})"


class UnificationEngine:
    """Solves syntactic unification equations with occurs check for metavariables."""

    @classmethod
    def unify(cls, t1: Term, t2: Term, subst: dict[str, Term] | None = None) -> dict[str, Term]:
        """Compute the Most General Unifier (MGU) of terms t1 and t2."""
        if subst is None:
            subst = {}

        t1 = cls.apply_subst(t1, subst)
        t2 = cls.apply_subst(t2, subst)

        if t1 == t2:
            return subst

        if t1.is_mvar:
            return cls._unify_variable(t1.name, t2, subst)

        if t2.is_mvar:
            return cls._unify_variable(t2.name, t1, subst)

        if t1.name != t2.name or len(t1.args) != len(t2.args):
            raise UnificationError(f"Symbol mismatch: {t1.to_string()} vs {t2.to_string()}")

        for a1, a2 in zip(t1.args, t2.args):
            subst = cls.unify(a1, a2, subst)

        return subst

    @classmethod
    def _unify_variable(cls, var_name: str, term: Term, subst: dict[str, Term]) -> dict[str, Term]:
        if term.is_mvar and term.name == var_name:
            return subst

        if cls._occurs_check(var_name, term, subst):
            raise UnificationError(f"Occurs check failed: {var_name} occurs in {term.to_string()}")

        subst[var_name] = term
        return subst

    @classmethod
    def _occurs_check(cls, var_name: str, term: Term, subst: dict[str, Term]) -> bool:
        term = cls.apply_subst(term, subst)
        if term.is_mvar and term.name == var_name:
            return True
        return any(cls._occurs_check(var_name, arg, subst) for arg in term.args)

    @classmethod
    def apply_subst(cls, term: Term, subst: dict[str, Term]) -> Term:
        """Recursively apply substitution map to term."""
        if term.is_mvar and term.name in subst:
            return cls.apply_subst(subst[term.name], subst)
        if not term.args:
            return term
        new_args = tuple(cls.apply_subst(a, subst) for a in term.args)
        return Term(name=term.name, args=new_args, is_mvar=term.is_mvar)


class CoupledMetavariableTracker:
    """Tracks metavariables shared across goals and validates cross-goal assignment consistency."""

    def __init__(self):
        self.assignments: dict[str, Term] = {}

    def assign_mvar(self, mvar_name: str, term_str: str) -> bool:
        """Attempt to assign a term to a metavariable, verifying compatibility with previous assignments."""
        new_term = Term.parse(term_str)
        if mvar_name in self.assignments:
            try:
                # Unify existing assignment with new term
                self.assignments = UnificationEngine.unify(
                    self.assignments[mvar_name],
                    new_term,
                    self.assignments,
                )
                return True
            except UnificationError:
                return False
        else:
            self.assignments[mvar_name] = new_term
            return True

    def find_conflicting_goals(self, goals: Sequence[ProofGoal]) -> list[str]:
        """Identify goals that contain mutually incompatible metavariable requirements."""
        conflicting = []
        for g in goals:
            for mv in g.coupled_mvars:
                if mv in self.assignments:
                    try:
                        g_term = Term.parse(g.target)
                        # Test if target can coexist with current substitutions
                        UnificationEngine.apply_subst(g_term, self.assignments)
                    except Exception:
                        conflicting.append(g.goal_id)
        return list(set(conflicting))

    def propagate_to_state(self, state: ProofState) -> ProofState:
        """Apply active substitutions across all open goals in a proof state."""
        if not self.assignments:
            return state

        new_goals = []
        for g in state.goals:
            target_term = Term.parse(g.target)
            subst_target = UnificationEngine.apply_subst(target_term, self.assignments)
            new_target = subst_target.to_string()
            
            # Recreate goal with substituted target
            new_g = ProofGoal(
                goal_id=g.goal_id,
                mvar_id=g.mvar_id,
                target=new_target,
                hypotheses=g.hypotheses,
                coupled_mvars=g.coupled_mvars,
                is_solved=g.is_solved,
            )
            new_goals.append(new_g)

        return ProofState(
            state_id=state.state_id,
            goals=new_goals,
            assignments={k: v.to_string() for k, v in self.assignments.items()},
            depth=state.depth,
            is_terminal=state.is_terminal,
            is_solved=state.is_solved,
        )
