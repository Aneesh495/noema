"""First-order term rewriting system (TRS), Knuth-Bendix completion, and equational prover."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Term:
    """First-order term representation."""
    symbol: str
    args: tuple[Term, ...] = ()
    is_var: bool = False

    @classmethod
    def var(cls, name: str) -> Term:
        return cls(symbol=name, args=(), is_var=True)

    @classmethod
    def app(cls, symbol: str, *args: Term) -> Term:
        return cls(symbol=symbol, args=args, is_var=False)

    @classmethod
    def const(cls, symbol: str) -> Term:
        return cls(symbol=symbol, args=(), is_var=False)

    def variables(self) -> set[str]:
        """Collect all variable names present in the term."""
        if self.is_var:
            return {self.symbol}
        vars_found: set[str] = set()
        for a in self.args:
            vars_found.update(a.variables())
        return vars_found

    def subterms(self) -> list[Term]:
        """Return all subterms in post-order traversal."""
        res: list[Term] = []
        for a in self.args:
            res.extend(a.subterms())
        res.append(self)
        return res

    def non_variable_subterms(self) -> list[Term]:
        """Return all non-variable subterms."""
        return [t for t in self.subterms() if not t.is_var]

    def substitute(self, subst: Mapping[str, Term]) -> Term:
        """Apply a variable substitution map to this term."""
        if self.is_var:
            return subst.get(self.symbol, self)
        new_args = tuple(a.substitute(subst) for a in self.args)
        return Term(symbol=self.symbol, args=new_args, is_var=False)

    def replace_subterm(self, target: Term, replacement: Term) -> Term:
        """Replace the first instance of target with replacement."""
        if self == target:
            return replacement
        if self.is_var or not self.args:
            return self
        new_args = list(self.args)
        for i, a in enumerate(self.args):
            sub = a.replace_subterm(target, replacement)
            if sub != a:
                new_args[i] = sub
                return Term(symbol=self.symbol, args=tuple(new_args), is_var=False)
        return self

    def size(self) -> int:
        """Count the number of symbols and variables in the term."""
        if self.is_var or not self.args:
            return 1
        return 1 + sum(a.size() for a in self.args)

    def __str__(self) -> str:
        if self.is_var:
            return f"?{self.symbol}"
        if not self.args:
            return self.symbol
        inner = ", ".join(str(a) for a in self.args)
        return f"{self.symbol}({inner})"

    @classmethod
    def parse(cls, expr: str) -> Term:
        """Parse an expression string like 'f(x, g(y))' or '?x' into a Term."""
        s = expr.strip()
        if not s:
            raise ValueError("Cannot parse empty expression into Term")
        if s.startswith("?"):
            return cls.var(s[1:])
        if "(" not in s:
            return cls.const(s)
        
        m = re.match(r"^([a-zA-Z0-9_+*-]+)\((.*)\)$", s)
        if not m:
            raise ValueError(f"Malformed term syntax: {expr}")
        sym = m.group(1)
        args_str = m.group(2)
        
        # Split args by top-level commas
        args: list[Term] = []
        depth = 0
        current: list[str] = []
        for char in args_str:
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                args.append(cls.parse("".join(current).strip()))
                current = []
            else:
                current.append(char)
        if current:
            args.append(cls.parse("".join(current).strip()))
        return cls.app(sym, *args)


FirstOrderTerm = Term


def match_term(pattern: Term, target: Term, subst: dict[str, Term] | None = None) -> dict[str, Term] | None:
    """Syntactic one-way pattern matching of pattern against target term."""
    if subst is None:
        subst = {}
    if pattern.is_var:
        if pattern.symbol in subst:
            return subst if subst[pattern.symbol] == target else None
        subst[pattern.symbol] = target
        return subst
    if target.is_var:
        return None
    if pattern.symbol != target.symbol or len(pattern.args) != len(target.args):
        return None
    for p_arg, t_arg in zip(pattern.args, target.args):
        res = match_term(p_arg, t_arg, subst)
        if res is None:
            return None
    return subst


def unify_terms(t1: Term, t2: Term, subst: dict[str, Term] | None = None) -> dict[str, Term] | None:
    """Syntactic first-order Robinson unification of two terms."""
    if subst is None:
        subst = {}
    
    t1 = t1.substitute(subst)
    t2 = t2.substitute(subst)
    
    if t1 == t2:
        return subst
    
    if t1.is_var:
        if t1.symbol in t2.variables():
            return None  # Occurs check
        subst[t1.symbol] = t2
        return subst
    
    if t2.is_var:
        if t2.symbol in t1.variables():
            return None  # Occurs check
        subst[t2.symbol] = t1
        return subst
    
    if t1.symbol != t2.symbol or len(t1.args) != len(t2.args):
        return None
    
    for a1, a2 in zip(t1.args, t2.args):
        res = unify_terms(a1, a2, subst)
        if res is None:
            return None
    return subst


@dataclass(frozen=True)
class RewriteRule:
    """Directed rewrite rule lhs -> rhs."""
    name: str
    lhs: Term
    rhs: Term

    def __post_init__(self):
        lhs_vars = self.lhs.variables()
        rhs_vars = self.rhs.variables()
        if not rhs_vars.issubset(lhs_vars):
            raise ValueError(f"Rule {self.name} has unbound rhs vars: {rhs_vars - lhs_vars}")

    def rename_variables(self, suffix: str) -> RewriteRule:
        """Rename all variables with a suffix to avoid variable capture during unification."""
        subst = {v: Term.var(f"{v}_{suffix}") for v in self.lhs.variables()}
        return RewriteRule(
            name=f"{self.name}_{suffix}",
            lhs=self.lhs.substitute(subst),
            rhs=self.rhs.substitute(subst),
        )

    def __str__(self) -> str:
        return f"{self.name}: {self.lhs} => {self.rhs}"


def lexicographic_path_order(
    s: Term, t: Term, precedence: Mapping[str, int]
) -> bool:
    """Decide whether s >_lpo t given symbol precedence map."""
    if s == t:
        return False
    if s.is_var:
        return False
    if t.is_var:
        return t.symbol in s.variables()
    
    # 1. Any argument of s is >= t
    for s_arg in s.args:
        if s_arg == t or lexicographic_path_order(s_arg, t, precedence):
            return True
            
    # 2. s has higher symbol precedence than t, and s > all t's arguments
    prec_s = precedence.get(s.symbol, 0)
    prec_t = precedence.get(t.symbol, 0)
    if prec_s > prec_t:
        return all(lexicographic_path_order(s, t_arg, precedence) for t_arg in t.args)
        
    # 3. Same symbol: lexicographic comparison on arguments
    if s.symbol == t.symbol and len(s.args) == len(t.args):
        if not all(lexicographic_path_order(s, t_arg, precedence) for t_arg in t.args):
            return False
        for s_arg, t_arg in zip(s.args, t.args):
            if s_arg == t_arg:
                continue
            return lexicographic_path_order(s_arg, t_arg, precedence)
            
    return False


class TermRewriter:
    """Executes single-step and normal form reductions using a collection of rewrite rules."""

    def __init__(self, rules: Sequence[RewriteRule] | None = None):
        self.rules: list[RewriteRule] = list(rules or [])

    def add_rule(self, rule: RewriteRule) -> None:
        self.rules.append(rule)

    def step(self, term: Term) -> tuple[Term, RewriteRule | None]:
        """Apply the first matching rewrite rule at the leftmost-innermost position."""
        # Try subterms first (innermost reduction)
        if not term.is_var and term.args:
            for i, a in enumerate(term.args):
                new_a, r = self.step(a)
                if r is not None:
                    new_args = list(term.args)
                    new_args[i] = new_a
                    return Term(symbol=term.symbol, args=tuple(new_args), is_var=False), r

        # Try root reduction
        for rule in self.rules:
            matched = match_term(rule.lhs, term, {})
            if matched is not None:
                contracted = rule.rhs.substitute(matched)
                return contracted, rule

        return term, None

    def normalize(self, term: Term, max_steps: int = 100) -> tuple[Term, list[RewriteRule]]:
        """Compute the normal form of term by repeated reduction."""
        cur = term
        trace: list[RewriteRule] = []
        for _ in range(max_steps):
            next_term, rule = self.step(cur)
            if rule is None or next_term == cur:
                break
            trace.append(rule)
            cur = next_term
        return cur, trace


@dataclass
class CriticalPair:
    """Critical pair (s, t) arising from overlapping rewrite rules."""
    rule1_name: str
    rule2_name: str
    left: Term
    right: Term


def find_critical_pairs(r1: RewriteRule, r2: RewriteRule) -> list[CriticalPair]:
    """Find all critical pairs between r1 and r2 by non-variable subterm overlap."""
    r1_renamed = r1.rename_variables("1")
    r2_renamed = r2.rename_variables("2")
    
    pairs: list[CriticalPair] = []
    
    # Overlap r2 into subterms of r1
    for sub in r1_renamed.lhs.non_variable_subterms():
        subst = unify_terms(sub, r2_renamed.lhs, {})
        if subst is not None:
            # Overlap found
            # 1. Substitute into full r1 LHS
            overlapped_lhs = r1_renamed.lhs.substitute(subst)
            # 2. Reduce via r1
            reduced_via_r1 = r1_renamed.rhs.substitute(subst)
            # 3. Reduce via r2 at the subterm position
            sub_replaced = r2_renamed.rhs.substitute(subst)
            reduced_via_r2 = overlapped_lhs.replace_subterm(
                sub.substitute(subst), sub_replaced
            )
            if reduced_via_r1 != reduced_via_r2:
                pairs.append(
                    CriticalPair(
                        rule1_name=r1.name,
                        rule2_name=r2.name,
                        left=reduced_via_r1,
                        right=reduced_via_r2,
                    )
                )
    return pairs


class KnuthBendixCompletion:
    """Automated Knuth-Bendix completion to construct confluent equational rewrite systems."""

    def __init__(self, precedence: Mapping[str, int]):
        self.precedence = dict(precedence)
        self.rules: list[RewriteRule] = []

    def orient(self, lhs: Term, rhs: Term, name: str) -> RewriteRule | None:
        """Orient an unoriented equation into a rewrite rule using LPO."""
        if lexicographic_path_order(lhs, rhs, self.precedence):
            return RewriteRule(name=name, lhs=lhs, rhs=rhs)
        if lexicographic_path_order(rhs, lhs, self.precedence):
            return RewriteRule(name=name, lhs=rhs, rhs=lhs)
        return None

    def complete(
        self, equations: Sequence[tuple[Term, Term]], max_iterations: int = 50
    ) -> list[RewriteRule]:
        """Execute completion loop until confluence or iteration bound."""
        queue: list[tuple[Term, Term]] = list(equations)
        self.rules.clear()
        rule_counter = 0

        for _ in range(max_iterations):
            if not queue:
                # Compute critical pairs between all pairs of rules
                cp_found = False
                for i, r1 in enumerate(self.rules):
                    for j, r2 in enumerate(self.rules):
                        cps = find_critical_pairs(r1, r2)
                        for cp in cps:
                            rewriter = TermRewriter(self.rules)
                            norm_l, _ = rewriter.normalize(cp.left)
                            norm_r, _ = rewriter.normalize(cp.right)
                            if norm_l != norm_r:
                                queue.append((norm_l, norm_r))
                                cp_found = True
                if not cp_found:
                    # Confluent convergent system found
                    break

            if not queue:
                break

            s, t = queue.pop(0)
            rewriter = TermRewriter(self.rules)
            norm_s, _ = rewriter.normalize(s)
            norm_t, _ = rewriter.normalize(t)

            if norm_s == norm_t:
                continue

            rule_counter += 1
            rule = self.orient(norm_s, norm_t, name=f"R{rule_counter}")
            if rule is None:
                # Cannot orient equation under current precedence
                continue

            self.rules.append(rule)
            
        return list(self.rules)


class EquationalProver:
    """Verifies equational conjectures by normalizing both sides under completed rewrite rules."""

    def __init__(self, rewriter: TermRewriter):
        self.rewriter = rewriter

    def prove_equality(self, left: Term, right: Term) -> tuple[bool, Term, Term, list[str]]:
        """Prove whether left == right under the term rewriting system."""
        norm_l, trace_l = self.rewriter.normalize(left)
        norm_r, trace_r = self.rewriter.normalize(right)
        
        steps = [f"LHS reduced via {r.name}" for r in trace_l]
        steps.extend([f"RHS reduced via {r.name}" for r in trace_r])
        
        is_equal = (norm_l == norm_r)
        return is_equal, norm_l, norm_r, steps
