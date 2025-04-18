"""Unit tests for first-order term rewriting and Knuth-Bendix completion."""

import pytest
from noema.search.rewriting import (
    EquationalProver,
    FirstOrderTerm,
    KnuthBendixCompletion,
    RewriteRule,
    Term,
    TermRewriter,
    find_critical_pairs,
    lexicographic_path_order,
    match_term,
    unify_terms,
)


def test_term_parsing_and_str():
    t1 = Term.parse("f(x, g(y))")
    assert t1.symbol == "f"
    assert len(t1.args) == 2
    assert t1.args[0].symbol == "x"
    assert t1.args[1].symbol == "g"
    assert str(t1) == "f(x, g(y))"

    t2 = Term.parse("?x")
    assert t2.is_var
    assert t2.symbol == "x"
    assert str(t2) == "?x"


def test_term_variables_and_subterms():
    t = Term.parse("mul(add(?x, zero), inv(?x))")
    vars_found = t.variables()
    assert vars_found == {"x"}
    subterms = t.subterms()
    assert len(subterms) > 3


def test_pattern_matching():
    pattern = Term.parse("add(zero, ?x)")
    target = Term.parse("add(zero, succ(zero))")
    subst = match_term(pattern, target)
    assert subst is not None
    assert "x" in subst
    assert subst["x"] == Term.parse("succ(zero)")

    # Non-matching
    target_fail = Term.parse("mul(zero, succ(zero))")
    assert match_term(pattern, target_fail) is None


def test_first_order_unification():
    t1 = Term.parse("f(?x, g(?y))")
    t2 = Term.parse("f(a, g(b))")
    subst = unify_terms(t1, t2)
    assert subst is not None
    assert subst["x"] == Term.const("a")
    assert subst["y"] == Term.const("b")

    # Occurs check failure
    t_occurs1 = Term.parse("?x")
    t_occurs2 = Term.parse("f(?x)")
    assert unify_terms(t_occurs1, t_occurs2) is None


def test_lexicographic_path_order():
    precedence = {"mul": 3, "add": 2, "zero": 1}
    # mul(x, y) > add(x, y) because precedence of mul > add
    t1 = Term.parse("mul(?x, ?y)")
    t2 = Term.parse("add(?x, ?y)")
    assert lexicographic_path_order(t1, t2, precedence)
    assert not lexicographic_path_order(t2, t1, precedence)


def test_term_rewriter_normalization():
    r1 = RewriteRule("add_zero", Term.parse("add(zero, ?x)"), Term.parse("?x"))
    r2 = RewriteRule(
        "add_succ",
        Term.parse("add(succ(?x), ?y)"),
        Term.parse("succ(add(?x, ?y))"),
    )
    rewriter = TermRewriter([r1, r2])

    # add(zero, succ(zero)) -> succ(zero)
    t = Term.parse("add(zero, succ(zero))")
    norm, trace = rewriter.normalize(t)
    assert norm == Term.parse("succ(zero)")
    assert len(trace) == 1
    assert trace[0].name == "add_zero"

    # add(succ(zero), zero) -> succ(add(zero, zero)) -> succ(zero)
    t2 = Term.parse("add(succ(zero), zero)")
    norm2, trace2 = rewriter.normalize(t2)
    assert norm2 == Term.parse("succ(zero)")
    assert len(trace2) == 2


def test_critical_pairs_and_completion():
    precedence = {"inv": 4, "mul": 3, "e": 2}
    kb = KnuthBendixCompletion(precedence)
    # Equations for identity: mul(e, x) = x, mul(inv(x), x) = e
    eqs = [
        (Term.parse("mul(e, ?x)"), Term.parse("?x")),
        (Term.parse("mul(inv(?x), ?x)"), Term.parse("e")),
    ]
    rules = kb.complete(eqs, max_iterations=20)
    assert len(rules) >= 2


def test_equational_prover():
    r1 = RewriteRule("id_left", Term.parse("mul(e, ?x)"), Term.parse("?x"))
    r2 = RewriteRule("inv_left", Term.parse("mul(inv(?x), ?x)"), Term.parse("e"))
    rewriter = TermRewriter([r1, r2])
    prover = EquationalProver(rewriter)

    left = Term.parse("mul(inv(a), a)")
    right = Term.parse("e")
    is_eq, norm_l, norm_r, steps = prover.prove_equality(left, right)
    assert is_eq
    assert norm_l == Term.const("e")
    assert norm_r == Term.const("e")
    assert len(steps) >= 1
