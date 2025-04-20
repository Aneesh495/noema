"""Unit tests for explainable congruence closure on ground equational graphs."""

import pytest
from noema.search.congruence import CongruenceClosure, JustificationKind


def test_congruence_initialization():
    cc = CongruenceClosure()
    id_a = cc.add_term("a")
    id_b = cc.add_term("b")
    assert id_a != id_b
    assert not cc.are_equal("a", "b")


def test_transitivity_of_equality():
    cc = CongruenceClosure()
    cc.assert_equality("a", "b", axiom_name="eq_ab")
    cc.assert_equality("b", "c", axiom_name="eq_bc")
    assert cc.are_equal("a", "c")
    assert cc.are_equal("c", "a")
    assert not cc.are_equal("a", "d")


def test_function_congruence():
    cc = CongruenceClosure()
    # a = b implies f(a) = f(b)
    cc.add_term("f(a)")
    cc.add_term("f(b)")
    assert not cc.are_equal("f(a)", "f(b)")

    cc.assert_equality("a", "b", axiom_name="a_equals_b")
    assert cc.are_equal("f(a)", "f(b)")


def test_nested_multi_argument_congruence():
    cc = CongruenceClosure()
    # x = y, u = v implies g(f(x), u) = g(f(y), v)
    cc.add_term("g(f(x), u)")
    cc.add_term("g(f(y), v)")
    assert not cc.are_equal("g(f(x), u)", "g(f(y), v)")

    cc.assert_equality("x", "y", axiom_name="ax_xy")
    assert cc.are_equal("f(x)", "f(y)")
    assert not cc.are_equal("g(f(x), u)", "g(f(y), v)")

    cc.assert_equality("u", "v", axiom_name="ax_uv")
    assert cc.are_equal("g(f(x), u)", "g(f(y), v)")


def test_explain_equality():
    cc = CongruenceClosure()
    cc.assert_equality("a", "b", axiom_name="hyp1")
    cc.assert_equality("b", "c", axiom_name="hyp2")
    steps = cc.explain("a", "c")
    assert len(steps) >= 1
    assert any("hyp" in s.rationale for s in steps)

    # Explanation with function application
    cc.add_term("f(a)")
    cc.add_term("f(c)")
    f_steps = cc.explain("f(a)", "f(c)")
    assert len(f_steps) >= 1
    assert any("congruence" in s.rationale for s in f_steps)


def test_explain_disjoint_returns_empty():
    cc = CongruenceClosure()
    cc.add_term("alpha")
    cc.add_term("beta")
    steps = cc.explain("alpha", "beta")
    assert steps == []
