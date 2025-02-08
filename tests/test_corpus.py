"""Unit tests for corpus generation, splits, and contamination audits."""

import pytest
from noema.contracts.environment import EnvironmentSpec, PremiseDecl
from noema.contracts.theorems import TheoremSpec
from noema.corpus import (
    TaskGenerator,
    get_public_held_out_theorems,
    get_minif2f_theorems,
    ContaminationAuditor,
    DependencyGraph,
)


def test_task_generator_families():
    gen = TaskGenerator(seed=123)
    # Check all families are represented
    seen_families = set()
    for i in range(len(TaskGenerator.FAMILIES)):
        t = gen.generate_task(i)
        seen_families.add(t.family)
        assert t.statement
        assert t.reference_proof is not None
        assert t.template_id is not None
    assert len(seen_families) == len(TaskGenerator.FAMILIES)


def test_public_mathlib_and_minif2f_counts():
    public_theorems = get_public_held_out_theorems(300)
    assert len(public_theorems) == 300
    
    minif2f_theorems = get_minif2f_theorems()
    assert len(minif2f_theorems) == 244
    assert all(t.family == "miniF2F_test" for t in minif2f_theorems)


def test_contamination_audit_scope_and_splits():
    target = TheoremSpec(
        theorem_id="eval_001",
        name="target_lemma",
        module="Noema.Eval.Core",
        statement="x = x",
    )
    
    p_clean = PremiseDecl(name="Nat.add_comm", type_str="...", module="Init.Data.Nat")
    p_bad = PremiseDecl(name="target_lemma", type_str="...", module="Noema.Eval.Core")
    p_desc = PremiseDecl(name="target_lemma.sublemma", type_str="...", module="Noema.Eval.Core")
    
    # 1. Scope audit with contaminated premise
    env_bad = EnvironmentSpec(
        toolchain="lean4",
        eligible_premises={"Nat.add_comm": p_clean, "target_lemma": p_bad},
    )
    clean, violations = ContaminationAuditor.audit_environment_scope(target, env_bad)
    assert not clean
    assert any("target_lemma" in v for v in violations)

    # 2. Scope audit with descendant declaration
    env_desc = EnvironmentSpec(
        toolchain="lean4",
        eligible_premises={"Nat.add_comm": p_clean, "target_lemma.sublemma": p_desc},
    )
    clean, violations = ContaminationAuditor.audit_environment_scope(target, env_desc)
    assert not clean
    assert any("sublemma" in v for v in violations)

    # 3. Clean environment
    env_clean = EnvironmentSpec(
        toolchain="lean4",
        eligible_premises={"Nat.add_comm": p_clean},
    )
    clean, violations = ContaminationAuditor.audit_environment_scope(target, env_clean)
    assert clean
    assert len(violations) == 0


def test_dependency_graph_closure():
    dg = DependencyGraph()
    # A -> B -> C (A depends on B, B depends on C)
    dg.add_declaration("A", ["B"])
    dg.add_declaration("B", ["C"])
    dg.add_declaration("C", [])
    
    # Transitive dependencies of A: B, C
    assert dg.get_transitive_dependencies("A") == {"B", "C"}
    # Downstream dependents of C: A, B
    assert dg.get_downstream_dependents("C") == {"A", "B"}
    
    # If target is C, candidate premises cannot include A or B
    clean, leaked = dg.verify_no_downstream_leakage("C", ["A", "D"])
    assert not clean
    assert "A" in leaked

    clean, leaked = dg.verify_no_downstream_leakage("C", ["D", "E"])
    assert clean
    assert len(leaked) == 0
