"""Unit tests for knowledge graph premise retriever and Personalized PageRank."""

import pytest
from noema.retrieval.graph_retriever import (
    PersonalizedPageRankRetriever,
    PremiseDependencyGraph,
)


def test_dependency_graph_closure():
    graph = PremiseDependencyGraph()
    # thm_c depends on thm_b, which depends on thm_a
    graph.add_premise("thm_a", "Init.Core", dependencies=[])
    graph.add_premise("thm_b", "Init.Core", dependencies=["thm_a"])
    graph.add_premise("thm_c", "Init.Core", dependencies=["thm_b"])

    closure_c = graph.transitive_dependencies("thm_c")
    assert closure_c == {"thm_b", "thm_a"}
    assert not graph.has_circular_dependency()


def test_dependency_graph_cycle_detection():
    graph = PremiseDependencyGraph()
    graph.add_premise("a", dependencies=["b"])
    graph.add_premise("b", dependencies=["c"])
    graph.add_premise("c", dependencies=["a"])
    assert graph.has_circular_dependency()


def test_personalized_pagerank_computation():
    graph = PremiseDependencyGraph()
    graph.add_premise("root", dependencies=["lemma_1", "lemma_2"])
    graph.add_premise("lemma_1", dependencies=["axiom_base"])
    graph.add_premise("lemma_2", dependencies=["axiom_base"])
    graph.add_premise("axiom_base", dependencies=[])

    retriever = PersonalizedPageRankRetriever(graph, damping=0.85)
    # Teleport seeded on root
    scores = retriever.compute_ppr({"root": 1.0})

    assert len(scores) == 4
    assert sum(scores.values()) > 0.99
    # Downstream dependencies should receive substantial random-walk probability
    assert scores["axiom_base"] > 0.0


def test_hybrid_ranking_filters_circular_dependencies():
    graph = PremiseDependencyGraph()
    graph.add_premise("axiom_add", dependencies=[])
    graph.add_premise("target_conjecture", dependencies=["axiom_add"])
    # downstream_lemma depends on target_conjecture!
    graph.add_premise("downstream_lemma", dependencies=["target_conjecture"])

    retriever = PersonalizedPageRankRetriever(graph)
    dense_scores = {
        "axiom_add": 0.8,
        "downstream_lemma": 0.99,  # High dense similarity, but circular!
        "target_conjecture": 1.0,
    }

    ranked = retriever.rank_premises(
        dense_scores=dense_scores,
        target_conjecture="target_conjecture",
        top_k=5,
        graph_weight=0.5,
    )

    ranked_names = [name for name, _ in ranked]
    assert "axiom_add" in ranked_names
    # downstream_lemma must be filtered to prevent circular proof reasoning
    assert "downstream_lemma" not in ranked_names
    assert "target_conjecture" not in ranked_names
