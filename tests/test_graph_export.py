"""Unit tests for symbolic proof graph exporter."""

import json
import pytest
from noema.artifacts.graph_export import ProofGraphExporter


def test_proof_graph_exporter_dot_generation():
    exporter = ProofGraphExporter()
    exporter.add_node("n0", "⊢ P ∧ Q", is_root=True, heuristic_cost=1.5)
    exporter.add_node("n1", "⊢ P", is_proved=True, depth=1)
    exporter.add_node("n2", "⊢ Q", is_proved=True, depth=1)

    # Multi-target AND hyperedge
    exporter.add_hyperedge("e0", "n0", ["n1", "n2"], tactic="constructor", confidence=0.9)
    # Terminal edges
    exporter.add_hyperedge("e1", "n1", [], tactic="exact hp", confidence=1.0)
    exporter.add_hyperedge("e2", "n2", [], tactic="exact hq", confidence=1.0)

    dot = exporter.to_dot()
    assert "digraph ProofHypergraph" in dot
    assert '"n0"' in dot
    assert '"n1"' in dot
    assert "constructor" in dot
    assert "QED" in dot
    assert "AND" in dot


def test_proof_graph_exporter_cytoscape_json():
    exporter = ProofGraphExporter()
    exporter.add_node("n0", "⊢ P → P", is_root=True)
    exporter.add_node("n1", "⊢ P", is_proved=True)
    exporter.add_hyperedge("e0", "n0", ["n1"], tactic="intro h", confidence=0.95)

    cyto_json = exporter.to_cytoscape_json()
    data = json.loads(cyto_json)

    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    assert data["nodes"][0]["data"]["id"] == "n0"
    assert data["edges"][0]["data"]["tactic"] == "intro h"


def test_proof_graph_exporter_svg_generation():
    exporter = ProofGraphExporter()
    exporter.add_node("root", "Goal: ∀ x, x = x", is_root=True)
    exporter.add_node("solved", "Empty goals", is_proved=True)
    exporter.add_hyperedge("e_rfl", "root", ["solved"], tactic="rfl")

    svg = exporter.to_svg()
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "provedGrad" in svg
    assert "Goal: ∀ x, x = x" in svg
    assert "<line" in svg
