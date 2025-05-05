"""Symbolic proof graph compiler, GraphViz DOT generator, and Cytoscape JSON exporter."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence
from noema.contracts.hypergraph import ProofHypergraph
from noema.contracts.states import ProofGoal, ProofState


@dataclass
class ExportedGraphNode:
    """Node in an exported proof graph representation."""
    node_id: str
    label: str
    is_proved: bool
    is_root: bool
    depth: int
    heuristic_cost: float = 0.0
    subgoals: list[str] = field(default_factory=list)


@dataclass
class ExportedGraphEdge:
    """Hyperedge or directed edge in an exported proof graph."""
    edge_id: str
    source_id: str
    target_ids: list[str]
    tactic: str
    confidence: float = 1.0


class ProofGraphExporter:
    """Compiles proof hypergraphs into GraphViz DOT, Cytoscape JSON, and standalone SVG formats."""

    def __init__(self):
        self.nodes: dict[str, ExportedGraphNode] = {}
        self.edges: list[ExportedGraphEdge] = []

    def add_node(
        self,
        node_id: str,
        label: str,
        is_proved: bool = False,
        is_root: bool = False,
        depth: int = 0,
        heuristic_cost: float = 0.0,
        subgoals: Sequence[str] | None = None,
    ) -> None:
        self.nodes[node_id] = ExportedGraphNode(
            node_id=node_id,
            label=label,
            is_proved=is_proved,
            is_root=is_root,
            depth=depth,
            heuristic_cost=heuristic_cost,
            subgoals=list(subgoals or []),
        )

    def add_hyperedge(
        self,
        edge_id: str,
        source_id: str,
        target_ids: Sequence[str],
        tactic: str,
        confidence: float = 1.0,
    ) -> None:
        self.edges.append(
            ExportedGraphEdge(
                edge_id=edge_id,
                source_id=source_id,
                target_ids=list(target_ids),
                tactic=tactic,
                confidence=confidence,
            )
        )

    def to_dot(self) -> str:
        """Render the proof hypergraph in GraphViz DOT notation."""
        lines = [
            "digraph ProofHypergraph {",
            "  rankdir=TB;",
            "  node [fontname=\"Courier New\", fontsize=10, shape=box, style=\"rounded,filled\"];",
            "  edge [fontname=\"Courier New\", fontsize=9];",
            "",
        ]

        # Nodes
        for nid, node in sorted(self.nodes.items()):
            color = "#a8e6cf" if node.is_proved else ("#ffaaa5" if node.heuristic_cost > 10.0 else "#dcedc1")
            shape = "doublecircle" if node.is_root else "box"
            escaped_lbl = html.escape(node.label.replace('"', '\\"'))
            lines.append(
                f'  "{nid}" [label="{escaped_lbl}\\n(cost={node.heuristic_cost:.2f})", fillcolor="{color}", shape="{shape}"];'
            )

        lines.append("")

        # Hyperedges
        for e in self.edges:
            escaped_tac = html.escape(e.tactic.replace('"', '\\"'))
            if not e.target_ids:
                # Terminal edge (QED)
                qed_id = f"qed_{e.edge_id}"
                lines.append(f'  "{qed_id}" [label="QED", shape=ellipse, fillcolor="#56ab2f", fontcolor=white, style=filled];')
                lines.append(f'  "{e.source_id}" -> "{qed_id}" [label="{escaped_tac}", color="#2b580c", penwidth=2.0];')
            elif len(e.target_ids) == 1:
                # Simple single-target edge
                dst = e.target_ids[0]
                lines.append(f'  "{e.source_id}" -> "{dst}" [label="{escaped_tac} (p={e.confidence:.2f})"];')
            else:
                # Multi-target AND-hyperedge represented via intermediary connector
                connector_id = f"and_{e.edge_id}"
                lines.append(f'  "{connector_id}" [label="AND", shape=diamond, fillcolor="#ffd3b6", style=filled, width=0.4, height=0.4];')
                lines.append(f'  "{e.source_id}" -> "{connector_id}" [label="{escaped_tac}", style=dashed];')
                for tgt in e.target_ids:
                    lines.append(f'  "{connector_id}" -> "{tgt}";')

        lines.append("}")
        return "\n".join(lines)

    def to_cytoscape_json(self) -> str:
        """Render in Cytoscape.js JSON format for web view rendering."""
        elements: dict[str, list[dict]] = {"nodes": [], "edges": []}

        for nid, node in sorted(self.nodes.items()):
            elements["nodes"].append({
                "data": {
                    "id": nid,
                    "label": node.label,
                    "is_proved": node.is_proved,
                    "is_root": node.is_root,
                    "depth": node.depth,
                    "cost": node.heuristic_cost,
                }
            })

        for e in self.edges:
            for tgt in e.target_ids:
                elements["edges"].append({
                    "data": {
                        "id": f"{e.edge_id}_{tgt}",
                        "source": e.source_id,
                        "target": tgt,
                        "tactic": e.tactic,
                        "confidence": e.confidence,
                    }
                })

        return json.dumps(elements, indent=2)

    def to_svg(self) -> str:
        """Render a clean standalone SVG representation of the proof tree."""
        # Simple vertical hierarchy layout
        svg_width = 800
        svg_height = max(400, 100 + len(self.nodes) * 80)
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="100%" height="100%">',
            '  <defs>',
            '    <linearGradient id="provedGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
            '      <stop offset="0%" stop-color="#2ecc71" />',
            '      <stop offset="100%" stop-color="#27ae60" />',
            '    </linearGradient>',
            '    <linearGradient id="openGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
            '      <stop offset="0%" stop-color="#3498db" />',
            '      <stop offset="100%" stop-color="#2980b9" />',
            '    </linearGradient>',
            '  </defs>',
            '  <rect width="100%" height="100%" fill="#1a1a24" />',
        ]

        y_offset = 60
        node_positions: dict[str, tuple[int, int]] = {}

        for nid, node in sorted(self.nodes.items()):
            x = 400
            y = y_offset
            node_positions[nid] = (x, y)
            grad = "url(#provedGrad)" if node.is_proved else "url(#openGrad)"
            escaped_text = html.escape(node.label[:45])
            lines.append(
                f'  <g id="node_{nid}">'
            )
            lines.append(
                f'    <rect x="{x - 180}" y="{y - 25}" width="360" height="50" rx="8" fill="{grad}" stroke="#ffffff" stroke-width="1.5" />'
            )
            lines.append(
                f'    <text x="{x}" y="{y + 5}" fill="#ffffff" font-family="Courier, monospace" font-size="12" text-anchor="middle">{escaped_text}</text>'
            )
            lines.append('  </g>')
            y_offset += 100

        # Draw connecting edges
        for e in self.edges:
            if e.source_id in node_positions:
                sx, sy = node_positions[e.source_id]
                for tgt in e.target_ids:
                    if tgt in node_positions:
                        tx, ty = node_positions[tgt]
                        lines.append(
                            f'  <line x1="{sx}" y1="{sy + 25}" x2="{tx}" y2="{ty - 25}" stroke="#95a5a6" stroke-width="2" stroke-dasharray="4" />'
                        )

        lines.append('</svg>')
        return "\n".join(lines)
