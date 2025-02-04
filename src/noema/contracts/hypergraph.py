"""AND/OR proof-hypergraph representation contracts."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from noema.contracts.actions import TacticAction
from noema.contracts.states import ProofGoal, ProofState


class HyperNode(BaseModel):
    """An OR-node in the proof hypergraph representing a proof state."""
    node_id: str
    state_id: str
    goal: ProofGoal
    depth: int = 0
    is_proved: bool = False
    is_failed: bool = False
    neural_value: float = 0.0
    prior_policy_prob: float = 1.0
    cost_estimate: float = 1.0
    visits: int = 0
    incoming_edge_ids: list[str] = Field(default_factory=list)
    outgoing_edge_ids: list[str] = Field(default_factory=list)


class HyperEdge(BaseModel):
    """An AND-edge representing a tactic application that transitions to multiple subgoals."""
    edge_id: str
    source_node_id: str
    target_node_ids: list[str] = Field(default_factory=list, description="Child OR-nodes (AND conjunction)")
    action: TacticAction
    action_logprob: float = 0.0
    is_proved: bool = False
    is_invalid: bool = False
    execution_time_ms: float = 0.0
    mvar_assignments: dict[str, str] = Field(default_factory=dict)


class ProofHypergraph(BaseModel):
    """Full AND/OR proof hypergraph tracking exploration history and proof solutions."""
    root_node_id: str
    nodes: dict[str, HyperNode] = Field(default_factory=dict)
    edges: dict[str, HyperEdge] = Field(default_factory=dict)
    shared_mvars: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map of mvar_id to list of dependent node IDs"
    )
    is_proved: bool = False
    proof_edge_ids: list[str] = Field(default_factory=list)

    def add_node(self, node: HyperNode) -> None:
        self.nodes[node.node_id] = node
        mvar = node.goal.mvar_id
        if mvar not in self.shared_mvars:
            self.shared_mvars[mvar] = []
        if node.node_id not in self.shared_mvars[mvar]:
            self.shared_mvars[mvar].append(node.node_id)

    def add_edge(self, edge: HyperEdge) -> None:
        self.edges[edge.edge_id] = edge
        if edge.source_node_id in self.nodes:
            self.nodes[edge.source_node_id].outgoing_edge_ids.append(edge.edge_id)
        for target_id in edge.target_node_ids:
            if target_id in self.nodes:
                self.nodes[target_id].incoming_edge_ids.append(edge.edge_id)

    def update_proved_status(self, node_id: str) -> bool:
        """Propagate proof success upwards through AND-edges and OR-nodes."""
        node = self.nodes.get(node_id)
        if not node:
            return False
        if node.is_proved:
            return True

        # An OR node is proved if at least one outgoing AND edge is proved
        for edge_id in node.outgoing_edge_ids:
            edge = self.edges.get(edge_id)
            if not edge or edge.is_invalid:
                continue
            
            # An AND edge is proved if ALL its child nodes are proved
            if not edge.target_node_ids:
                # 0 child goals means the tactic closed this goal completely!
                edge.is_proved = True
                node.is_proved = True
                return True
            
            all_children_proved = all(
                self.nodes[cid].is_proved for cid in edge.target_node_ids if cid in self.nodes
            )
            if all_children_proved:
                edge.is_proved = True
                node.is_proved = True
                return True

        return False
