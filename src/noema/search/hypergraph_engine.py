"""AND/OR proof-hypergraph engine with metavariable tracking and proof extraction."""

from __future__ import annotations

from noema.contracts.actions import TacticAction
from noema.contracts.hypergraph import HyperEdge, HyperNode, ProofHypergraph
from noema.contracts.states import ProofGoal, ProofState


class HypergraphEngine:
    """Orchestrates AND/OR hypergraph structure, metavariable unification, and proof extraction."""

    def __init__(self, root_state: ProofState):
        self.root_node_id = f"node_{root_state.state_id[:12]}"
        root_goal = root_state.get_open_goals()[0] if root_state.get_open_goals() else ProofGoal(
            goal_id="root_empty", target="True", mvar_id="?m.0"
        )
        self.hypergraph = ProofHypergraph(root_node_id=self.root_node_id)
        
        root_node = HyperNode(
            node_id=self.root_node_id,
            state_id=root_state.state_id,
            goal=root_goal,
            depth=0,
        )
        self.hypergraph.add_node(root_node)
        self.state_to_node_id: dict[str, str] = {root_state.state_id: self.root_node_id}

    def register_transition(
        self,
        parent_state_id: str,
        child_state: ProofState,
        action: TacticAction,
        execution_time_ms: float = 0.0,
    ) -> HyperEdge:
        """Add transition edge and child nodes to the hypergraph."""
        parent_node_id = self.state_to_node_id[parent_state_id]
        edge_id = f"edge_{len(self.hypergraph.edges)}_{action.action_id}"

        child_node_ids = []
        for idx, g in enumerate(child_state.get_open_goals()):
            c_node_id = f"node_{child_state.state_id[:8]}_{idx}"
            if c_node_id not in self.hypergraph.nodes:
                c_node = HyperNode(
                    node_id=c_node_id,
                    state_id=child_state.state_id,
                    goal=g,
                    depth=child_state.depth,
                )
                self.hypergraph.add_node(c_node)
            child_node_ids.append(c_node_id)

        self.state_to_node_id[child_state.state_id] = (
            child_node_ids[0] if child_node_ids else f"solved_{child_state.state_id[:8]}"
        )

        edge = HyperEdge(
            edge_id=edge_id,
            source_node_id=parent_node_id,
            target_node_ids=child_node_ids,
            action=action,
            execution_time_ms=execution_time_ms,
        )
        self.hypergraph.add_edge(edge)

        # Check if child state is solved
        if not child_state.has_open_goals():
            edge.is_proved = True
            self.hypergraph.update_proved_status(parent_node_id)

        return edge

    def is_root_proved(self) -> bool:
        return self.hypergraph.nodes[self.root_node_id].is_proved

    def extract_proof_script(self) -> list[str]:
        """Extract sequence of tactics that successfully prove the root goal."""
        if not self.is_root_proved():
            return []

        script: list[str] = []
        
        # Traverse proven path
        def visit(node_id: str):
            node = self.hypergraph.nodes.get(node_id)
            if not node:
                return
            for e_id in node.outgoing_edge_ids:
                edge = self.hypergraph.edges[e_id]
                if edge.is_proved:
                    script.append(edge.action.raw_tactic)
                    for child_id in edge.target_node_ids:
                        visit(child_id)
                    break

        visit(self.root_node_id)
        return script
