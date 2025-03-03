"""Monte Carlo Tree Search (MCTS) with PUCT over AND/OR proof hypergraphs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence
from noema.contracts.actions import TacticAction
from noema.contracts.states import ProofState
from noema.contracts.theorems import TheoremSpec
from noema.formal.bridge import FormalBridge
from noema.formal.types import TransitionStatus


@dataclass
class MCTSHyperedge:
    """A directed hyperedge connecting a proof state to one or more subgoal states."""
    edge_id: str
    action: TacticAction
    prior_prob: float
    child_state_ids: list[str] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    is_proved: bool = False
    is_failed: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass
class MCTSNode:
    """State node in the proof search hypergraph."""
    state_id: str
    state: ProofState
    depth: int = 0
    is_expanded: bool = False
    is_proved: bool = False
    is_failed: bool = False
    outgoing_edges: list[MCTSHyperedge] = field(default_factory=list)
    visit_count: int = 0
    value_estimate: float = 0.0
    history: list[str] = field(default_factory=list)


class MCTSProofSearch:
    """PUCT-guided Monte Carlo Tree Search designed for theorem proving in Lean 4."""

    def __init__(
        self,
        bridge: FormalBridge,
        c_puct: float = 1.414,
        max_depth: int = 30,
        virtual_loss: float = 1.0,
    ):
        self.bridge = bridge
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.virtual_loss = virtual_loss
        self.nodes: dict[str, MCTSNode] = {}

    def search(
        self,
        theorem: TheoremSpec,
        initial_state: ProofState,
        policy_fn: Callable[[ProofState], list[tuple[TacticAction, float]]],
        value_fn: Callable[[ProofState], float],
        max_iterations: int = 100,
    ) -> tuple[bool, list[TacticAction], dict[str, int]]:
        """Run MCTS iterations to discover a valid closed proof."""
        root_id = initial_state.state_id
        root_node = MCTSNode(
            state_id=root_id,
            state=initial_state,
            depth=0,
            value_estimate=value_fn(initial_state),
        )
        self.nodes[root_id] = root_node

        if initial_state.is_solved:
            root_node.is_proved = True
            return True, [], {"iterations": 0, "nodes": 1}

        iterations_run = 0
        for it in range(max_iterations):
            iterations_run += 1
            path: list[tuple[str, MCTSHyperedge | None]] = []
            curr_node = root_node
            depth = 0

            # 1. Selection
            while curr_node.is_expanded and not curr_node.is_proved and not curr_node.is_failed:
                best_edge = self._select_best_edge(curr_node)
                if best_edge is None:
                    curr_node.is_failed = True
                    break

                path.append((curr_node.state_id, best_edge))
                
                # For AND-hyperedges, select unproved child with highest exploration priority
                unproved_children = [
                    self.nodes[cid] for cid in best_edge.child_state_ids
                    if cid in self.nodes and not self.nodes[cid].is_proved
                ]
                if not unproved_children:
                    break
                
                # Pick child with least visits among open goals
                curr_node = min(unproved_children, key=lambda n: n.visit_count)
                depth += 1
                if depth >= self.max_depth:
                    curr_node.is_failed = True
                    break

            if root_node.is_proved:
                break

            # 2. Expansion & Evaluation
            if not curr_node.is_proved and not curr_node.is_failed and not curr_node.is_expanded:
                leaf_value = self._expand_node(theorem, curr_node, policy_fn, value_fn)
            else:
                leaf_value = 1.0 if curr_node.is_proved else 0.0

            # 3. Backpropagation
            self._backpropagate(path, leaf_value)

            if root_node.is_proved:
                break

        # Extract proof actions if proved
        proved = root_node.is_proved
        proof_actions: list[TacticAction] = []
        if proved:
            proof_actions = self._extract_proof_actions(root_node)

        stats = {
            "iterations": iterations_run,
            "nodes": len(self.nodes),
            "root_visits": root_node.visit_count,
        }
        return proved, proof_actions, stats

    def _select_best_edge(self, node: MCTSNode) -> MCTSHyperedge | None:
        """Select edge maximizing PUCT formula: Q(s, a) + U(s, a)."""
        valid_edges = [e for e in node.outgoing_edges if not e.is_failed]
        if not valid_edges:
            return None

        total_visits = sum(e.visit_count for e in valid_edges)
        sqrt_total = math.sqrt(max(1, total_visits))

        best_score = -float("inf")
        best_edge: MCTSHyperedge | None = None

        for edge in valid_edges:
            if edge.is_proved:
                return edge
            u_score = self.c_puct * edge.prior_prob * sqrt_total / (1 + edge.visit_count)
            q_score = edge.q_value
            score = q_score + u_score

            if score > best_score:
                best_score = score
                best_edge = edge

        return best_edge

    def _expand_node(
        self,
        theorem: TheoremSpec,
        node: MCTSNode,
        policy_fn: Callable[[ProofState], list[tuple[TacticAction, float]]],
        value_fn: Callable[[ProofState], float],
    ) -> float:
        """Expand node with candidate tactics and evaluate transition outcomes."""
        candidates = policy_fn(node.state)
        node.is_expanded = True

        if not candidates:
            node.is_failed = True
            return 0.0

        edge_counter = 0
        proved_any = False

        for action, prior in candidates:
            edge_counter += 1
            # Step in formal bridge with accumulated tactic history
            res = self.bridge.step(theorem, node.state, action, tactic_history=node.history)
            if not res.is_valid_transition:
                continue

            child_ids: list[str] = []
            if res.after_state is not None:
                c_id = res.after_state.state_id
                child_ids.append(c_id)
                if c_id not in self.nodes:
                    c_node = MCTSNode(
                        state_id=c_id,
                        state=res.after_state,
                        depth=node.depth + 1,
                        history=node.history + [action.raw_tactic],
                        is_proved=res.status == TransitionStatus.PROVED or res.after_state.is_solved,
                        value_estimate=1.0 if res.status == TransitionStatus.PROVED else value_fn(res.after_state),
                    )
                    self.nodes[c_id] = c_node

            edge = MCTSHyperedge(
                edge_id=f"{node.state_id}_e{edge_counter}",
                action=action,
                prior_prob=prior,
                child_state_ids=child_ids,
                is_proved=res.status == TransitionStatus.PROVED,
            )
            node.outgoing_edges.append(edge)

            if edge.is_proved:
                proved_any = True

        if proved_any:
            node.is_proved = True
            return 1.0

        if not node.outgoing_edges:
            node.is_failed = True
            return 0.0

        return node.value_estimate

    def _backpropagate(
        self,
        path: list[tuple[str, MCTSHyperedge | None]],
        value: float,
    ) -> None:
        """Propagate visit counts and proof values back up the tree."""
        for state_id, edge in reversed(path):
            node = self.nodes.get(state_id)
            if node is not None:
                node.visit_count += 1

            if edge is not None:
                edge.visit_count += 1
                edge.total_value += value

                # Check if all children of this hyperedge are proved
                if edge.child_state_ids:
                    all_ch_proved = all(
                        self.nodes[cid].is_proved
                        for cid in edge.child_state_ids
                        if cid in self.nodes
                    )
                    if all_ch_proved:
                        edge.is_proved = True
                        if node is not None:
                            node.is_proved = True

    def _extract_proof_actions(self, root: MCTSNode) -> list[TacticAction]:
        """Extract a linear or topological sequence of tactic actions that solves the goal."""
        actions: list[TacticAction] = []
        curr = root

        while curr and curr.is_proved:
            proved_edges = [e for e in curr.outgoing_edges if e.is_proved]
            if not proved_edges:
                break
            best_e = proved_edges[0]
            actions.append(best_e.action)

            # Follow first child
            if best_e.child_state_ids:
                curr = self.nodes.get(best_e.child_state_ids[0])
            else:
                break

        return actions
