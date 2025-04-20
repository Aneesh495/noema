"""Explainable congruence closure on ground equational graphs and e-graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class JustificationKind(Enum):
    AXIOM = "axiom"
    CONGRUENCE = "congruence"
    REFLEXIVITY = "reflexivity"


@dataclass(frozen=True)
class Justification:
    """Explains why two terms are equivalent in the congruence closure forest."""
    kind: JustificationKind
    source_node: int
    target_node: int
    axiom_name: str = ""
    argument_justifications: tuple[Justification, ...] = ()


@dataclass
class ENode:
    """Node in the congruence closure graph representing a constant or function application."""
    node_id: int
    symbol: str
    args: tuple[int, ...] = ()
    parent: int = field(init=False)
    rank: int = 0
    justification_to_parent: Justification | None = None

    def __post_init__(self):
        self.parent = self.node_id


@dataclass(frozen=True)
class ProofStep:
    """Human-readable and formal proof step."""
    step_id: int
    lhs_repr: str
    rhs_repr: str
    rationale: str


class CongruenceClosure:
    """Nelson-Oppen explainable congruence closure algorithm over ground equational terms."""

    def __init__(self):
        self.nodes: list[ENode] = []
        self.term_to_id: dict[str, int] = {}
        # Signature table: (symbol, tuple of canon arg IDs) -> node_id
        self.sig_table: dict[tuple[str, tuple[int, ...]], int] = {}
        # Use lists of parents to propagate congruence when classes merge
        self.use_list: dict[int, list[int]] = {}

    def add_term(self, expr: str) -> int:
        """Parse or register a ground term string, returning its node ID."""
        s = expr.strip()
        if s in self.term_to_id:
            return self.term_to_id[s]

        if "(" in s and s.endswith(")"):
            sym = s[:s.index("(")].strip()
            inside = s[s.index("(") + 1 : -1].strip()
            # Split top-level args
            args_str: list[str] = []
            depth = 0
            cur: list[str] = []
            for ch in inside:
                if ch == "(":
                    depth += 1
                    cur.append(ch)
                elif ch == ")":
                    depth -= 1
                    cur.append(ch)
                elif ch == "," and depth == 0:
                    args_str.append("".join(cur).strip())
                    cur = []
                else:
                    cur.append(ch)
            if cur:
                args_str.append("".join(cur).strip())
            
            arg_ids = tuple(self.add_term(a) for a in args_str)
            nid = len(self.nodes)
            node = ENode(node_id=nid, symbol=sym, args=arg_ids)
            self.nodes.append(node)
            self.term_to_id[s] = nid
            self.use_list[nid] = []

            # Register in use-lists of arguments
            for aid in arg_ids:
                if aid not in self.use_list:
                    self.use_list[aid] = []
                self.use_list[aid].append(nid)

            # Check if congruent to an existing node
            sig = (sym, tuple(self.find(a) for a in arg_ids))
            if sig in self.sig_table:
                existing = self.sig_table[sig]
                self.merge(nid, existing, Justification(
                    kind=JustificationKind.CONGRUENCE,
                    source_node=nid,
                    target_node=existing,
                ))
            else:
                self.sig_table[sig] = nid

            return nid

        # Constant symbol
        nid = len(self.nodes)
        node = ENode(node_id=nid, symbol=s, args=())
        self.nodes.append(node)
        self.term_to_id[s] = nid
        self.use_list[nid] = []
        self.sig_table[(s, ())] = nid
        return nid

    def find(self, i: int) -> int:
        """Find representative without path compression to preserve explanation trees."""
        curr = i
        while self.nodes[curr].parent != curr:
            curr = self.nodes[curr].parent
        return curr

    def merge(self, i: int, j: int, just: Justification) -> bool:
        """Merge equivalence classes of node i and j with justification."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return False

        # Union by rank
        if self.nodes[root_i].rank < self.nodes[root_j].rank:
            root_i, root_j = root_j, root_i
            i, j = j, i
            just = Justification(
                kind=just.kind,
                source_node=j,
                target_node=i,
                axiom_name=just.axiom_name,
                argument_justifications=just.argument_justifications,
            )

        # Make root_j point to root_i
        self.nodes[root_j].parent = root_i
        self.nodes[root_j].justification_to_parent = just
        if self.nodes[root_i].rank == self.nodes[root_j].rank:
            self.nodes[root_i].rank += 1

        # Check congruence propagation over parents in use_list
        parents_to_check = list(self.use_list.get(root_j, [])) + list(self.use_list.get(j, []))
        for p in parents_to_check:
            p_node = self.nodes[p]
            sig = (p_node.symbol, tuple(self.find(a) for a in p_node.args))
            if sig in self.sig_table:
                other_p = self.sig_table[sig]
                if self.find(p) != self.find(other_p):
                    self.merge(p, other_p, Justification(
                        kind=JustificationKind.CONGRUENCE,
                        source_node=p,
                        target_node=other_p,
                    ))
            else:
                self.sig_table[sig] = p

        return True

    def assert_equality(self, term_a: str, term_b: str, axiom_name: str = "") -> bool:
        """Assert an equality axiom between two ground terms."""
        id_a = self.add_term(term_a)
        id_b = self.add_term(term_b)
        just = Justification(
            kind=JustificationKind.AXIOM,
            source_node=id_a,
            target_node=id_b,
            axiom_name=axiom_name or f"{term_a} = {term_b}",
        )
        return self.merge(id_a, id_b, just)

    def are_equal(self, term_a: str, term_b: str) -> bool:
        """Query whether two ground terms are equivalent under the congruence closure."""
        id_a = self.add_term(term_a)
        id_b = self.add_term(term_b)
        return self.find(id_a) == self.find(id_b)

    def term_repr(self, node_id: int) -> str:
        """Format node ID back to its ground term representation."""
        node = self.nodes[node_id]
        if not node.args:
            return node.symbol
        inner = ", ".join(self.term_repr(a) for a in node.args)
        return f"{node.symbol}({inner})"

    def explain(self, term_a: str, term_b: str) -> list[ProofStep]:
        """Produce an explainable chain of equalities connecting term_a and term_b."""
        id_a = self.add_term(term_a)
        id_b = self.add_term(term_b)
        if self.find(id_a) != self.find(id_b):
            return []

        # Find path to root for id_a and id_b
        path_a: list[int] = [id_a]
        curr = id_a
        while self.nodes[curr].parent != curr:
            curr = self.nodes[curr].parent
            path_a.append(curr)

        path_b: list[int] = [id_b]
        curr = id_b
        while self.nodes[curr].parent != curr:
            curr = self.nodes[curr].parent
            path_b.append(curr)

        # Find Lowest Common Ancestor (LCA)
        lca = path_a[-1]
        set_b = set(path_b)
        for node in path_a:
            if node in set_b:
                lca = node
                break

        steps: list[ProofStep] = []
        step_counter = 1

        # Trace id_a up to LCA
        idx = 0
        while path_a[idx] != lca:
            src = path_a[idx]
            dst = path_a[idx + 1]
            just = self.nodes[src].justification_to_parent
            rat = just.axiom_name if just and just.axiom_name else (
                "congruence" if just and just.kind == JustificationKind.CONGRUENCE else "definition"
            )
            steps.append(ProofStep(
                step_id=step_counter,
                lhs_repr=self.term_repr(src),
                rhs_repr=self.term_repr(dst),
                rationale=rat,
            ))
            step_counter += 1
            idx += 1

        # Trace LCA down to id_b (reverse of path_b from lca)
        lca_idx_b = path_b.index(lca)
        for idx in range(lca_idx_b - 1, -1, -1):
            src = path_b[idx + 1]
            dst = path_b[idx]
            just = self.nodes[dst].justification_to_parent
            rat = just.axiom_name if just and just.axiom_name else (
                "congruence" if just and just.kind == JustificationKind.CONGRUENCE else "definition"
            )
            steps.append(ProofStep(
                step_id=step_counter,
                lhs_repr=self.term_repr(src),
                rhs_repr=self.term_repr(dst),
                rationale=f"symmetric application of {rat}",
            ))
            step_counter += 1

        return steps
