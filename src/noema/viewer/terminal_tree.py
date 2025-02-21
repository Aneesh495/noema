"""Terminal visualization for proof hypergraphs, transition trees, and model scores."""

from __future__ import annotations

import sys
from noema.contracts.certificates import ProofCertificate
from noema.contracts.hypergraph import ProofHypergraph
from noema.search.budget import SearchAccounting
from noema.search.engine import SearchResult


class TerminalViewer:
    """Formats and prints search trees, model scores, and verified certificates."""

    @classmethod
    def render_search_result(cls, result: SearchResult, stream=sys.stdout) -> None:
        thm = result.theorem
        acc = result.accounting
        hg = result.hypergraph_engine.hypergraph

        print("=" * 70, file=stream)
        print(f"NOEMA PROOF SEARCH REPORT: {thm.name}", file=stream)
        print("=" * 70, file=stream)
        print(f"Theorem ID:       {thm.theorem_id}", file=stream)
        print(f"Statement:        {thm.statement}", file=stream)
        print(f"Outcome:          {'VERIFIED PROOF' if result.is_proved else 'UNSOLVED'}", file=stream)
        print(f"Termination:      {acc.termination_reason.value}", file=stream)
        print(f"Search Time:      {acc.total_time_ms:.1f} ms (Kernel: {acc.kernel_time_ms:.1f} ms)", file=stream)
        print(f"Transitions:      {acc.attempted_transitions} attempted, {acc.valid_transitions} valid", file=stream)
        print(f"Proposals:        {acc.generated_proposals} generated, {acc.rejected_actions} rejected", file=stream)
        print(f"Hypergraph Size:  {len(hg.nodes)} nodes, {len(hg.edges)} edges", file=stream)
        print("-" * 70, file=stream)

        if result.is_proved:
            print("ACCEPTED TACTIC PROOF SCRIPT:", file=stream)
            for idx, tac in enumerate(result.proof_script, 1):
                print(f"  {idx}. {tac}", file=stream)
            print("-" * 70, file=stream)

        if result.certificate:
            cert = result.certificate
            print("FRESH KERNEL VERIFICATION CERTIFICATE:", file=stream)
            print(f"  Status:          {cert.status.value.upper()}", file=stream)
            print(f"  Certificate Hash: {cert.certificate_hash[:32]}...", file=stream)
            print(f"  Permitted Axioms: {cert.used_axioms or ['[none - purely definitional]']}", file=stream)
            print(f"  Kernel Output:   {cert.kernel_diagnostics}", file=stream)
        print("=" * 70, file=stream)

    @classmethod
    def render_hypergraph_tree(cls, hg: ProofHypergraph, stream=sys.stdout) -> None:
        """Render tree structure of proof hypergraph to stream."""
        print("\nPROOF HYPERGRAPH AND/OR TREE:", file=stream)
        visited = set()

        def _print_node(node_id: str, prefix: str = "", is_last: bool = True):
            if node_id in visited:
                print(f"{prefix}{'└── ' if is_last else '├── '}[CYCLE/REF] Node {node_id}", file=stream)
                return
            visited.add(node_id)
            node = hg.nodes.get(node_id)
            if not node:
                return

            status_str = "[PROVED]" if node.is_proved else "[OPEN]"
            tgt_short = node.goal.target[:40] + ("..." if len(node.goal.target) > 40 else "")
            print(f"{prefix}{'└── ' if is_last else '├── '}{status_str} (Depth {node.depth}) ⊢ {tgt_short}", file=stream)

            new_prefix = prefix + ("    " if is_last else "│   ")
            edges = [hg.edges[e_id] for e_id in node.outgoing_edge_ids if e_id in hg.edges]
            for i, edge in enumerate(edges):
                e_last = (i == len(edges) - 1)
                edge_status = "✓" if edge.is_proved else "✗"
                print(f"{new_prefix}{'└── ' if e_last else '├── '}[Tactic: {edge.action.raw_tactic}] {edge_status}", file=stream)
                edge_prefix = new_prefix + ("    " if e_last else "│   ")
                for j, child_id in enumerate(edge.target_node_ids):
                    _print_node(child_id, prefix=edge_prefix, is_last=(j == len(edge.target_node_ids) - 1))

        _print_node(hg.root_node_id)
