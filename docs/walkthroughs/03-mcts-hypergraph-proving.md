# Walkthrough 3: Proof Search with AND/OR Hypergraphs and MCTS

A proof goal can have several candidate tactics. Choosing a tactic is an OR decision. If the tactic creates several subgoals, the resulting AND edge succeeds once all of them are proved.

## Search for a proof

This example uses a fixed tactic policy to make the search repeatable. Lean checks the transitions. The independent checker replays the returned tactics.

```python
from noema.contracts.actions import TacticAction
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.theorems import TheoremSpec
from noema.formal.bridge import FormalBridge
from noema.formal.checker import IndependentChecker
from noema.search.mcts import MCTSProofSearch

environment = EnvironmentSpec(
    toolchain="leanprover/lean4:v4.8.0",
    imported_modules=["Init.Data.Nat.Basic"],
)
theorem = TheoremSpec(
    theorem_id="nat_reflexive",
    name="nat_reflexive",
    module="Init.Data.Nat.Basic",
    statement="∀ (x : Nat), x = x",
)
bridge = FormalBridge(environment)
initial, _ = bridge.get_initial_state(theorem)
search = MCTSProofSearch(bridge, max_depth=5)


def policy(state):
    tactic = "intro x" if state.depth == 0 else "rfl"
    return [(TacticAction.from_string(tactic), 1.0)]


proved, actions, stats = search.search(
    theorem, initial, policy_fn=policy, value_fn=lambda state: 0.5, max_iterations=10
)
if not proved:
    raise RuntimeError(f"Search ended after {stats['iterations']} iterations")

script = "\n".join(action.raw_tactic for action in actions)
_, audit = IndependentChecker(environment).verify_proof(theorem, script)
assert audit.is_valid
print(script)
```

## Explain a congruence

Register both function applications and then assert equality of their arguments. The explanation records how the closure engine derived their equivalence.

```python
from noema.search.congruence import CongruenceClosure

closure = CongruenceClosure()
closure.add_term("f(a)")
closure.add_term("f(c)")
closure.assert_equality("a", "b", axiom_name="hyp_ab")
closure.assert_equality("b", "c", axiom_name="hyp_bc")

assert closure.are_equal("f(a)", "f(c)")
for step in closure.explain("f(a)", "f(c)"):
    print(step.rationale)
```

## Export the search graph

After running the search example, pass its states and edges to the graph exporter. A DOT file can be rendered with Graphviz.

```python
from pathlib import Path
from noema.artifacts.graph_export import ProofGraphExporter

exporter = ProofGraphExporter()
for node in search.nodes.values():
    exporter.add_node(
        node.state_id,
        label=f"Depth {node.depth}",
        is_proved=node.is_proved,
        is_root=node.state_id == initial.state_id,
        depth=node.depth,
    )
    for edge in node.outgoing_edges:
        exporter.add_hyperedge(
            edge.edge_id, node.state_id, edge.child_state_ids, edge.action.raw_tactic
        )

Path("proof.dot").write_text(exporter.to_dot(), encoding="utf-8")
```
