# Walkthrough 3: Proof Search with AND/OR Hypergraphs and MCTS

This walkthrough explains how Noema navigates complex formal proofs using Monte Carlo Tree Search (MCTS) guided by AND/OR proof hypergraphs and equational reasoning engines.

## Hypergraph Proof Search Mechanics

Unlike simple single-path game trees, formal mathematical proof trees feature branching subgoals:
- **OR-Nodes**: Represent open proof goals. Any valid branch that solves this goal is sufficient (disjunctive).
- **AND-Hyperedges**: Represent multi-goal tactic applications (e.g. `constructor`, `split`, `apply And.intro`). All generated subgoals must be solved concurrently (conjunctive).

## Search Algorithm Execution

Execute proof search programmatically using `MCTSSearchEngine`:

```python
from noema.search.mcts import MCTSSearchEngine, MCTSConfig
from noema.formal.repl import LeanREPLSession
from noema.models.transformer import ProverTransformer

# Initialize environment and neural guide
repl = LeanREPLSession(project_dir="lean")
repl.start()

model = ProverTransformer.load_pretrained("checkpoints/model_24block_final.pt")

config = MCTSConfig(
    max_expansions=200,
    c_puct=1.414,
    temperature=0.7,
    max_depth=30,
)

search_engine = MCTSSearchEngine(
    repl=repl,
    model=model,
    config=config,
)

theorem_statement = "theorem demo (P Q : Prop) : P ∧ Q → P := by"
proof_result = search_engine.prove(theorem_statement)

if proof_result.success:
    print("Proof discovered successfully.")
    print("Extracted tactic script:")
    for step in proof_result.tactics:
        print(f"  {step}")
else:
    print("Search budget exhausted before proof resolution.")
```

## Hybridizing with Equational Rewriting

When algebraic identities or equality chains are present in local hypotheses, Noema integrates a congruence closure engine:

```python
from noema.search.congruence import CongruenceClosure

# Build equational graph from hypotheses: a = b, b = c, f(a) = d
cc = CongruenceClosure()
cc.merge("a", "b")
cc.merge("b", "c")
cc.add_function("f", ["a"], "d")

# Check whether f(c) = d holds
is_congruent = cc.are_congruent("f(c)", "d")
print(f"Are f(c) and d congruent: {is_congruent}")

# Extract formal Lean 4 tactic sequence witnessing the equivalence
witness_proof = cc.explain_equivalence("f(c)", "d")
print(f"Generated Lean tactic: {witness_proof}")
```

## Exporting Proof Visualizations

Noema can export search hypergraphs and solved proof DAGs to DOT, Cytoscape JSON, or SVG formats:

```python
from noema.artifacts.graph_export import export_proof_graph

# Export to DOT file for Graphviz rendering
dot_path = "artifacts/proof_tree.dot"
export_proof_graph(proof_result.graph, target_path=dot_path, format="dot")

# Export to Cytoscape JSON for interactive web viewers
json_path = "artifacts/proof_tree.json"
export_proof_graph(proof_result.graph, target_path=json_path, format="cytoscape")
```
