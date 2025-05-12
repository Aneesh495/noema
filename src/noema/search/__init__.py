"""Proof-hypergraph search subsystem: budgets, priority heuristics, and best-first search engine."""

from noema.search.backtracking import (
    BacktrackDecision,
    ConflictAnalyzer,
    DependencyDirectedBacktracker,
    Nogood,
    NogoodStore,
)
from noema.search.budget import SearchAccounting, SearchBudget, TerminationReason
from noema.search.canonical import CanonicalGoal, CyclePruner, GoalCanonicalizer
from noema.search.engine import ProofSearchEngine, SearchResult
from noema.search.hypergraph_engine import HypergraphEngine
from noema.search.mcts import MCTSHyperedge, MCTSNode, MCTSProofSearch
from noema.search.priority import PriorityScorer, SearchStrategy
from noema.search.beam import (
    BeamCandidate,
    BeamGroup,
    DiverseBeamSearch,
    SpeculativeTacticTree,
)
from noema.search.congruence import CongruenceClosure, Justification, ProofStep
from noema.search.rewriting import (
    EquationalProver,
    FirstOrderTerm,
    KnuthBendixCompletion,
    RewriteRule,
    TermRewriter,
    lexicographic_path_order,
)
from noema.search.tableau import (
    And,
    Atom,
    Formula,
    Imp,
    Not,
    Or,
    SemanticTableauProver,
    SignedFormula,
    TableauBranch,
)
from noema.search.unification import (
    CoupledMetavariableTracker,
    Term,
    UnificationEngine,
    UnificationError,
)

__all__ = [
    "SearchBudget",
    "SearchAccounting",
    "TerminationReason",
    "HypergraphEngine",
    "SearchStrategy",
    "PriorityScorer",
    "ProofSearchEngine",
    "SearchResult",
    "MCTSProofSearch",
    "MCTSNode",
    "MCTSHyperedge",
    "GoalCanonicalizer",
    "CanonicalGoal",
    "CyclePruner",
    "Nogood",
    "NogoodStore",
    "ConflictAnalyzer",
    "DependencyDirectedBacktracker",
    "BacktrackDecision",
    "Term",
    "UnificationEngine",
    "UnificationError",
    "CoupledMetavariableTracker",
    "FirstOrderTerm",
    "RewriteRule",
    "TermRewriter",
    "KnuthBendixCompletion",
    "EquationalProver",
    "lexicographic_path_order",
    "CongruenceClosure",
    "Justification",
    "ProofStep",
    "DiverseBeamSearch",
    "BeamCandidate",
    "BeamGroup",
    "SpeculativeTacticTree",
    "SemanticTableauProver",
    "Formula",
    "Atom",
    "Not",
    "And",
    "Or",
    "Imp",
    "SignedFormula",
    "TableauBranch",
]
