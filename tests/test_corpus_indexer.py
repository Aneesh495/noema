"""Tests for SQLite-backed corpus indexing and streaming."""

import tempfile
from pathlib import Path
import pytest
from noema.contracts.theorems import TheoremSpec
from noema.corpus.indexer import CorpusIndexer


def test_corpus_indexer_insert_and_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_corpus.db"
        indexer = CorpusIndexer(db_path)

        thm1 = TheoremSpec(
            theorem_id="thm_add_comm",
            name="add_comm_demo",
            module="Math.Algebra",
            statement="∀ (x y : Nat), x + y = y + x",
            family="algebra",
        )
        proof1 = ["intro x y", "rw [Nat.add_comm]", "rfl"]
        goals1 = ["⊢ ∀ x y, x + y = y + x", "x y : Nat ⊢ x + y = y + x", "x y : Nat ⊢ y + x = y + x"]

        indexer.index_theorem(thm1, proof_steps=proof1, goal_states=goals1)

        thm2 = TheoremSpec(
            theorem_id="thm_refl",
            name="refl_demo",
            module="Math.Logic",
            statement="∀ (x : Nat), x = x",
            family="logic",
        )
        proof2 = ["intro x", "rfl"]
        indexer.index_theorem(thm2, proof_steps=proof2)

        # Count checks
        assert indexer.count_theorems() == 2
        assert indexer.count_theorems(family="algebra") == 1
        assert indexer.count_theorems(family="logic") == 1
        assert indexer.count_theorems(family="topology") == 0

        # Retrieve theorem
        fetched = indexer.get_theorem("thm_add_comm")
        assert fetched is not None
        assert fetched.name == "add_comm_demo"

        # Count steps
        assert indexer.count_steps() == 5

        # Streaming iterator
        steps = list(indexer.iter_trace_steps(batch_size=2))
        assert len(steps) == 5
        assert steps[0].theorem_id == "thm_add_comm"
        assert steps[0].tactic_text == "intro x y"

        # Premise lookup
        using_nat_add_comm = indexer.find_theorems_using_premise("Nat.add_comm")
        assert "thm_add_comm" in using_nat_add_comm
        assert "thm_refl" not in using_nat_add_comm
