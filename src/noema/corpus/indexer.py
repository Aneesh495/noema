"""Chunked disk-backed storage and SQLite indexing for theorem proving corpora."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence
from noema.contracts.actions import TacticAction
from noema.contracts.theorems import TheoremSpec


@dataclass(frozen=True)
class IndexedTraceStep:
    """An individual indexed tactic step within a formal proof trace."""
    step_id: str
    theorem_id: str
    step_idx: int
    goal_state_text: str
    tactic_text: str
    premises_used: list[str]
    goal_fingerprint: str


class CorpusIndexer:
    """SQLite-backed indexer and chunked storage engine for theorem corpora."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS theorems (
                    theorem_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    module TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    family TEXT NOT NULL,
                    proof_length INTEGER NOT NULL,
                    is_proved INTEGER NOT NULL,
                    spec_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trace_steps (
                    step_id TEXT PRIMARY KEY,
                    theorem_id TEXT NOT NULL,
                    step_idx INTEGER NOT NULL,
                    goal_state_text TEXT NOT NULL,
                    tactic_text TEXT NOT NULL,
                    premises_json TEXT NOT NULL,
                    goal_fingerprint TEXT NOT NULL,
                    FOREIGN KEY (theorem_id) REFERENCES theorems(theorem_id)
                );

                CREATE INDEX IF NOT EXISTS idx_theorems_family ON theorems(family);
                CREATE INDEX IF NOT EXISTS idx_theorems_name ON theorems(name);
                CREATE INDEX IF NOT EXISTS idx_trace_fingerprint ON trace_steps(goal_fingerprint);
                CREATE INDEX IF NOT EXISTS idx_trace_thm_step ON trace_steps(theorem_id, step_idx);
            """)

    def index_theorem(
        self,
        theorem: TheoremSpec,
        proof_steps: Sequence[str] | None = None,
        goal_states: Sequence[str] | None = None,
    ) -> None:
        """Insert or replace a theorem and its corresponding proof trace steps."""
        proof_len = len(proof_steps) if proof_steps else 0
        is_proved = int(proof_len > 0)
        spec_json = json.dumps(theorem.model_dump())

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO theorems 
                (theorem_id, name, module, statement, family, proof_length, is_proved, spec_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    theorem.theorem_id,
                    theorem.name,
                    theorem.module,
                    theorem.statement,
                    theorem.family,
                    proof_len,
                    is_proved,
                    spec_json,
                ),
            )

            # Insert trace steps if provided
            if proof_steps:
                conn.execute("DELETE FROM trace_steps WHERE theorem_id = ?", (theorem.theorem_id,))
                step_rows = []
                for idx, step_tactic in enumerate(proof_steps):
                    step_id = f"{theorem.theorem_id}_s{idx}"
                    g_text = goal_states[idx] if goal_states and idx < len(goal_states) else ""
                    act = TacticAction.from_string(step_tactic)
                    fp = hashlib.sha256(g_text.encode("utf-8")).hexdigest()[:16]
                    prems_json = json.dumps(act.premises)
                    step_rows.append((
                        step_id,
                        theorem.theorem_id,
                        idx,
                        g_text,
                        step_tactic,
                        prems_json,
                        fp,
                    ))

                conn.executemany(
                    """
                    INSERT INTO trace_steps
                    (step_id, theorem_id, step_idx, goal_state_text, tactic_text, premises_json, goal_fingerprint)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    step_rows,
                )

    def count_theorems(self, family: str | None = None) -> int:
        """Count total indexed theorems, optionally filtering by family."""
        with self._get_connection() as conn:
            if family:
                cur = conn.execute("SELECT COUNT(*) FROM theorems WHERE family = ?", (family,))
            else:
                cur = conn.execute("SELECT COUNT(*) FROM theorems")
            return cur.fetchone()[0]

    def count_steps(self) -> int:
        """Count total recorded proof steps."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM trace_steps")
            return cur.fetchone()[0]

    def get_theorem(self, theorem_id: str) -> TheoremSpec | None:
        """Retrieve a TheoremSpec by theorem_id."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT spec_json FROM theorems WHERE theorem_id = ?", (theorem_id,))
            row = cur.fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            return TheoremSpec(**data)

    def iter_trace_steps(self, batch_size: int = 500) -> Iterator[IndexedTraceStep]:
        """Stream trace steps in chunked batches to keep RAM usage strictly bounded."""
        offset = 0
        while True:
            with self._get_connection() as conn:
                cur = conn.execute(
                    """
                    SELECT step_id, theorem_id, step_idx, goal_state_text, tactic_text, premises_json, goal_fingerprint
                    FROM trace_steps
                    ORDER BY rowid ASC
                    LIMIT ? OFFSET ?
                    """,
                    (batch_size, offset),
                )
                rows = cur.fetchall()
                if not rows:
                    break

                for r in rows:
                    yield IndexedTraceStep(
                        step_id=r[0],
                        theorem_id=r[1],
                        step_idx=r[2],
                        goal_state_text=r[3],
                        tactic_text=r[4],
                        premises_used=json.loads(r[5]),
                        goal_fingerprint=r[6],
                    )

                offset += len(rows)

    def find_theorems_using_premise(self, premise_name: str) -> list[str]:
        """Find theorem IDs whose proofs utilize a specified premise."""
        with self._get_connection() as conn:
            # Match premise within JSON list
            pattern = f'%"{premise_name}"%'
            cur = conn.execute(
                "SELECT DISTINCT theorem_id FROM trace_steps WHERE premises_json LIKE ?",
                (pattern,),
            )
            return [row[0] for row in cur.fetchall()]
