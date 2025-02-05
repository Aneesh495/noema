"""Lean 4 goal output and state parser."""

from __future__ import annotations

import re
from noema.contracts.states import LocalHypothesis, ProofGoal, ProofState


class GoalParser:
    """Parses raw Lean 4 goal state output into structured ProofGoal and ProofState objects."""

    _MVAR_PATTERN = re.compile(r"\?m\.\d+")
    _GOAL_SEP = re.compile(r"(?:^|\n)(?:case\s+[^\n]+|\d+\s+goals?|goal\s+\d+:)", re.MULTILINE)

    @classmethod
    def parse_goals(cls, raw_output: str, prefix_mvar: str = "?m") -> list[ProofGoal]:
        text = raw_output.strip()
        if not text or "no goals" in text.lower():
            return []

        # Split multiple goals if present
        sections = []
        lines = text.splitlines()
        current_chunk: list[str] = []

        for line in lines:
            if line.startswith("case ") and current_chunk:
                sections.append("\n".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        if current_chunk:
            sections.append("\n".join(current_chunk))

        if not sections:
            sections = [text]

        parsed_goals: list[ProofGoal] = []
        all_found_mvars: list[str] = cls._MVAR_PATTERN.findall(raw_output)

        for idx, sec in enumerate(sections):
            goal = cls._parse_single_goal(sec, goal_idx=idx, prefix_mvar=prefix_mvar)
            # Find metavariables present in other goals for coupling
            coupled = [m for m in all_found_mvars if m != goal.mvar_id]
            goal.coupled_mvars = sorted(list(set(coupled)))
            parsed_goals.append(goal)

        return parsed_goals

    @classmethod
    def _parse_single_goal(cls, text: str, goal_idx: int, prefix_mvar: str) -> ProofGoal:
        lines = text.strip().splitlines()
        hypotheses: list[LocalHypothesis] = []
        target = ""
        in_target = False
        target_lines: list[str] = []

        for line in lines:
            line_str = line.strip()
            if line_str.startswith("case "):
                continue
            if line_str.startswith("⊢ "):
                in_target = True
                target_lines.append(line_str[2:].strip())
            elif in_target:
                target_lines.append(line_str)
            else:
                # Local hypothesis line: name : type or name : type := val
                if ":" in line_str:
                    parts = line_str.split(":", 1)
                    name = parts[0].strip()
                    rest = parts[1].strip()
                    val = None
                    is_let = False
                    if ":=" in rest:
                        is_let = True
                        type_part, val_part = rest.split(":=", 1)
                        type_str = type_part.strip()
                        val = val_part.strip()
                    else:
                        type_str = rest

                    hypotheses.append(
                        LocalHypothesis(
                            name=name,
                            type_str=type_str,
                            val_str=val,
                            is_let_binding=is_let,
                        )
                    )

        target = " ".join(target_lines).strip()
        if not target and lines:
            # Fallback for short unadorned goal target
            target = lines[-1].strip()

        mvars_in_target = cls._MVAR_PATTERN.findall(target)
        mvar_id = mvars_in_target[0] if mvars_in_target else f"{prefix_mvar}.{goal_idx}"

        return ProofGoal(
            goal_id=f"g_{goal_idx}_{mvar_id.replace('?', '').replace('.', '_')}",
            target=target,
            hypotheses=hypotheses,
            mvar_id=mvar_id,
        )

    @classmethod
    def make_proof_state(
        cls,
        raw_output: str,
        env_hash: str = "",
        depth: int = 0,
        parent_state_id: str | None = None
    ) -> ProofState:
        goals = cls.parse_goals(raw_output)
        is_solved = len(goals) == 0
        state_id = ProofState.compute_state_id(goals, {}, env_hash=env_hash)
        return ProofState(
            state_id=state_id,
            goals=goals,
            parent_state_id=parent_state_id,
            depth=depth,
            is_terminal=is_solved,
            is_solved=is_solved,
        )
