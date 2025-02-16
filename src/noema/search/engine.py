"""Best-first AND/OR proof-hypergraph search engine."""

from __future__ import annotations

import heapq
import time
from typing import Sequence
import torch
from noema.contracts.actions import ActionGrammar, TacticAction, TacticKind
from noema.contracts.certificates import ProofCertificate, VerificationStatus
from noema.contracts.environment import EnvironmentSpec
from noema.contracts.states import ProofGoal, ProofState
from noema.contracts.theorems import TheoremSpec
from noema.formal.bridge import FormalBridge
from noema.formal.checker import IndependentChecker
from noema.formal.types import TransitionStatus
from noema.models.transformer import NoemaTransformer
from noema.retrieval.index import PremiseIndex
from noema.search.budget import SearchAccounting, SearchBudget, TerminationReason
from noema.search.hypergraph_engine import HypergraphEngine
from noema.search.priority import PriorityScorer, SearchStrategy
from noema.tokenizer.batching import BatchCollator


class SearchResult:
    """Full outcome of a theorem proving search attempt."""

    def __init__(
        self,
        theorem: TheoremSpec,
        is_proved: bool,
        proof_script: list[str],
        accounting: SearchAccounting,
        hypergraph_engine: HypergraphEngine,
        certificate: ProofCertificate | None = None,
    ):
        self.theorem = theorem
        self.is_proved = is_proved
        self.proof_script = proof_script
        self.accounting = accounting
        self.hypergraph_engine = hypergraph_engine
        self.certificate = certificate


class ProofSearchEngine:
    """Coordinates neural action scoring, premise retrieval, and budgeted Lean 4 search."""

    def __init__(
        self,
        bridge: FormalBridge,
        checker: IndependentChecker,
        model: NoemaTransformer | None = None,
        retriever_index: PremiseIndex | None = None,
        collator: BatchCollator | None = None,
        strategy: SearchStrategy = SearchStrategy.NEURAL_GUIDED,
        device: torch.device | str = "cpu",
    ):
        self.bridge = bridge
        self.checker = checker
        self.model = model
        self.retriever_index = retriever_index
        self.collator = collator
        self.strategy = strategy
        self.scorer = PriorityScorer(strategy=strategy)
        self.device = torch.device(device)
        if self.model:
            self.model.to(self.device)
            self.model.eval()

    def search(
        self,
        theorem: TheoremSpec,
        budget: SearchBudget | None = None,
    ) -> SearchResult:
        """Execute budgeted search to discover a verified proof."""
        b = budget or SearchBudget()
        accounting = SearchAccounting()
        start_time = time.perf_counter()

        # 1. Obtain initial proof state from Lean bridge
        init_state, _ = self.bridge.get_initial_state(theorem)
        hg_engine = HypergraphEngine(root_state=init_state)

        if not init_state.has_open_goals():
            # Trivial or already proved theorem
            accounting.is_proved = True
            accounting.termination_reason = TerminationReason.PROVED
            accounting.total_time_ms = (time.perf_counter() - start_time) * 1000.0
            cert, _ = self.checker.verify_proof(theorem, tactic_script="rfl")
            return SearchResult(
                theorem=theorem,
                is_proved=True,
                proof_script=["rfl"],
                accounting=accounting,
                hypergraph_engine=hg_engine,
                certificate=cert,
            )

        # Priority queue entries: (priority_score, counter, state, history)
        counter = 0
        pqueue: list[tuple[float, int, ProofState, list[str]]] = []
        
        # Initial score
        init_score = self.scorer.compute_score(init_state)
        heapq.heappush(pqueue, (init_score, counter, init_state, []))
        visited_states: set[str] = {init_state.state_id}

        while pqueue:
            can_run, term_reason = accounting.can_continue(b, start_time)
            if not can_run:
                accounting.termination_reason = term_reason or TerminationReason.TIMEOUT
                break

            score, _, current_state, history = heapq.heappop(pqueue)

            # Generate candidate tactic proposals
            proposals = self._propose_tactics(theorem, current_state)
            accounting.generated_proposals += len(proposals)

            for action in proposals:
                can_run, term_reason = accounting.can_continue(b, start_time)
                if not can_run:
                    accounting.termination_reason = term_reason or TerminationReason.TRANSITIONS_EXHAUSTED
                    break

                # Attempt formal transition in Lean
                accounting.attempted_transitions += 1
                t0 = time.perf_counter()
                res = self.bridge.step(
                    theorem=theorem,
                    current_state=current_state,
                    action=action,
                    tactic_history=history,
                )
                dt = (time.perf_counter() - t0) * 1000.0
                accounting.kernel_time_ms += dt

                if not res.is_valid_transition or res.after_state is None:
                    accounting.rejected_actions += 1
                    continue

                accounting.valid_transitions += 1
                next_state = res.after_state

                # Check for duplicate state
                if next_state.state_id in visited_states:
                    accounting.duplicate_states += 1
                    continue
                visited_states.add(next_state.state_id)

                # Register in hypergraph
                hg_engine.register_transition(
                    parent_state_id=current_state.state_id,
                    child_state=next_state,
                    action=action,
                    execution_time_ms=dt,
                )

                new_history = history + [action.raw_tactic]

                # Check if proof is complete
                if res.is_proved() or hg_engine.is_root_proved():
                    accounting.is_proved = True
                    accounting.subgoal_closures += 1
                    accounting.termination_reason = TerminationReason.PROVED
                    accounting.total_time_ms = (time.perf_counter() - start_time) * 1000.0
                    
                    full_script = hg_engine.extract_proof_script() or new_history
                    # Check proof in fresh independent process!
                    cert, audit = self.checker.verify_proof(
                        theorem,
                        tactic_script="\n".join(full_script),
                    )

                    return SearchResult(
                        theorem=theorem,
                        is_proved=(cert.status == VerificationStatus.VERIFIED),
                        proof_script=full_script,
                        accounting=accounting,
                        hypergraph_engine=hg_engine,
                        certificate=cert,
                    )

                # Neural / heuristic scoring for child state
                c_score = self._score_child_state(next_state, action)
                counter += 1
                heapq.heappush(pqueue, (c_score, counter, next_state, new_history))

        accounting.total_time_ms = (time.perf_counter() - start_time) * 1000.0
        if not accounting.is_proved and accounting.termination_reason == TerminationReason.PROVED:
            accounting.termination_reason = TerminationReason.QUEUE_EMPTY

        return SearchResult(
            theorem=theorem,
            is_proved=False,
            proof_script=[],
            accounting=accounting,
            hypergraph_engine=hg_engine,
            certificate=None,
        )

    def _propose_tactics(self, theorem: TheoremSpec, state: ProofState) -> list[TacticAction]:
        """Propose candidate tactics for the current state."""
        open_goals = state.get_open_goals()
        if not open_goals:
            return []
        goal = open_goals[0]

        candidates = []
        
        # 1. Structural reflexivity
        candidates.append(TacticAction.from_string("rfl"))

        # 2. Variable intro candidates
        if "∀" in goal.target or "→" in goal.target:
            candidates.append(TacticAction.from_string("intro"))
            candidates.append(TacticAction.from_string("intros"))

        # 3. Local hypothesis assumptions
        if goal.hypotheses:
            candidates.append(TacticAction.from_string("assumption"))
            for h in goal.hypotheses:
                candidates.append(TacticAction.from_string(f"exact {h.name}"))
                candidates.append(TacticAction.from_string(f"apply {h.name}"))
                candidates.append(TacticAction.from_string(f"cases {h.name}"))

        # 4. Premise retrieval candidates (using retriever if available)
        if self.retriever_index:
            retrieved = self.retriever_index.retrieve(goal, top_k=5, target_name=theorem.name)
            for prem_name, _ in retrieved:
                candidates.append(TacticAction.from_string(f"apply {prem_name}"))
                candidates.append(TacticAction.from_string(f"exact {prem_name}"))
                candidates.append(TacticAction.from_string(f"rw [{prem_name}]"))
                candidates.append(TacticAction.from_string(f"rw [← {prem_name}]"))

        # 5. Common logic constructors
        if "∧" in goal.target:
            candidates.append(TacticAction.from_string("constructor"))
        if "∨" in goal.target:
            candidates.append(TacticAction.from_string("apply Or.inl"))
            candidates.append(TacticAction.from_string("apply Or.inr"))

        return candidates

    def _score_child_state(self, state: ProofState, action: TacticAction) -> float:
        """Compute search queue priority score using model or heuristics."""
        value_est = 0.5
        policy_prob = action.confidence

        if self.model and self.collator and state.get_open_goals():
            goal = state.get_open_goals()[0]
            with torch.no_grad():
                batch = self.collator.collate_goals([goal])
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                out = self.model(input_ids, attention_mask=mask)
                value_est = float(out["value"].item())

        return self.scorer.compute_score(
            state=state,
            action=action,
            policy_prob=policy_prob,
            value_estimate=value_est,
        )
