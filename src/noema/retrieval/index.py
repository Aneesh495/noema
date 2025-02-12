"""Premise index manager and hybrid retrieval pipeline."""

from __future__ import annotations

import torch
from noema.contracts.environment import EnvironmentSpec, PremiseDecl
from noema.contracts.states import ProofGoal
from noema.retrieval.dense import DenseDualEncoder
from noema.retrieval.lexical import LexicalRetriever
from noema.tokenizer.batching import BatchCollator


class PremiseIndex:
    """Manages premise representations, hybrid scoring, and recall evaluation."""

    def __init__(
        self,
        env_spec: EnvironmentSpec,
        dense_model: DenseDualEncoder | None = None,
        collator: BatchCollator | None = None,
        alpha_dense: float = 0.5,
    ):
        self.env_spec = env_spec
        self.dense_model = dense_model
        self.collator = collator
        self.alpha_dense = alpha_dense
        self.lexical = LexicalRetriever()
        self.lexical.index_premises(env_spec.eligible_premises)
        
        # Precomputed premise vectors if dense model is available
        self.premise_names: list[str] = list(env_spec.eligible_premises.keys())
        self.premise_vectors: torch.Tensor | None = None

        if dense_model is not None and collator is not None and self.premise_names:
            self._precompute_dense_vectors()

    def _precompute_dense_vectors(self) -> None:
        if self.dense_model is None or self.collator is None:
            return
        self.dense_model.eval()
        texts = [
            f"{p.name} : {p.type_str}"
            for p in (self.env_spec.eligible_premises[n] for n in self.premise_names)
        ]
        with torch.no_grad():
            batch = self.collator.collate_texts(texts)
            input_ids = batch["input_ids"]
            mask = batch["attention_mask"]
            self.premise_vectors = self.dense_model.encode_premise(input_ids, mask)

    def retrieve(
        self,
        goal: ProofGoal,
        top_k: int = 10,
        target_name: str | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve top_k premise names ranked by hybrid (lexical + dense) score."""
        eligible_set = {
            k for k in self.env_spec.eligible_premises
            if target_name is None or k != target_name
        }

        # 1. Lexical BM25 ranking
        lex_results = self.lexical.retrieve(goal, top_k=top_k * 3, eligible_filter=eligible_set)
        lex_dict = {name: score for name, score in lex_results}
        # Normalize lexical scores
        max_lex = max(lex_dict.values()) if lex_dict else 1.0
        norm_lex = {k: v / max(max_lex, 1e-6) for k, v in lex_dict.items()}

        # 2. Dense ranking if available
        dense_dict: dict[str, float] = {}
        if self.dense_model is not None and self.collator is not None and self.premise_vectors is not None:
            self.dense_model.eval()
            with torch.no_grad():
                g_batch = self.collator.collate_texts([goal.context_target_str()])
                g_vec = self.dense_model.encode_goal(g_batch["input_ids"], g_batch["attention_mask"])
                # [1, D] @ [P, D].T -> [P]
                sim = torch.matmul(g_vec, self.premise_vectors.t()).squeeze(0)
                sim_scores = sim.tolist()
                for name, s in zip(self.premise_names, sim_scores):
                    if name in eligible_set:
                        dense_dict[name] = s

        # 3. Combine scores
        all_candidates = set(norm_lex.keys()) | set(dense_dict.keys())
        if not all_candidates:
            all_candidates = set(list(eligible_set)[:top_k])

        final_scores: list[tuple[str, float]] = []
        for cand in all_candidates:
            s_lex = norm_lex.get(cand, 0.0)
            s_dense = dense_dict.get(cand, 0.0)
            combined = (1.0 - self.alpha_dense) * s_lex + self.alpha_dense * s_dense
            final_scores.append((cand, combined))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]

    @classmethod
    def evaluate_recall(
        cls,
        queries: list[tuple[ProofGoal, list[str]]],
        retriever_func,
        k_values: tuple[int, ...] = (1, 5, 10),
    ) -> dict[str, float]:
        """Compute recall@k on a list of (goal, ground_truth_premises) pairs."""
        recalls = {f"recall@{k}": 0.0 for k in k_values}
        if not queries:
            return recalls

        for goal, gt_premises in queries:
            if not gt_premises:
                continue
            ranked = retriever_func(goal, top_k=max(k_values))
            ranked_names = [r[0] for r in ranked]

            for k in k_values:
                top_k_names = set(ranked_names[:k])
                # Check if at least one ground-truth premise was retrieved
                hit = any(p in top_k_names for p in gt_premises)
                if hit:
                    recalls[f"recall@{k}"] += 1.0

        num_q = float(len(queries))
        return {k: v / num_q for k, v in recalls.items()}
