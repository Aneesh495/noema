"""Original lexical (BM25 / TF-IDF) premise retrieval baseline."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence
from noema.contracts.environment import PremiseDecl
from noema.contracts.states import ProofGoal


class LexicalRetriever:
    """Original BM25 ranker over eligible formal mathematical premises."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: dict[str, int] = {}
        self.doc_term_freqs: dict[str, Counter[str]] = {}
        self.inverted_index: dict[str, list[str]] = {}
        self.idf: dict[str, float] = {}
        self.premises: dict[str, PremiseDecl] = {}

    def _tokenize(self, text: str) -> list[str]:
        # Split on whitespace, punctuation, math symbols
        return [tok.lower() for tok in re.findall(r"[a-zA-Z0-9_]+|[∀∃→↔∧∨¬≤≥≠=+\-*/]", text) if tok]

    def index_premises(self, premises: dict[str, PremiseDecl]) -> None:
        """Build BM25 index over a collection of eligible premise declarations."""
        self.premises = premises
        self.corpus_size = len(premises)
        self.doc_lens = {}
        self.doc_term_freqs = {}
        self.inverted_index = {}
        self.idf = {}

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        total_tokens = 0
        df: Counter[str] = Counter()

        for name, decl in premises.items():
            doc_text = f"{decl.name} {decl.type_str} {decl.module} {decl.docstring or ''}"
            tokens = self._tokenize(doc_text)
            t_len = len(tokens)
            self.doc_lens[name] = t_len
            total_tokens += t_len
            
            tf = Counter(tokens)
            self.doc_term_freqs[name] = tf

            for term in tf:
                df[term] += 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(name)

        self.avg_doc_len = total_tokens / float(self.corpus_size)

        # Compute Robertson-Sparck Jones IDF
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def retrieve(
        self,
        query: str | ProofGoal,
        top_k: int = 10,
        eligible_filter: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve top_k premise names ranked by BM25 score against query."""
        if self.corpus_size == 0:
            return []

        if isinstance(query, ProofGoal):
            q_text = query.context_target_str()
        else:
            q_text = str(query)

        query_tokens = self._tokenize(q_text)
        if not query_tokens:
            return []

        scores: Counter[str] = Counter()
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            postings = self.inverted_index.get(term, [])
            for doc_name in postings:
                if eligible_filter is not None and doc_name not in eligible_filter:
                    continue
                tf = self.doc_term_freqs[doc_name][term]
                doc_len = self.doc_lens[doc_name]
                
                # BM25 term weighting
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score = idf_val * (numerator / denominator)
                scores[doc_name] += score

        ranked = scores.most_common(top_k)
        return ranked
