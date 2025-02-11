"""Original math-aware tokenizer trained strictly on training corpus material."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence


class MathTokenizer:
    """Original tokenizer tailored for formal Lean mathematical syntax."""

    SPECIAL_TOKENS = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<goal>",
        "<hyp>",
        "<target>",
        "<premise>",
        "<action>",
        "<sep>",
    ]

    # Math and Lean syntactic tokens that must never be split arbitrarily
    MATH_OPERATORS = [
        "∀", "∃", "→", "↔", "∧", "∨", "¬", "≤", "≥", "≠", "⊢",
        ":=", "::", "++", "->", "<->", "=>", "<=", ">=", "!=",
        "(", ")", "[", "]", "{", "}", ":", ";", ",", ".", "=", "+", "*", "-", "/", "%"
    ]

    def __init__(self, vocab: dict[str, int] | None = None):
        if vocab is not None:
            self.vocab = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        else:
            self.vocab = {}
            self.id_to_token = {}
            self._init_special_vocab()

        self.pad_token_id = self.vocab.get("<pad>", 0)
        self.bos_token_id = self.vocab.get("<bos>", 1)
        self.eos_token_id = self.vocab.get("<eos>", 2)
        self.unk_token_id = self.vocab.get("<unk>", 3)

    def _init_special_vocab(self) -> None:
        idx = 0
        for tok in self.SPECIAL_TOKENS:
            self.vocab[tok] = idx
            self.id_to_token[idx] = tok
            idx += 1

        for op in self.MATH_OPERATORS:
            if op not in self.vocab:
                self.vocab[op] = idx
                self.id_to_token[idx] = op
                idx += 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize_text(self, text: str) -> list[str]:
        """Lexical tokenization preserving mathematical operators and identifiers."""
        # Regex capturing whitespace, special ops, numbers, and identifiers
        # Sort ops by length descending to match multi-char ops like ':=' or '++' first
        sorted_ops = sorted(self.MATH_OPERATORS, key=len, reverse=True)
        escaped_ops = [re.escape(op) for op in sorted_ops]
        op_pattern = "|".join(escaped_ops)

        # Token pattern: special ops, words/identifiers, digits
        pattern = re.compile(rf"{op_pattern}|[a-zA-Z_][a-zA-Z0-9_]*|\d+|\S")
        tokens = pattern.findall(text)
        return tokens

    def train_from_corpus(self, texts: Iterable[str], max_vocab_size: int = 8192) -> None:
        """Fit vocabulary exclusively on training texts."""
        self._init_special_vocab()
        freq: dict[str, int] = {}

        for txt in texts:
            toks = self.tokenize_text(txt)
            for t in toks:
                freq[t] = freq.get(t, 0) + 1

        # Sort by frequency
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        idx = len(self.vocab)
        for tok, _ in sorted_tokens:
            if tok not in self.vocab:
                self.vocab[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1
                if idx >= max_vocab_size:
                    break

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        """Encode text to token IDs with optional BOS/EOS and truncation."""
        raw_tokens = self.tokenize_text(text)
        token_ids = []
        if add_bos:
            token_ids.append(self.bos_token_id)

        for tok in raw_tokens:
            token_ids.append(self.vocab.get(tok, self.unk_token_id))

        if add_eos:
            token_ids.append(self.eos_token_id)

        if max_length is not None and len(token_ids) > max_length:
            if add_eos:
                token_ids = token_ids[:max_length - 1] + [self.eos_token_id]
            else:
                token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a string."""
        tokens = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "<unk>")
            if skip_special_tokens and tok in self.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path | str) -> MathTokenizer:
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
