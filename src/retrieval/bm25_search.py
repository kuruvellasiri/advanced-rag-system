"""BM25 keyword search over the AWS content chunk JSON dataset."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"(?u)\b[\w]+(?:[.+#-][\w]+)*\b")
DEFAULT_CONTENT_PATH = Path("data/chunks/content_chunks.json")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


class BM25Search:
    """Small in-process Okapi BM25 index with no external service dependency."""

    def __init__(self, chunks: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        unique: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            unique.setdefault(str(chunk["chunk_id"]), chunk)
        self.chunks = list(unique.values())
        self.tokens = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for frequency in self.term_frequencies:
            document_frequency.update(frequency.keys())
        count = len(self.chunks)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CONTENT_PATH, **kwargs: Any) -> "BM25Search":
        chunks = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chunks, **kwargs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return []

        scored = []
        for index, chunk in enumerate(self.chunks):
            score = 0.0
            length = self.lengths[index]
            for term in query_terms:
                frequency = self.term_frequencies[index].get(term, 0)
                if not frequency:
                    continue
                norm = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                ) if self.average_length else frequency + self.k1
                score += self.idf.get(term, 0.0) * (frequency * (self.k1 + 1) / norm)
            if score > 0:
                scored.append((score, index, chunk))

        scored.sort(key=lambda item: (-item[0], item[2]["chunk_id"]))
        return [
            {**chunk, "score": score, "rank": rank}
            for rank, (score, _, chunk) in enumerate(scored[:top_k], start=1)
        ]
