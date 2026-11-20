"""Deterministic retrieval evaluation metrics."""

from __future__ import annotations

from typing import Iterable


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def precision_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float | None:
    """Fraction of the top-k retrieved IDs that are relevant."""
    relevant = set(str(item) for item in relevant_ids)
    if not relevant:
        return None
    top = _unique(retrieved_ids)[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)


def recall_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float | None:
    """Fraction of all relevant IDs found in the top-k retrieved IDs."""
    relevant = set(str(item) for item in relevant_ids)
    if not relevant:
        return None
    top = _unique(retrieved_ids)[:k]
    return sum(1 for item in top if item in relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int | None = None) -> float | None:
    """Reciprocal rank of the first relevant result."""
    relevant = set(str(item) for item in relevant_ids)
    if not relevant:
        return None
    ranked = _unique(retrieved_ids)
    if k is not None:
        ranked = ranked[:k]
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float | None:
    """Whether at least one relevant ID appears in the top-k results."""
    relevant = set(str(item) for item in relevant_ids)
    if not relevant:
        return None
    top = _unique(retrieved_ids)[:k]
    return 1.0 if any(item in relevant for item in top) else 0.0


def evaluate_retrieval(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> dict[str, float | None]:
    """Return all deterministic retrieval metrics for one question."""
    if k < 1:
        raise ValueError("k must be positive")
    return {
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids, k),
        "hit_rate_at_k": hit_rate_at_k(retrieved_ids, relevant_ids, k),
    }
