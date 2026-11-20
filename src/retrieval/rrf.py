"""Weighted Reciprocal Rank Fusion for ranked retrieval lists."""

from typing import Any


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    *,
    top_k: int = 5,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    if top_k < 1 or rrf_k < 0:
        raise ValueError("top_k must be positive and rrf_k must be non-negative")
    if vector_weight < 0 or keyword_weight < 0 or vector_weight + keyword_weight == 0:
        raise ValueError("At least one non-negative retrieval weight must be positive")

    fused: dict[str, dict[str, Any]] = {}
    for source, results, weight in (
        ("vector", vector_results, vector_weight),
        ("keyword", keyword_results, keyword_weight),
    ):
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            chunk_id = str(result["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            entry = fused.setdefault(chunk_id, {**result, "chunk_id": chunk_id, "score": 0.0})
            entry["score"] += weight / (rrf_k + rank)
            entry[f"{source}_rank"] = rank
            entry[f"{source}_score"] = result.get("score")
            if source == "keyword" and not entry.get("text"):
                entry["text"] = result.get("text", "")
                entry["metadata"] = result.get("metadata", {})
                entry["page_number"] = result.get("page_number")

    ordered = sorted(fused.values(), key=lambda result: (-result["score"], result["chunk_id"]))
    for rank, result in enumerate(ordered[:top_k], start=1):
        result["rank"] = rank
    return ordered[:top_k]
