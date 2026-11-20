"""Hybrid retrieval combining Chroma vector search, BM25, and weighted RRF."""

from pathlib import Path
from typing import Any

from .bm25_search import BM25Search, DEFAULT_CONTENT_PATH
from .rrf import reciprocal_rank_fusion
from .vector_search import VectorSearch


class HybridSearch:
    def __init__(self, vector_store: Any, keyword_search: BM25Search | None = None) -> None:
        self.vector_search = vector_store if isinstance(vector_store, VectorSearch) else VectorSearch(vector_store)
        self.keyword_search = keyword_search or BM25Search.from_json(DEFAULT_CONTENT_PATH)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> dict[str, list[dict[str, Any]]]:
        vector_results = self.vector_search.search(query, top_k=top_k)
        keyword_results = self.keyword_search.search(query, top_k=top_k)
        fused_results = reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            top_k=top_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )
        return {
            "vector": vector_results,
            "keyword": keyword_results,
            "hybrid": fused_results,
            "vector_only": vector_results[:top_k],
        }
