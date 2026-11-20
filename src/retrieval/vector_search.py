"""Semantic similarity retrieval over the persistent Chroma collection."""

from typing import Any


class VectorSearch:
    def __init__(self, vector_store: Any) -> None:
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.vector_store.count() == 0:
            return []
        query_embedding = self.vector_store.embedding_service.embed_query(query)
        return self.vector_store.query_by_embedding(query_embedding, top_k=top_k)
