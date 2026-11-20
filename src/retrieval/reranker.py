"""Optional cross-encoder reranking for hybrid retrieval candidates."""

from __future__ import annotations

import math
from typing import Any, Callable


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerError(RuntimeError):
    """A cross-encoder could not be loaded or failed to score candidates."""


class CrossEncoderReranker:
    """Load one Sentence Transformers CrossEncoder and reuse it for queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        batch_size: int = 16,
        model: Any | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("RERANKER_MODEL must not be empty")
        if batch_size < 1:
            raise ValueError("reranker batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model
        self._model_factory = model_factory

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            if self._model_factory is not None:
                self._model = self._model_factory(self.model_name)
            else:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device="cpu")
        except Exception as exc:
            raise RerankerError(
                f"Unable to load reranker model {self.model_name!r}: {type(exc).__name__}: {exc}"
            ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []
        try:
            model = self._load_model()
            pairs = [(query, str(chunk.get("text") or "")) for chunk in candidates]
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores = [float(score) for score in raw_scores]
            if len(scores) != len(candidates) or not all(math.isfinite(score) for score in scores):
                raise ValueError("cross-encoder returned invalid or incomplete scores")
        except RerankerError:
            raise
        except Exception as exc:
            raise RerankerError(
                f"Unable to score candidates with {self.model_name!r}: {type(exc).__name__}: {exc}"
            ) from exc

        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: (
                -pair[1],
                int(pair[0].get("candidate_rank", pair[0].get("rank", 0))),
                str(pair[0].get("chunk_id", "")),
            ),
        )[:top_k]
        return [{**chunk, "reranker_score": score, "rank": rank}
                for rank, (chunk, score) in enumerate(ranked, start=1)]


class RerankedHybridSearch:
    """Retrieve an expanded hybrid candidate pool, then return reranked top-K."""

    def __init__(
        self,
        hybrid_search: Any,
        reranker: CrossEncoderReranker,
        *,
        candidate_count: int = 10,
        rerank_top_k: int = 5,
    ) -> None:
        if candidate_count < 1:
            raise ValueError("RERANKER_CANDIDATE_COUNT must be positive")
        if rerank_top_k < 1:
            raise ValueError("RERANKER_TOP_K must be positive")
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.rerank_top_k = rerank_top_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> dict[str, list[dict[str, Any]]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        candidate_k = max(self.candidate_count, top_k)
        retrieval = self.hybrid_search.search(
            query,
            top_k=candidate_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )
        candidates = [
            {**chunk, "candidate_rank": chunk.get("rank", index)}
            for index, chunk in enumerate(retrieval.get("hybrid", []), start=1)
        ]
        ranked = self.reranker.rerank(
            query, candidates, top_k=min(top_k, self.rerank_top_k)
        )
        return {
            **retrieval,
            "hybrid": ranked,
            "vector": list(retrieval.get("vector", []))[:top_k],
            "keyword": list(retrieval.get("keyword", []))[:top_k],
            "vector_only": list(retrieval.get("vector_only", retrieval.get("vector", [])))[:top_k],
        }
