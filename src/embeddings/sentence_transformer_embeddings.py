"""Local Sentence Transformers embeddings with the project embedding interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32


class LocalEmbeddingError(RuntimeError):
    """A local model could not load or returned invalid vectors."""


class SentenceTransformerEmbeddingService:
    """Generate normalized document and query vectors using one local model."""

    provider = "local"
    dimension = EXPECTED_DIMENSION

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: Any | None = None,
    ) -> None:
        load_dotenv(Path(".env"))
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise LocalEmbeddingError(
                    "Sentence Transformers is not installed. Run "
                    "`python -m pip install -r requirements.txt`."
                ) from exc
            try:
                model = SentenceTransformer(self.model_name)
                dimension_reader = getattr(model, "get_embedding_dimension", None)
                if dimension_reader is None:
                    dimension_reader = model.get_sentence_embedding_dimension
                dimension = dimension_reader()
            except Exception as exc:
                raise LocalEmbeddingError(
                    f"Could not load local embedding model {self.model_name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if dimension != EXPECTED_DIMENSION:
                raise LocalEmbeddingError(
                    f"Model {self.model_name!r} has dimension {dimension}; "
                    f"expected {EXPECTED_DIMENSION}."
                )
            self._model = model
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("All embedding inputs must be strings")
        model = self._get_model()
        try:
            encoded = model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = [[float(value) for value in vector] for vector in encoded]
        except Exception as exc:
            raise LocalEmbeddingError(
                f"Local embedding failed for {len(texts)} text(s): {type(exc).__name__}: {exc}"
            ) from exc
        if len(vectors) != len(texts):
            raise LocalEmbeddingError(
                f"Model returned {len(vectors)} vectors for {len(texts)} input texts."
            )
        if any(len(vector) != EXPECTED_DIMENSION for vector in vectors):
            dimensions = sorted({len(vector) for vector in vectors})
            raise LocalEmbeddingError(
                f"Model returned vector dimension(s) {dimensions}; expected {EXPECTED_DIMENSION}."
            )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts in bounded batches."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._encode(texts[start:start + self.batch_size]))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed one query with the same model and normalization as documents."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        vectors = self._encode([query])
        return vectors[0]

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return embedded chunk records while preserving IDs and metadata."""
        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Input chunk IDs must be unique")
        vectors = self.embed_documents([str(chunk["text"]) for chunk in chunks])
        return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]
