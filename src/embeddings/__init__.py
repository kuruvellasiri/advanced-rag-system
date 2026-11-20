"""Embedding provider implementations and factory."""

from .gemini_embeddings import GeminiEmbeddingError, GeminiEmbeddingService, MODEL_NAME
from .factory import create_embedding_service, embedding_configuration
from .sentence_transformer_embeddings import (
    DEFAULT_MODEL_NAME as LOCAL_MODEL_NAME,
    EXPECTED_DIMENSION,
    LocalEmbeddingError,
    SentenceTransformerEmbeddingService,
)

__all__ = [
    "GeminiEmbeddingError", "GeminiEmbeddingService", "MODEL_NAME",
    "SentenceTransformerEmbeddingService", "LocalEmbeddingError",
    "LOCAL_MODEL_NAME", "EXPECTED_DIMENSION", "create_embedding_service",
    "embedding_configuration",
]
