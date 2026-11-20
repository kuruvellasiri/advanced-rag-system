"""Resolve the configured document/query embedding provider."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from .gemini_embeddings import MODEL_NAME as GEMINI_MODEL_NAME, GeminiEmbeddingService
from .sentence_transformer_embeddings import (
    DEFAULT_MODEL_NAME as LOCAL_MODEL_NAME,
    SentenceTransformerEmbeddingService,
)

DEFAULT_PROVIDER = "local"


def embedding_configuration(
    provider: str | None = None,
    model_name: str | None = None,
) -> tuple[str, str]:
    """Return validated provider/model values from arguments or the environment."""
    load_dotenv(Path(".env"))
    selected_provider = (provider or os.getenv("EMBEDDING_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if selected_provider not in {"local", "gemini"}:
        raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'gemini'")
    default_model = LOCAL_MODEL_NAME if selected_provider == "local" else GEMINI_MODEL_NAME
    selected_model = (model_name or os.getenv("EMBEDDING_MODEL") or default_model).strip()
    if not selected_model:
        raise ValueError("EMBEDDING_MODEL must not be empty")
    return selected_provider, selected_model


def create_embedding_service(
    provider: str | None = None,
    model_name: str | None = None,
    *,
    retry_callback: Callable[[int | None, int, float], None] | None = None,
    local_model: Any | None = None,
) -> Any:
    """Construct the selected provider without coupling it to answer generation."""
    selected_provider, selected_model = embedding_configuration(provider, model_name)
    if selected_provider == "local":
        raw_batch_size = os.getenv("EMBEDDING_BATCH_SIZE", "32").strip()
        try:
            batch_size = int(raw_batch_size)
        except ValueError as exc:
            raise ValueError("EMBEDDING_BATCH_SIZE must be an integer") from exc
        return SentenceTransformerEmbeddingService(
            selected_model,
            batch_size=batch_size,
            model=local_model,
        )
    return GeminiEmbeddingService(selected_model, retry_callback=retry_callback)
