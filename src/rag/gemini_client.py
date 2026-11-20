"""Shared Gemini chat client helpers for answer generation and evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_FIXED_SAMPLING_MODELS = {"gemini-3.5-flash-lite"}


def _uses_fixed_sampling(model: str) -> bool:
    normalized = model.casefold().removeprefix("models/")
    return normalized in _FIXED_SAMPLING_MODELS or any(
        normalized.startswith(f"{fixed_model}-") for fixed_model in _FIXED_SAMPLING_MODELS
    )


def ensure_api_key() -> None:
    """Load `.env` and verify that a Gemini API key is available."""
    load_dotenv(Path(".env"))
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError(
            "Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env before using Gemini."
        )


def create_chat_client(
    model: str,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    request_timeout: float = 60.0,
) -> Any:
    """Create a Gemini chat client without hidden retries.

    Retries are handled by the shared project retry helper so rate limits and
    long quota reset delays are handled consistently.
    """
    ensure_api_key()
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "Gemini chat integration is missing. Install requirements with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    model_options = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "retries": 0,
        "request_timeout": request_timeout,
    }
    # This Gemini model ignores custom sampling settings. Omit temperature so
    # LangChain does not emit a warning for a parameter it will discard.
    if not _uses_fixed_sampling(model):
        model_options["temperature"] = temperature
    return ChatGoogleGenerativeAI(
        **model_options,
    )


def message_text(message: Any) -> str:
    """Extract plain text from a LangChain or Gemini chat response."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")
