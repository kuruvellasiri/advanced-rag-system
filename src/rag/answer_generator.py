"""Grounded Gemini answer generation over the existing hybrid retriever."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from src.embeddings.gemini_embeddings import call_with_retries

from .gemini_client import create_chat_client, message_text
from .prompts import NO_ANSWER_TEXT, build_answer_prompt, chunk_page_number

DEFAULT_TOP_K = 5
DEFAULT_CHAT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_SERVER_RETRY_SECONDS = 300.0
_CONTEXT_CITATION = re.compile(r"\[([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\]")


class AnswerGenerationError(RuntimeError):
    """Gemini answer generation failed."""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def sources_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return unique source records preserving chunk IDs and page numbers."""
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = str(chunk.get("chunk_id") or "unknown")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        sources.append({
            "chunk_id": chunk_id,
            "page_number": chunk_page_number(chunk),
            "document_name": Path(str((chunk.get("metadata") or {}).get("source") or "AWS document")).name,
            "rank": chunk.get("rank", index),
            "score": chunk.get("score"),
        })
    return sources


def format_sources_block(sources: list[dict[str, Any]]) -> str:
    """Format deterministic page and chunk attribution for display."""
    if not sources:
        return ""
    lines = ["**Source References:**"]
    for source in sources:
        lines.extend([
            f"- Document: {source['document_name']}",
            f"  Page: {source['page_number']}",
            f"  Chunk ID: {source['chunk_id']}",
        ])
    return "\n".join(lines)


def _append_sources(answer: str, sources: list[dict[str, Any]]) -> str:
    block = format_sources_block(sources)
    if not block:
        return answer
    return f"{answer.rstrip()}\n\n{block}"


def _validated_citation_positions(answer: str, context_count: int) -> list[int] | None:
    """Return unique 1-based context positions, or None for invalid/missing cites."""
    matches = list(_CONTEXT_CITATION.finditer(answer))
    if not matches:
        return None
    residue = _CONTEXT_CITATION.sub("", answer)
    if "[" in residue or "]" in residue:
        return None

    positions: list[int] = []
    for match in matches:
        for value in match.group(1).split(","):
            position = int(value.strip())
            if position < 1 or position > context_count:
                return None
            if position not in positions:
                positions.append(position)
    return positions


def _clean_display_answer(answer: str) -> str:
    """Remove internal inline citations and ensure a clear answer heading."""
    answer = re.split(r"(?im)^\s*\*\*Source References:\*\*\s*$", answer, maxsplit=1)[0]
    answer = _CONTEXT_CITATION.sub("", answer)
    answer = re.sub(r"\s+([,.;:!?])", r"\1", answer)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in answer.splitlines()]
    cleaned_lines: list[str] = []
    for line in lines:
        if line or (cleaned_lines and cleaned_lines[-1]):
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    if cleaned.casefold().startswith("answer:"):
        cleaned = "**Answer:**" + cleaned[len("Answer:"):]
    elif not cleaned.casefold().startswith("**answer:**"):
        cleaned = f"**Answer:**\n{cleaned}" if cleaned else "**Answer:**"
    return cleaned


class GeminiAnswerGenerator:
    """Retrieve with HybridSearch and generate a context-grounded Gemini answer."""

    def __init__(
        self,
        hybrid_search: Any,
        *,
        model: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        max_retries: int | None = None,
        max_server_delay: float | None = None,
        retry_callback: Callable[[int | None, int, float], None] | None = None,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        load_dotenv(Path(".env"))
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.hybrid_search = hybrid_search
        self.model = model or os.getenv("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.top_k = int(top_k)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = (
            max_retries if max_retries is not None
            else _env_int("GEMINI_MAX_RETRIES", 2)
        )
        self.max_server_delay = (
            max_server_delay if max_server_delay is not None
            else _env_float("GEMINI_MAX_SERVER_RETRY_SECONDS", DEFAULT_MAX_SERVER_RETRY_SECONDS)
        )
        self.retry_callback = retry_callback
        self._client = client
        self.client_factory = client_factory or self._default_client_factory

    def _default_client_factory(self) -> Any:
        return create_chat_client(
            self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                self._client = self.client_factory()
            except Exception as exc:
                raise AnswerGenerationError(
                    f"Unable to initialize Gemini chat client: {type(exc).__name__}: {exc}"
                ) from exc
        return self._client

    def generate(
        self,
        question: str,
        *,
        top_k: int | None = None,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> dict[str, Any]:
        """Retrieve context and produce a grounded answer with source attribution."""
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        k = self.top_k if top_k is None else int(top_k)
        if k < 1:
            raise ValueError("top_k must be positive")

        try:
            retrieval = self.hybrid_search.search(
                question,
                top_k=k,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )
        except Exception as exc:
            raise AnswerGenerationError(
                f"Question retrieval failed: {type(exc).__name__}: {exc}"
            ) from exc
        chunks = list(retrieval.get("hybrid", []))[:k]

        if not chunks:
            return {
                "question": question,
                "answer": NO_ANSWER_TEXT,
                "model_answer": NO_ANSWER_TEXT,
                "sources": [],
                "retrieved": [],
                "retrieval": retrieval,
                "top_k": k,
                "model": self.model,
            }

        client = self._ensure_client()
        messages = build_answer_prompt(question, chunks)
        try:
            response = call_with_retries(
                lambda: client.invoke(messages),
                max_retries=self.max_retries,
                max_server_delay=self.max_server_delay,
                on_retry=self.retry_callback,
            )
        except Exception as exc:
            raise AnswerGenerationError(
                f"Gemini answer generation failed: {type(exc).__name__}: {exc}"
            ) from exc

        model_answer = message_text(response).strip()
        if not model_answer:
            raise AnswerGenerationError("Gemini returned an empty answer.")

        if NO_ANSWER_TEXT.casefold() in model_answer.casefold():
            final_answer = NO_ANSWER_TEXT
            sources = []
        else:
            # Context block numbers map directly to the exact chunks sent to the model.
            citation_positions = _validated_citation_positions(model_answer, len(chunks))
            if not citation_positions:
                model_answer = NO_ANSWER_TEXT
                final_answer = NO_ANSWER_TEXT
                sources = []
            else:
                cited_chunks = [chunks[position - 1] for position in citation_positions]
                sources = sources_from_chunks(cited_chunks)
                final_answer = _append_sources(_clean_display_answer(model_answer), sources)

        return {
            "question": question,
            "answer": final_answer,
            "model_answer": model_answer,
            "sources": sources,
            "retrieved": chunks,
            "retrieval": retrieval,
            "top_k": k,
            "model": self.model,
        }
