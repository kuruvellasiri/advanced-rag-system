"""Optional Gemini-based answer evaluation (automated estimates)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from src.embeddings.gemini_embeddings import call_with_retries
from src.rag.answer_generator import DEFAULT_CHAT_MODEL, DEFAULT_MAX_SERVER_RETRY_SECONDS
from src.rag.gemini_client import create_chat_client, message_text
from src.rag.prompts import format_context

from .prompts import build_evaluation_prompt

SCORE_KEYS = ("answer_relevance", "faithfulness", "context_relevance", "correctness")


class AnswerEvaluationError(RuntimeError):
    """Gemini answer evaluation failed or returned invalid output."""


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AnswerEvaluationError("Gemini evaluator did not return a JSON object")
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise AnswerEvaluationError(f"Gemini evaluator returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AnswerEvaluationError("Gemini evaluator JSON must be an object")
    return data


def parse_evaluation_scores(text: str) -> dict[str, Any]:
    """Parse and validate evaluator JSON scores."""
    data = _extract_json_object(text)
    scores: dict[str, Any] = {}
    for key in SCORE_KEYS:
        if key not in data:
            raise AnswerEvaluationError(f"Gemini evaluator omitted score: {key}")
        try:
            value = float(data[key])
        except (TypeError, ValueError) as exc:
            raise AnswerEvaluationError(f"Gemini evaluator score {key} is not numeric") from exc
        if not 0.0 <= value <= 1.0:
            raise AnswerEvaluationError(f"Gemini evaluator score {key} must be between 0 and 1")
        scores[key] = value
    scores["reasoning"] = str(data.get("reasoning") or "").strip()
    return scores


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


class GeminiAnswerEvaluator:
    """Score generated answers with Gemini as an optional automated judge."""

    def __init__(
        self,
        *,
        model: str | None = None,
        max_output_tokens: int = 2048,
        max_retries: int | None = None,
        max_server_delay: float | None = None,
        retry_callback: Callable[[int | None, int, float], None] | None = None,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        load_dotenv(Path(".env"))
        self.model = model or os.getenv("GEMINI_EVAL_MODEL", os.getenv("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL))
        self.max_output_tokens = int(max_output_tokens)
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
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                self._client = self.client_factory()
            except Exception as exc:
                raise AnswerEvaluationError(
                    f"Unable to initialize Gemini evaluator: {type(exc).__name__}: {exc}"
                ) from exc
        return self._client

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[dict[str, Any]],
        reference_answer: str,
    ) -> dict[str, Any]:
        """Return automated Gemini scores and short reasoning for one answer."""
        messages = build_evaluation_prompt(
            question=question,
            reference_answer=reference_answer,
            context=format_context(contexts),
            answer=answer,
        )
        client = self._ensure_client()
        try:
            response = call_with_retries(
                lambda: client.invoke(messages),
                max_retries=self.max_retries,
                max_server_delay=self.max_server_delay,
                on_retry=self.retry_callback,
            )
        except Exception as exc:
            raise AnswerEvaluationError(
                f"Gemini answer evaluation failed: {type(exc).__name__}: {exc}"
            ) from exc
        return parse_evaluation_scores(message_text(response))
