"""LangChain Gemini embeddings with quota-aware rate limiting and retries."""

from __future__ import annotations

import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

MODEL_NAME = "gemini-embedding-001"
DEFAULT_BATCH_SIZE = 32
DEFAULT_BATCH_DELAY_SECONDS = 1.0
DEFAULT_REQUESTS_PER_MINUTE = 80
DEFAULT_429_BACKOFF_SECONDS = 60.0
DEFAULT_503_BACKOFF_SECONDS = 2.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_BACKOFF_SECONDS = 300.0
DEFAULT_MAX_SERVER_DELAY_SECONDS = 300.0


class GeminiEmbeddingError(RuntimeError):
    """An embedding request failed after retry handling."""


class RequestRateLimiter:
    """Thread-safe minimum-interval limiter shared by document and query calls."""

    def __init__(
        self,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= requests_per_minute < 100:
            raise ValueError("requests_per_minute must be between 1 and 99 to stay below the 100 RPM quota")
        self.interval = 60.0 / requests_per_minute
        self.monotonic = monotonic
        self.sleep = sleep
        self.next_request_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = self.monotonic()
            request_at = max(now, self.next_request_at)
            delay = request_at - now
            if delay > 0:
                self.sleep(delay)
            self.next_request_at = request_at + self.interval


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _exception_chain(exc: Exception):
    seen: set[int] = set()
    current: Exception | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_from_value(value: Any) -> int | None:
    if isinstance(value, int) and 100 <= value <= 599:
        return value
    if value is None:
        return None
    text = str(getattr(value, "name", value)).upper()
    for status, markers in {
        429: ("429", "RESOURCE_EXHAUSTED", "TOO_MANY_REQUESTS"),
        503: ("503", "UNAVAILABLE", "SERVICE_UNAVAILABLE"),
        408: ("408", "DEADLINE_EXCEEDED"),
        500: ("500", "INTERNAL"),
        502: ("502", "BAD_GATEWAY"),
        504: ("504", "GATEWAY_TIMEOUT"),
    }.items():
        if any(marker in text for marker in markers):
            return status
    return None


def _error_status(exc: Exception) -> tuple[int | None, str]:
    texts = []
    for item in _exception_chain(exc):
        texts.append(f"{type(item).__name__} {item}")
        for name in ("status_code", "code", "status"):
            status = _status_from_value(getattr(item, name, None))
            if status is not None:
                return status, " ".join(texts).upper()
        response = getattr(item, "response", None)
        status = _status_from_value(getattr(response, "status_code", None))
        if status is not None:
            return status, " ".join(texts).upper()
    combined = " ".join(texts).upper()
    status_match = re.search(r"\b(429|503|408|500|502|504)\b", combined)
    return (int(status_match.group(1)) if status_match else None), combined


def _duration_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, dict):
        seconds = float(value.get("seconds", 0) or 0)
        nanos = float(value.get("nanos", 0) or 0)
        return max(0.0, seconds + nanos / 1_000_000_000)
    if not isinstance(value, str):
        return None
    value = value.strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds?|m|min|minutes?)?", value, re.I)
    if match:
        amount = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit == "ms":
            return amount / 1000
        if unit in {"m", "min", "minute", "minutes"}:
            return amount * 60
        return amount
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _find_retry_delay(value: Any) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).replace("_", "").replace("-", "").casefold()
            if normalized in {"retrydelay", "retryafter"}:
                parsed = _duration_seconds(item)
                if parsed is not None:
                    return parsed
        for item in value.values():
            parsed = _find_retry_delay(item)
            if parsed is not None:
                return parsed
    elif isinstance(value, (list, tuple)):
        for item in value:
            parsed = _find_retry_delay(item)
            if parsed is not None:
                return parsed
    elif isinstance(value, str):
        match = re.search(
            r"retry[-_ ]?delay['\"]?\s*[=:]\s*['\"]?([0-9.]+\s*(?:ms|s|m)?)",
            value,
            re.I,
        )
        if match:
            return _duration_seconds(match.group(1))
    return None


def _server_retry_delay(exc: Exception) -> float | None:
    for item in _exception_chain(exc):
        for name in ("retry_delay", "retryDelay", "retry_after", "retry_after_seconds"):
            parsed = _duration_seconds(getattr(item, name, None))
            if parsed is not None:
                return parsed
        details = getattr(item, "details", None)
        parsed = _find_retry_delay(details)
        if parsed is not None:
            return parsed
        response = getattr(item, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            parsed = _duration_seconds(headers.get("Retry-After") or headers.get("retry-after"))
            if parsed is not None:
                return parsed
    return _find_retry_delay(str(exc))


def _is_retryable(exc: Exception) -> bool:
    status, text = _error_status(exc)
    if status in {408, 429, 500, 502, 503, 504}:
        return True
    return any(marker in text for marker in (
        "RESOURCE_EXHAUSTED", "TOO MANY REQUESTS", "RATE LIMIT", "UNAVAILABLE",
        "SERVICE UNAVAILABLE", "INTERNAL SERVER ERROR", "DEADLINE EXCEEDED",
        "TIMED OUT", "TIMEOUT", "CONNECTION RESET", "TEMPORARY",
    ))


def call_with_retries(
    operation: Callable[[], Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_503_BACKOFF_SECONDS,
    quota_backoff: float = DEFAULT_429_BACKOFF_SECONDS,
    max_backoff: float = DEFAULT_MAX_BACKOFF_SECONDS,
    max_server_delay: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
    before_request: Callable[[], None] | None = None,
    on_retry: Callable[[int | None, int, float], None] | None = None,
) -> Any:
    """Retry 429/503 separately, respecting Retry-After and Google retryDelay.

    ``max_server_delay`` optionally rejects provider delays that are too long to
    wait for, such as daily-quota reset delays, instead of sleeping for hours.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    retry_counts: dict[int | str, int] = {}
    for attempt in range(max_retries + 1):
        if before_request:
            before_request()
        try:
            return operation()
        except Exception as exc:
            status, _ = _error_status(exc)
            if attempt >= max_retries or not _is_retryable(exc):
                category = {429: "rate limit (429)", 503: "service unavailable (503)"}.get(status, f"API error ({status or 'unknown'})")
                raise GeminiEmbeddingError(
                    f"Gemini {category} failed after {attempt + 1} attempt(s): {exc}"
                ) from exc

            retry_key: int | str = status or "transient"
            retry_counts[retry_key] = retry_counts.get(retry_key, 0) + 1
            server_delay = _server_retry_delay(exc)
            if server_delay is not None and max_server_delay is not None and server_delay > max_server_delay:
                raise GeminiEmbeddingError(
                    f"Gemini requested a {server_delay:.0f}s retry delay, above the "
                    f"{max_server_delay:.0f}s limit. Treating this as quota exhaustion."
                ) from exc
            if server_delay is not None:
                delay = max(server_delay, 1.0) if status == 429 else server_delay
            elif status == 429:
                delay = min(quota_backoff * (2 ** (retry_counts[retry_key] - 1)), max_backoff)
            else:
                delay = min(base_delay * (2 ** (retry_counts[retry_key] - 1)), max_backoff)
                delay *= 0.75 + 0.5 * jitter()
            if on_retry:
                on_retry(status, attempt + 1, delay)
            sleep(delay)
    raise AssertionError("unreachable")


class GeminiEmbeddingService:
    """Embed recursive chunk records with a configured quota-safe request pace."""

    def __init__(
        self,
        model: str = MODEL_NAME,
        *,
        client_factory: Callable[[str, str], Any] | None = None,
        max_retries: int | None = None,
        base_delay: float | None = None,
        batch_size: int | None = None,
        batch_pause: float | None = None,
        requests_per_minute: int | None = None,
        quota_backoff: float | None = None,
        max_backoff: float | None = None,
        max_server_delay: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
        retry_callback: Callable[[int | None, int, float], None] | None = None,
    ) -> None:
        load_dotenv(Path(".env"))
        self.model = model
        self.client_factory = client_factory or self._default_client_factory
        self.max_retries = max_retries if max_retries is not None else _env_int("GEMINI_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        self.base_delay = base_delay if base_delay is not None else _env_float("GEMINI_503_BACKOFF_SECONDS", DEFAULT_503_BACKOFF_SECONDS)
        self.batch_size = batch_size if batch_size is not None else _env_int("GEMINI_EMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        self.batch_pause = batch_pause if batch_pause is not None else _env_float(
            "GEMINI_BATCH_DELAY_SECONDS", DEFAULT_BATCH_DELAY_SECONDS
        )
        self.requests_per_minute = requests_per_minute if requests_per_minute is not None else _env_int(
            "GEMINI_REQUESTS_PER_MINUTE", DEFAULT_REQUESTS_PER_MINUTE
        )
        self.quota_backoff = quota_backoff if quota_backoff is not None else _env_float(
            "GEMINI_429_BACKOFF_SECONDS", DEFAULT_429_BACKOFF_SECONDS
        )
        self.max_backoff = max_backoff if max_backoff is not None else _env_float(
            "GEMINI_MAX_BACKOFF_SECONDS", DEFAULT_MAX_BACKOFF_SECONDS
        )
        self.max_server_delay = max_server_delay if max_server_delay is not None else _env_float(
            "GEMINI_MAX_SERVER_RETRY_SECONDS", DEFAULT_MAX_SERVER_DELAY_SECONDS
        )
        if not 1 <= self.batch_size <= 100:
            raise ValueError("GEMINI_EMBED_BATCH_SIZE must be between 1 and 100")
        if self.max_retries < 0:
            raise ValueError("GEMINI_MAX_RETRIES must be non-negative")
        if self.quota_backoff <= 0 or self.max_backoff <= 0:
            raise ValueError("429 and maximum backoff values must be greater than zero")
        self.sleep = sleep
        self.jitter = jitter
        self.retry_callback = retry_callback
        self.rate_limiter = RequestRateLimiter(
            self.requests_per_minute,
            monotonic=monotonic,
            sleep=sleep,
        )
        self._clients: dict[str, Any] = {}

    @staticmethod
    def _default_client_factory(model: str, task_type: str) -> Any:
        load_dotenv(Path(".env"))
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env before using Gemini.")
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from google import genai
            from google.genai.types import HttpOptions, HttpRetryOptions
        except ImportError as exc:
            raise RuntimeError(
                "Gemini integration is missing. Install requirements with: python -m pip install -r requirements.txt"
            ) from exc
        timeout_ms = int(_env_float("GEMINI_EMBED_REQUEST_TIMEOUT_SECONDS", 30.0) * 1000)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(
                timeout=timeout_ms,
                retry_options=HttpRetryOptions(attempts=1),
            ),
        )
        embeddings = GoogleGenerativeAIEmbeddings(model=model, task_type=task_type)
        embeddings.client = client
        return embeddings

    def _client(self, task_type: str) -> Any:
        if task_type not in self._clients:
            self._clients[task_type] = self.client_factory(self.model, task_type)
        return self._clients[task_type]

    def _retry(self, operation: Callable[[], Any]) -> Any:
        return call_with_retries(
            operation,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            quota_backoff=self.quota_backoff,
            max_backoff=self.max_backoff,
            max_server_delay=self.max_server_delay,
            sleep=self.sleep,
            jitter=self.jitter,
            before_request=self.rate_limiter.wait,
            on_retry=self.retry_callback,
        )

    def wait_between_batches(self) -> None:
        if self.batch_pause:
            self.sleep(self.batch_pause)

    def embed_texts(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        """Embed text batches, spacing every provider request and retry."""
        if not texts:
            return []
        client = self._client(task_type)
        vectors: list[list[float]] = []
        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start:batch_start + self.batch_size]
            batch_vectors = self._retry(lambda batch=batch: client.embed_documents(batch))
            if len(batch_vectors) != len(batch):
                raise GeminiEmbeddingError(
                    f"Gemini returned {len(batch_vectors)} vectors for {len(batch)} input texts."
                )
            vectors.extend([[float(value) for value in vector] for vector in batch_vectors])
            if batch_start + self.batch_size < len(texts) and self.batch_pause:
                self.sleep(self.batch_pause)
        return vectors

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return records with embeddings while preserving IDs and metadata."""
        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Input chunk IDs must be unique")
        vectors = self.embed_texts([chunk["text"] for chunk in chunks], task_type="RETRIEVAL_DOCUMENT")
        return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]

    def embed_query(self, query: str) -> list[float]:
        client = self._client("RETRIEVAL_QUERY")
        vector = self._retry(lambda: client.embed_query(query))
        return [float(value) for value in vector]
