"""Persistent Chroma collection for AWS content chunks."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.embeddings.factory import create_embedding_service, embedding_configuration
from src.embeddings.gemini_embeddings import DEFAULT_BATCH_SIZE

DEFAULT_PERSIST_DIRECTORY = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "aws_content"
DEFAULT_BACKUP_ROOT = Path("data/chroma_backups")


def backup_chroma(
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
) -> Path | None:
    """Copy the persistent Chroma directory before any destructive update."""
    source = Path(persist_directory)
    if not source.is_dir() or not any(source.iterdir()):
        return None
    destination_root = Path(backup_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = destination_root / f"chroma-{timestamp}"
    suffix = 1
    while destination.exists():
        suffix += 1
        destination = destination_root / f"chroma-{timestamp}-{suffix}"
    shutil.copytree(source, destination)
    return destination


def _chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata = dict(chunk.get("metadata") or {})
    if "page_number" not in metadata:
        if chunk.get("page_number") is not None:
            metadata["page_number"] = int(chunk["page_number"])
        elif metadata.get("page") is not None:
            metadata["page_number"] = int(metadata["page"]) + 1
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[str(key)] = value
        else:
            cleaned[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return cleaned


class ChromaStore:
    """Manage persistent Chroma indexing and cosine similarity searches."""

    def __init__(
        self,
        embedding_service: Any | None = None,
        *,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        retry_callback: Callable[[int | None, int, float], None] | None = None,
        client: Any | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is missing. Install with: python -m pip install -r requirements.txt") from exc
        if embedding_service is None:
            provider, model_name = embedding_configuration(provider, model_name)
            embedding_service = create_embedding_service(
                provider,
                model_name,
                retry_callback=retry_callback,
            )
        else:
            provider = provider or getattr(embedding_service, "provider", "custom")
            model_name = model_name or getattr(
                embedding_service, "model_name", getattr(embedding_service, "model", "custom")
            )
        self.embedding_service = embedding_service
        self.provider = str(provider)
        self.model_name = str(model_name)
        self.dimension = getattr(embedding_service, "dimension", None)
        if collection_name is None:
            if self.provider == "gemini" and self.model_name == "gemini-embedding-001":
                # Preserve access to the pre-provider Gemini collection.
                collection_name = DEFAULT_COLLECTION_NAME
            else:
                suffix = "_".join(
                    part for part in (self.provider, self.model_name.lower().replace("/", "_").replace("-", "_"))
                    if part
                )
                collection_name = f"{DEFAULT_COLLECTION_NAME}_{suffix}"
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = client or chromadb.PersistentClient(path=str(self.persist_directory))
        collection_metadata: dict[str, Any] = {
            "hnsw:space": "cosine",
            "embedding_model": self.model_name,
            "embedding_provider": self.provider,
        }
        if self.dimension is not None:
            collection_metadata["embedding_dimension"] = int(self.dimension)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=collection_metadata,
        )
        current_metadata = self.collection.metadata or {}
        existing_model = current_metadata.get("embedding_model")
        existing_provider = current_metadata.get("embedding_provider")
        if existing_model and existing_model != self.model_name:
            raise ValueError(
                f"Collection {self.collection_name!r} uses model {existing_model!r}, "
                f"but this client is configured for {self.model_name!r}. "
                "Choose a separate collection or explicitly migrate the collection."
            )
        if existing_provider and existing_provider != self.provider:
            raise ValueError(
                f"Collection {self.collection_name!r} uses provider {existing_provider!r}, "
                f"but this client is configured for {self.provider!r}. Choose a separate collection."
            )

    def count(self) -> int:
        return int(self.collection.count())

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        progress_callback: Callable[[dict[str, int]], None] | None = None,
        force: bool = False,
        remove_stale: bool = False,
    ) -> dict[str, int]:
        """Upsert in durable batches, embedding only new or text-changed IDs.

        Each batch is committed to Chroma before the next Gemini request. If an
        embedding call fails, a later run discovers completed IDs and resumes
        at the first missing or changed chunk. ``force`` re-embeds every chunk;
        ``remove_stale`` deletes stored IDs that are absent from ``chunks`` only
        after every new embedding has been persisted successfully.
        """
        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Input chunk IDs must be unique")
        if not chunks:
            return {
                "total": 0, "embedded": 0, "skipped": 0, "metadata_updated": 0,
                "removed": 0, "processed_chunks": 0, "remaining_chunks": 0,
                "failed_batches": 0,
            }

        existing = self.collection.get(ids=ids, include=["documents", "metadatas"])
        existing_map = {
            chunk_id: {"document": document, "metadata": metadata or {}}
            for chunk_id, document, metadata in zip(
                existing.get("ids", []), existing.get("documents", []), existing.get("metadatas", [])
            )
        }
        stale_ids: list[str] = []
        if remove_stale:
            stored_ids = set(self.collection.get(include=[]).get("ids", []))
            stale_ids = sorted(stored_ids - set(ids))

        to_embed: list[dict[str, Any]] = []
        unchanged_ids: list[str] = []
        unchanged_metadata: list[dict[str, Any]] = []
        same_text = 0
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            metadata = _chroma_metadata(chunk)
            current = existing_map.get(chunk_id)
            if force or current is None or current["document"] != chunk["text"]:
                to_embed.append({**chunk, "chunk_id": chunk_id, "metadata": metadata})
            else:
                same_text += 1
                if current["metadata"] != metadata:
                    unchanged_ids.append(chunk_id)
                    unchanged_metadata.append(metadata)

        if unchanged_ids:
            self.collection.update(ids=unchanged_ids, metadatas=unchanged_metadata)

        processed = same_text
        failed_batches = 0

        def report_progress() -> None:
            if progress_callback:
                progress_callback({
                    "total_chunks": len(chunks),
                    "processed_chunks": processed,
                    "remaining_chunks": len(chunks) - processed,
                    "failed_batches": failed_batches,
                })

        report_progress()
        persist_batch_size = max(1, int(getattr(self.embedding_service, "batch_size", DEFAULT_BATCH_SIZE)))
        batch_count = (len(to_embed) + persist_batch_size - 1) // persist_batch_size
        for batch_index, start in enumerate(range(0, len(to_embed), persist_batch_size)):
            batch = to_embed[start:start + persist_batch_size]
            try:
                embedded_records = self.embedding_service.embed_chunks(batch)
                expected_ids = [record["chunk_id"] for record in batch]
                actual_ids = [record.get("chunk_id") for record in embedded_records]
                if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
                    raise ValueError("Embedding service returned missing, duplicate, or unexpected chunk IDs")
                self.collection.upsert(
                    ids=[record["chunk_id"] for record in embedded_records],
                    documents=[record["text"] for record in embedded_records],
                    metadatas=[_chroma_metadata(record) for record in embedded_records],
                    embeddings=[record["embedding"] for record in embedded_records],
                )
            except Exception:
                failed_batches += 1
                report_progress()
                raise
            processed += len(batch)
            report_progress()
            if batch_index + 1 < batch_count:
                wait_between_batches = getattr(self.embedding_service, "wait_between_batches", None)
                if wait_between_batches:
                    wait_between_batches()

        if stale_ids:
            self.collection.delete(ids=stale_ids)

        return {
            "total": len(chunks),
            "embedded": len(to_embed),
            "skipped": same_text - len(unchanged_ids),
            "metadata_updated": len(unchanged_ids),
            "removed": len(stale_ids),
            "processed_chunks": processed,
            "remaining_chunks": len(chunks) - processed,
            "failed_batches": failed_batches,
        }

    def query_by_embedding(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        records = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            metadata = metadata or {}
            records.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "page_number": metadata.get("page_number"),
                "score": 1.0 - float(distance),
                "distance": float(distance),
                "rank": rank,
            })
        return records
