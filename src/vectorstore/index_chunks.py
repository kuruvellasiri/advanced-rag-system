"""Embed and persist the existing content chunks into ChromaDB."""

import argparse
import json
from pathlib import Path

from .chroma_store import ChromaStore, backup_chroma

CHUNKS_PATH = Path("data/chunks/content_chunks.json")


def _chunk_length_stats(chunks: list[dict]) -> tuple[int, int, int]:
    lengths = [len(chunk.get("text", "")) for chunk in chunks]
    if not lengths:
        return 0, 0, 0
    return min(lengths), max(lengths), sum(1 for length in lengths if length > 500)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Embed content chunks into ChromaDB")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-embed every current chunk while preserving existing stored IDs",
    )
    args = parser.parse_args(argv)

    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Content chunks not found: {CHUNKS_PATH}. Run preprocessing first.")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    min_length, max_length, over_500 = _chunk_length_stats(chunks)
    print(
        f"Loaded {len(chunks)} chunks | text length min={min_length}, "
        f"max={max_length}, chunks>500={over_500}",
        flush=True,
    )
    if chunks and max_length <= 500:
        print(
            "Warning: no chunk exceeds 500 characters. Regenerate chunks with "
            "`python -m src.preprocess` before re-indexing.",
            flush=True,
        )

    backup = backup_chroma()
    if backup:
        print(f"Backed up existing ChromaDB to {backup}", flush=True)
    else:
        print("No existing ChromaDB data to back up.", flush=True)

    def retry_notice(status: int | None, attempt: int, delay: float) -> None:
        label = "429 quota exhausted" if status == 429 else f"HTTP {status or 'transient'}"
        print(f"Gemini {label}; retry {attempt} after {delay:.1f}s", flush=True)

    store = ChromaStore(retry_callback=retry_notice)
    print(
        f"Embedding provider: {store.provider} | model: {store.model_name} | "
        f"collection: {store.collection_name}",
        flush=True,
    )

    def progress(state: dict[str, int]) -> None:
        print(
            f"Total chunks: {state['total_chunks']} | "
            f"Processed: {state['processed_chunks']} | "
            f"Remaining: {state['remaining_chunks']} | "
            f"Failed batches: {state['failed_batches']}",
            flush=True,
        )

    try:
        summary = store.index_chunks(
            chunks,
            progress_callback=progress,
            force=args.rebuild,
            # Preserve indexed records when a source chunk disappears. This
            # command never deletes existing collection data.
            remove_stale=False,
        )
    except Exception as exc:
        print(f"Indexing stopped after preserving completed batches: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1) from exc
    print("Chroma indexing complete: " + ", ".join(f"{key}={value}" for key, value in summary.items()))
    print(f"Persistent collection contains {store.count()} chunks.")


if __name__ == "__main__":
    main()
