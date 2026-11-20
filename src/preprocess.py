"""Run PDF extraction, text cleanup, and recursive chunking."""

import argparse
import json
from pathlib import Path
from typing import Any

from .chunking.recursive_chunker import chunk_pages
from .ingestion.pdf_loader import load_pdf_pages
from .ingestion.section_router import separate_document_sections
from .ingestion.text_cleaner import clean_pages

SOURCE_PDF = Path("documents/aws-overview.pdf")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_preprocessing(
    source_pdf: str | Path = SOURCE_PDF,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict[str, int]:
    """Run all preprocessing stages for the configured AWS whitepaper."""
    extracted_pages = load_pdf_pages(source_pdf)
    cleaned_pages = clean_pages(extracted_pages)
    content_pages, navigation_pages, history_pages = separate_document_sections(cleaned_pages)

    content_chunks = chunk_pages(content_pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    navigation_chunks = chunk_pages(navigation_pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    history_chunks = chunk_pages(history_pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    write_json(Path("data/extracted_text/extracted_pages.json"), extracted_pages)
    write_json(Path("data/extracted_text/cleaned_pages.json"), cleaned_pages)
    write_json(Path("data/extracted_text/document_history_pages.json"), history_pages)
    write_json(Path("data/chunks/content_chunks.json"), content_chunks)
    write_json(Path("data/chunks/navigation_chunks.json"), navigation_chunks)
    write_json(Path("data/chunks/document_history_chunks.json"), history_chunks)

    return {
        "pages": len(extracted_pages),
        "content_pages": len(content_pages),
        "navigation_pages": len(navigation_pages),
        "history_pages": len(history_pages),
        "content_chunks": len(content_chunks),
        "navigation_chunks": len(navigation_chunks),
        "history_chunks": len(history_chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess documents/aws-overview.pdf")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    try:
        summary = run_preprocessing(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    except ValueError as exc:
        parser.error(str(exc))
    print("AWS PDF preprocessing complete: " + ", ".join(f"{key}={value}" for key, value in summary.items()))


if __name__ == "__main__":
    main()
