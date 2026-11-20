"""Recursive LangChain text chunking with page-level metadata retention."""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 150
) -> list[dict[str, Any]]:
    """Split pages recursively, assigning deterministic page-scoped IDs."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[dict[str, Any]] = []
    page_chunk_counts: dict[int, int] = {}

    for page in pages:
        page_number = int(page["page_number"])
        source_document = Document(
            page_content=page["text"],
            metadata=dict(page["metadata"]),
        )
        for chunk_document in splitter.split_documents([source_document]):
            page_chunk_counts[page_number] = page_chunk_counts.get(page_number, 0) + 1
            chunks.append({
                "chunk_id": f"page-{page_number:04d}-chunk-{page_chunk_counts[page_number]:04d}",
                "page_number": page_number,
                "metadata": dict(chunk_document.metadata),
                "text": chunk_document.page_content,
            })
    return chunks
