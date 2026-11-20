"""Load PDF pages with LangChain's PyMuPDFLoader."""

from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Return one page record per PDF page, retaining loader metadata.

    LangChain's ``page`` metadata is zero-based. ``page_number`` is added as a
    human-facing 1-based value while the original metadata remains untouched.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    documents = PyMuPDFLoader(str(path)).load()
    pages: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        metadata = dict(document.metadata)
        zero_based_page = int(metadata.get("page", index))
        page_number = zero_based_page + 1
        metadata["page_number"] = page_number
        pages.append({
            "page_number": page_number,
            "metadata": metadata,
            "text": document.page_content,
        })
    return pages
