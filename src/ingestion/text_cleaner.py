"""Whitespace cleanup that retains page layout cues and list structure."""

import re
from typing import Any


def clean_text(text: str) -> str:
    """Normalize whitespace without joining paragraphs or bullet lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean page text while retaining page numbers and metadata."""
    return [{**page, "text": clean_text(page["text"])} for page in pages]
