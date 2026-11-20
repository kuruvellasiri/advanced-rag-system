"""Route extracted pages into content, navigation, and document history."""

import re
from typing import Any


NAVIGATION_PAGES = set(range(3, 11))
_DOCUMENT_HISTORY_HEADING = re.compile(r"(?im)^\s*document\s+history\s*$")
_GLOSSARY_HEADING = re.compile(r"(?im)^\s*aws\s+glossary\s*$")


def separate_document_sections(
    pages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(content, navigation, history)`` without dropping any page.

    Navigation uses the specified physical PDF pages 3–10. Document history is
    routed from its standalone heading through the page before the standalone
    AWS Glossary heading. The glossary page remains in normal content.
    """
    content: list[dict[str, Any]] = []
    navigation: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    in_document_history = False

    for page in pages:
        page_number = int(page["page_number"])
        text = page["text"]

        if page_number in NAVIGATION_PAGES:
            navigation.append(page)
            continue

        if _DOCUMENT_HISTORY_HEADING.search(text):
            in_document_history = True
        if _GLOSSARY_HEADING.search(text):
            in_document_history = False

        if in_document_history:
            history.append(page)
        else:
            content.append(page)

    return content, navigation, history
