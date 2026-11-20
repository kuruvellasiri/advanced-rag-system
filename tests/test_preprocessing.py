from langchain_core.documents import Document

from src.chunking.recursive_chunker import chunk_pages
from src.ingestion.pdf_loader import load_pdf_pages
from src.ingestion.section_router import separate_document_sections
from src.ingestion.text_cleaner import clean_pages, clean_text
from src.preprocess import SOURCE_PDF


def test_pdf_loader_retains_metadata_and_adds_one_based_page(tmp_path, monkeypatch):
    class FakeLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content="Page text", metadata={"page": 0, "source": self.path})]

    import src.ingestion.pdf_loader as loader_module

    monkeypatch.setattr(loader_module, "PyMuPDFLoader", FakeLoader)
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"placeholder")
    pages = load_pdf_pages(pdf_path)
    assert pages[0]["page_number"] == 1
    assert pages[0]["metadata"]["page"] == 0
    assert pages[0]["metadata"]["source"].endswith("sample.pdf")


def test_clean_text_normalizes_whitespace_and_keeps_paragraphs_and_bullets():
    raw = "  Heading  \r\n\r\nFirst   paragraph.\tMore text.\r\n\r\n\u2022   item one\r\n\u2022 item two  "
    assert clean_text(raw) == "Heading\n\nFirst paragraph. More text.\n\n\u2022 item one\n\u2022 item two"


def test_clean_pages_preserves_page_metadata():
    page = {"page_number": 4, "metadata": {"page": 3, "source": "x.pdf"}, "text": " A   B "}
    assert clean_pages([page]) == [
        {"page_number": 4, "metadata": {"page": 3, "source": "x.pdf"}, "text": "A B"}
    ]


def test_recursive_chunks_keep_metadata_ids_and_configured_overlap():
    page = {
        "page_number": 1,
        "metadata": {"page": 0, "page_number": 1, "source": "x.pdf"},
        "text": "A sentence for chunking. " * 60,
    }
    chunks = chunk_pages([page])
    assert len(chunks) > 1
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(len(chunk["text"]) <= 800 for chunk in chunks)
    assert all(chunk["metadata"]["source"] == "x.pdf" for chunk in chunks)
    assert chunks[0]["text"][-150:].strip() == chunks[1]["text"][:150].strip()


def test_chunker_rejects_invalid_overlap():
    try:
        chunk_pages([], chunk_size=100, chunk_overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= chunk_size")


def test_aws_pdf_pages_are_separated_without_losing_glossary_or_history():
    pages = clean_pages(load_pdf_pages(SOURCE_PDF))
    content, navigation, history = separate_document_sections(pages)

    assert len(pages) == 162
    assert {p["page_number"] for p in navigation} == set(range(3, 11))
    assert {p["page_number"] for p in history} == set(range(156, 162))
    assert 162 in {p["page_number"] for p in content}
    glossary_page = next(p for p in content if p["page_number"] == 162)
    assert "AWS Glossary" in glossary_page["text"]
    assert len(content) + len(navigation) + len(history) == len(pages)
