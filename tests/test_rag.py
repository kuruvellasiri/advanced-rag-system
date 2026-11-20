from types import SimpleNamespace

import pytest

from src.embeddings.gemini_embeddings import GeminiEmbeddingError, call_with_retries
from src.rag.answer_generator import (
    AnswerGenerationError,
    GeminiAnswerGenerator,
    format_sources_block,
    sources_from_chunks,
)
from src.rag.prompts import NO_ANSWER_TEXT, build_answer_prompt, format_context
from src.retrieval.reranker import CrossEncoderReranker, RerankedHybridSearch


class FakeHybridSearch:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, question, *, top_k=5, vector_weight=1.0, keyword_weight=1.0):
        self.calls.append({
            "question": question,
            "top_k": top_k,
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
        })
        hybrid = self.chunks[:top_k]
        return {
            "vector": hybrid,
            "keyword": hybrid,
            "hybrid": hybrid,
            "vector_only": hybrid,
        }


class FakeChatClient:
    def __init__(self, response="Grounded AWS answer [1]", error=None):
        self.response = response
        self.error = error
        self.calls = 0
        self.messages = None

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.response)


CHUNKS = [
    {
        "chunk_id": "page-0149-chunk-0003",
        "page_number": 149,
        "metadata": {"page": 148, "page_number": 149, "source": "documents/aws-overview.pdf"},
        "text": "Amazon S3 is an object storage service.",
        "score": 0.9,
        "rank": 1,
    },
    {
        "chunk_id": "page-0149-chunk-0004",
        "page_number": 149,
        "metadata": {"page": 148, "page_number": 149, "source": "documents/aws-overview.pdf"},
        "text": "Amazon S3 is designed for 99.999999999% durability.",
        "score": 0.8,
        "rank": 2,
    },
]


def test_format_context_includes_page_chunk_id_and_text():
    formatted = format_context(CHUNKS)
    assert "[1] page 149 | chunk-id page-0149-chunk-0003" in formatted
    assert "Amazon S3 is an object storage service." in formatted
    assert format_context([]) == "(no retrieved context)"


def test_answer_prompt_requests_numbered_context_citations():
    messages = build_answer_prompt("What is Amazon S3?", CHUNKS)
    prompt_text = "\n".join(message.content for message in messages)
    assert "[1] page 149 | chunk-id page-0149-chunk-0003" in prompt_text
    assert "[2]" in prompt_text
    assert "[2] or [1, 5]" in prompt_text


def test_answer_prompt_forbids_unsupported_claims_and_outside_knowledge():
    messages = build_answer_prompt("Explain Game Tech", CHUNKS)
    prompt_text = "\n".join(message.content for message in messages).casefold()

    assert "do not use general knowledge" in prompt_text
    assert "training knowledge" in prompt_text
    assert "include a claim only when it is explicitly stated" in prompt_text
    assert "do not add features, benefits, examples" in prompt_text
    assert "answer only that part" in prompt_text
    assert "keep the answer concise" in prompt_text


def test_partially_supported_answer_stays_within_retrieved_facts():
    partial_chunk = {
        "chunk_id": "page-0072-chunk-0003",
        "page_number": 72,
        "metadata": {"page_number": 72, "source": "documents/aws-overview.pdf"},
        "text": "Amazon GameLift Servers provides dedicated game servers.",
    }
    client = FakeChatClient(
        "**Answer:** Amazon GameLift Servers provides dedicated game servers [1]."
    )
    result = GeminiAnswerGenerator(
        FakeHybridSearch([partial_chunk]), client=client
    ).generate("Explain Game Tech")

    assert "provides dedicated game servers" in result["answer"]
    assert "latency" not in result["answer"].casefold()
    assert "monthly contracts" not in result["answer"].casefold()
    prompt_text = " ".join(
        " ".join(message.content.split()) for message in client.messages
    ).casefold()
    assert "answer only that part" in prompt_text
    assert "omit any feature, benefit, example, or detail" in prompt_text


def test_sources_are_deduplicated_and_preserve_page_numbers():
    sources = sources_from_chunks(CHUNKS + [dict(CHUNKS[0], rank=3)])
    assert sources == [
        {
            "chunk_id": "page-0149-chunk-0003",
            "page_number": 149,
            "document_name": "aws-overview.pdf",
            "rank": 1,
            "score": 0.9,
        },
        {
            "chunk_id": "page-0149-chunk-0004",
            "page_number": 149,
            "document_name": "aws-overview.pdf",
            "rank": 2,
            "score": 0.8,
        },
    ]
    block = format_sources_block(sources)
    assert "Document: aws-overview.pdf" in block
    assert "Page: 149" in block
    assert "page-0149-chunk-0003" in block


def test_answer_generation_uses_hybrid_results_and_appends_sources():
    searcher = FakeHybridSearch(CHUNKS)
    client = FakeChatClient()
    generator = GeminiAnswerGenerator(searcher, client=client, top_k=2)

    result = generator.generate("What is Amazon S3?", top_k=2)

    assert result["model_answer"].startswith("Grounded AWS answer")
    assert "**Source References:**" in result["answer"]
    assert result["answer"].startswith("**Answer:**")
    assert "aws-overview.pdf" in result["answer"]
    assert "Page: 149" in result["answer"]
    assert "Chunk ID: page-0149-chunk-0003" in result["answer"]
    assert "[1]" not in result["answer"]
    assert result["sources"][0]["chunk_id"] == "page-0149-chunk-0003"
    assert len(result["sources"]) == 1
    assert result["retrieved"] == CHUNKS
    assert searcher.calls[0]["top_k"] == 2
    assert client.calls == 1
    human_prompt = client.messages[-1].content
    assert "Amazon S3 is an object storage service." in human_prompt
    assert "What is Amazon S3?" in human_prompt


def test_reranked_chunks_are_exactly_the_context_passed_to_answer_generation():
    class FakeCrossEncoder:
        def predict(self, pairs, **_kwargs):
            return [1.0 if "durability" in text else 0.0 for _, text in pairs]

    search = RerankedHybridSearch(
        FakeHybridSearch(CHUNKS),
        CrossEncoderReranker(model=FakeCrossEncoder()),
        candidate_count=2,
        rerank_top_k=1,
    )
    client = FakeChatClient("**Answer:** Durable storage [1].")
    result = GeminiAnswerGenerator(search, client=client, top_k=1).generate(
        "What is Amazon S3?", top_k=1,
    )

    assert result["retrieved"][0]["chunk_id"] == "page-0149-chunk-0004"
    assert "Amazon S3 is designed for 99.999999999% durability." in client.messages[-1].content
    assert "Amazon S3 is an object storage service." not in client.messages[-1].content


def test_answer_removes_inline_citations_and_lists_unique_valid_sources_at_end():
    response = (
        "**Answer:** Amazon S3 stores objects [1]. "
        "It is designed for high durability [2]. [1]\n\n"
        "**Source References:**\n- Invalid model-generated reference"
    )
    result = GeminiAnswerGenerator(
        FakeHybridSearch(CHUNKS), client=FakeChatClient(response)
    ).generate("What is Amazon S3?")

    answer, references = result["answer"].split("**Source References:**", maxsplit=1)
    assert answer.startswith("**Answer:**")
    assert "Amazon S3 stores objects." in answer
    assert "It is designed for high durability." in answer
    assert "[" not in answer and "]" not in answer
    assert references.count("Chunk ID:") == 2
    assert "Chunk ID: page-0149-chunk-0003" in references
    assert "Chunk ID: page-0149-chunk-0004" in references
    assert [source["chunk_id"] for source in result["sources"]] == [
        "page-0149-chunk-0003", "page-0149-chunk-0004",
    ]


def test_single_numbered_citation_maps_to_its_retrieved_chunk():
    result = GeminiAnswerGenerator(
        FakeHybridSearch(CHUNKS),
        client=FakeChatClient("**Answer:** The service is durable [2]."),
    ).generate("What is Amazon S3?")

    assert [source["chunk_id"] for source in result["sources"]] == ["page-0149-chunk-0004"]
    assert result["sources"][0]["page_number"] == 149
    assert result["sources"][0]["document_name"] == "aws-overview.pdf"
    assert "Chunk ID: page-0149-chunk-0004" in result["answer"]
    assert "[2]" not in result["answer"]


def test_multiple_citation_indices_map_to_first_and_fifth_chunks():
    chunks = CHUNKS + [
        {
            "chunk_id": f"page-015{i}-chunk-0001",
            "page_number": 150 + i,
            "metadata": {"page_number": 150 + i, "source": "documents/aws-overview.pdf"},
            "text": f"Supporting content {i}.",
        }
        for i in range(1, 4)
    ]
    result = GeminiAnswerGenerator(
        FakeHybridSearch(chunks),
        client=FakeChatClient("**Answer:** Supported by excerpts [1, 5]."),
        top_k=5,
    ).generate("What is Amazon S3?", top_k=5)

    assert [source["chunk_id"] for source in result["sources"]] == [
        "page-0149-chunk-0003", "page-0153-chunk-0001",
    ]
    assert [source["page_number"] for source in result["sources"]] == [149, 153]


def test_out_of_range_citation_returns_insufficient_context():
    result = GeminiAnswerGenerator(
        FakeHybridSearch(CHUNKS), client=FakeChatClient("**Answer:** Supported [99].")
    ).generate("What is Amazon S3?")
    assert result["answer"] == NO_ANSWER_TEXT
    assert result["model_answer"] == NO_ANSWER_TEXT
    assert result["sources"] == []


@pytest.mark.parametrize("citation", ["[1,]", "[1, foo]", "[1", "[0]"])
def test_malformed_citation_returns_insufficient_context(citation):
    result = GeminiAnswerGenerator(
        FakeHybridSearch(CHUNKS), client=FakeChatClient(f"**Answer:** Unsupported {citation}")
    ).generate("What is Amazon S3?")
    assert result["answer"] == NO_ANSWER_TEXT
    assert result["sources"] == []


def test_answer_without_valid_context_citations_returns_insufficient_context():
    result = GeminiAnswerGenerator(
        FakeHybridSearch(CHUNKS), client=FakeChatClient("A guessed answer without a citation.")
    ).generate("What is Amazon S3?")
    assert result["answer"] == NO_ANSWER_TEXT
    assert result["model_answer"] == NO_ANSWER_TEXT
    assert result["sources"] == []


def test_answer_generation_without_context_skips_llm():
    searcher = FakeHybridSearch([])
    client = FakeChatClient()
    generator = GeminiAnswerGenerator(searcher, client=client)

    result = generator.generate("What is an unrelated topic?")

    assert result["answer"] == NO_ANSWER_TEXT
    assert result["model_answer"] == NO_ANSWER_TEXT
    assert result["sources"] == []
    assert client.calls == 0


def test_answer_generation_wraps_gemini_errors():
    client = FakeChatClient(error=ValueError("invalid request"))
    generator = GeminiAnswerGenerator(
        FakeHybridSearch(CHUNKS),
        client=client,
        max_retries=0,
    )

    with pytest.raises(AnswerGenerationError, match="invalid request"):
        generator.generate("What is Amazon S3?")


def test_answer_generation_wraps_retrieval_quota_errors():
    class FailingHybridSearch:
        def search(self, *args, **kwargs):
            raise RuntimeError("Gemini quota exhaustion: requested a 6311s retry delay, above the 300s limit")

    generator = GeminiAnswerGenerator(FailingHybridSearch(), client=FakeChatClient())
    with pytest.raises(AnswerGenerationError, match="Question retrieval failed.*quota"):
        generator.generate("What is Amazon Lightsail?")


def test_answer_generation_handles_missing_api_key(monkeypatch):
    import src.rag.answer_generator as generator_module
    import src.rag.gemini_client as client_module

    monkeypatch.setattr(generator_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(client_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    generator = GeminiAnswerGenerator(FakeHybridSearch(CHUNKS))

    with pytest.raises(AnswerGenerationError, match="GOOGLE_API_KEY"):
        generator.generate("What is Amazon S3?")


def test_retry_helper_rejects_excessive_server_retry_delay():
    class QuotaError(RuntimeError):
        status_code = 429
        retry_delay = 82259

    sleeps = []

    def operation():
        raise QuotaError("quota exhausted")

    with pytest.raises(GeminiEmbeddingError, match="above the"):
        call_with_retries(
            operation,
            max_retries=3,
            max_server_delay=300,
            sleep=sleeps.append,
        )
    assert sleeps == []
