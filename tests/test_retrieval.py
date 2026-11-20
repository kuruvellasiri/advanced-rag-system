import pytest
from types import SimpleNamespace

from src.embeddings.gemini_embeddings import (
    GeminiEmbeddingError,
    GeminiEmbeddingService,
    RequestRateLimiter,
)
from src.embeddings.factory import create_embedding_service, embedding_configuration
from src.embeddings.sentence_transformer_embeddings import (
    EXPECTED_DIMENSION,
    LocalEmbeddingError,
    SentenceTransformerEmbeddingService,
)
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.vector_search import VectorSearch
from src.vectorstore.chroma_store import ChromaStore, backup_chroma
from src.rag.answer_generator import GeminiAnswerGenerator


class FakeSentenceTransformer:
    def __init__(self):
        self.batch_sizes = []

    def get_sentence_embedding_dimension(self):
        return EXPECTED_DIMENSION

    def encode(self, texts, *, batch_size, **kwargs):
        self.batch_sizes.append(len(texts))
        result = []
        for text in texts:
            vector = [0.0] * EXPECTED_DIMENSION
            vector[0 if "alpha" in text.casefold() else 1] = 1.0
            result.append(vector)
        return result


def test_local_sentence_transformer_embeddings_batch_and_match_dimensions():
    model = FakeSentenceTransformer()
    service = SentenceTransformerEmbeddingService(batch_size=2, model=model)
    vectors = service.embed_documents(["alpha one", "beta", "alpha two"])
    query = service.embed_query("alpha query")

    assert model.batch_sizes == [2, 1, 1]
    assert len(vectors) == 3
    assert all(len(vector) == EXPECTED_DIMENSION for vector in vectors)
    assert len(query) == EXPECTED_DIMENSION
    assert vectors[0] == query
    assert vectors[1][1] == 1.0


def test_local_embedding_wraps_model_errors():
    class BrokenModel(FakeSentenceTransformer):
        def encode(self, *args, **kwargs):
            raise ValueError("bad model output")

    service = SentenceTransformerEmbeddingService(model=BrokenModel())
    with pytest.raises(LocalEmbeddingError, match="bad model output"):
        service.embed_documents(["alpha"])


def test_embedding_factory_reads_local_provider_configuration(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "7")
    assert embedding_configuration() == ("local", "sentence-transformers/all-MiniLM-L6-v2")
    service = create_embedding_service(local_model=FakeSentenceTransformer())
    assert service.provider == "local"
    assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert service.batch_size == 7


class FakeGeminiClient:
    def __init__(self, vectors=None):
        self.vectors = vectors
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.vectors:
            return self.vectors[:len(texts)]
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, query):
        return [1.0, 0.0]


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_embedding_batches_preserve_chunk_ids_and_metadata():
    clients = {}

    def factory(model, task_type):
        clients[task_type] = FakeGeminiClient()
        return clients[task_type]

    service = GeminiEmbeddingService(
        client_factory=factory,
        batch_size=2,
        batch_pause=0,
        sleep=lambda _: None,
    )
    chunks = [
        {"chunk_id": f"id-{i}", "text": f"text {i}", "page_number": i + 1,
         "metadata": {"page_number": i + 1, "source": "aws.pdf"}}
        for i in range(3)
    ]
    embedded = service.embed_chunks(chunks)

    assert [item["chunk_id"] for item in embedded] == [item["chunk_id"] for item in chunks]
    assert [item["metadata"] for item in embedded] == [item["metadata"] for item in chunks]
    assert [item["page_number"] for item in embedded] == [1, 2, 3]
    assert all(len(item["embedding"]) == 2 for item in embedded)
    assert clients["RETRIEVAL_DOCUMENT"].calls == 2


def test_embedding_retries_gemini_rate_limits():
    class RateLimitedClient(FakeGeminiClient):
        def embed_documents(self, texts):
            self.calls += 1
            if self.calls == 1:
                error = RuntimeError("rate limit")
                error.status_code = 429
                raise error
            return [[1.0, 2.0] for _ in texts]

    client = RateLimitedClient()
    clock = FakeClock()
    delays = []

    def sleep(seconds):
        delays.append(seconds)
        clock.sleep(seconds)

    service = GeminiEmbeddingService(
        client_factory=lambda *_: client,
        max_retries=2,
        batch_pause=0,
        sleep=sleep,
        monotonic=clock.monotonic,
    )
    assert service.embed_texts(["hello"], task_type="RETRIEVAL_DOCUMENT") == [[1.0, 2.0]]
    assert client.calls == 2
    assert delays == [60.0]


def test_embedding_honors_google_retry_delay_and_reports_429_separately():
    class RateLimitedClient(FakeGeminiClient):
        def embed_documents(self, texts):
            self.calls += 1
            if self.calls == 1:
                error = RuntimeError("RESOURCE_EXHAUSTED")
                error.status_code = 429
                error.details = {"error": {"details": [{"retryDelay": "12s"}]}}
                raise error
            return [[1.0, 2.0] for _ in texts]

    client = RateLimitedClient()
    clock = FakeClock()
    delays, notices = [], []

    def sleep(seconds):
        delays.append(seconds)
        clock.sleep(seconds)

    service = GeminiEmbeddingService(
        client_factory=lambda *_: client,
        max_retries=1,
        batch_pause=0,
        sleep=sleep,
        monotonic=clock.monotonic,
        retry_callback=lambda status, attempt, delay: notices.append((status, attempt, delay)),
    )
    assert service.embed_texts(["hello"], task_type="RETRIEVAL_DOCUMENT") == [[1.0, 2.0]]
    assert delays == [12.0]
    assert notices == [(429, 1, 12.0)]


def test_embedding_rejects_excessive_quota_reset_delay():
    class ExhaustedClient(FakeGeminiClient):
        def embed_documents(self, texts):
            self.calls += 1
            error = RuntimeError("RESOURCE_EXHAUSTED")
            error.status_code = 429
            error.details = {"error": {"details": [{"retryDelay": "82259s"}]}}
            raise error

    client, delays = ExhaustedClient(), []
    service = GeminiEmbeddingService(
        client_factory=lambda *_: client,
        max_retries=3,
        batch_pause=0,
        max_server_delay=300,
        sleep=delays.append,
        monotonic=lambda: 0.0,
    )

    with pytest.raises(GeminiEmbeddingError, match="above the 300s limit"):
        service.embed_texts(["hello"], task_type="RETRIEVAL_DOCUMENT")
    assert client.calls == 1
    assert delays == []


def test_embedding_reads_quoted_retry_delay_from_wrapped_error_text():
    class ExhaustedClient(FakeGeminiClient):
        def embed_documents(self, texts):
            self.calls += 1
            error = RuntimeError("429 RESOURCE_EXHAUSTED {'retryDelay': '82259s'}")
            error.status_code = 429
            raise error

    client, delays = ExhaustedClient(), []
    service = GeminiEmbeddingService(
        client_factory=lambda *_: client,
        max_retries=3,
        batch_pause=0,
        max_server_delay=300,
        sleep=delays.append,
        monotonic=lambda: 0.0,
    )

    with pytest.raises(GeminiEmbeddingError, match="above the 300s limit"):
        service.embed_texts(["hello"], task_type="RETRIEVAL_DOCUMENT")
    assert client.calls == 1
    assert delays == []


def test_service_unavailable_uses_short_exponential_backoff():
    class UnavailableClient(FakeGeminiClient):
        def embed_documents(self, texts):
            self.calls += 1
            error = RuntimeError("UNAVAILABLE")
            error.status_code = 503
            raise error

    client, delays = UnavailableClient(), []
    clock = FakeClock()

    def sleep(seconds):
        delays.append(seconds)
        clock.sleep(seconds)

    service = GeminiEmbeddingService(
        client_factory=lambda *_: client,
        max_retries=2,
        base_delay=2,
        batch_pause=0,
        sleep=sleep,
        monotonic=clock.monotonic,
        jitter=lambda: 0.5,
    )
    with pytest.raises(GeminiEmbeddingError, match="503"):
        service.embed_texts(["hello"], task_type="RETRIEVAL_DOCUMENT")
    assert client.calls == 3
    assert delays == [2.0, 4.0]


def test_request_rate_limiter_enforces_configured_rpm():
    clock = FakeClock(now=100.0)
    limiter = RequestRateLimiter(80, monotonic=clock.monotonic, sleep=clock.sleep)
    limiter.wait()
    limiter.wait()
    assert clock.now == pytest.approx(100.75)


def test_batch_size_and_delay_read_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_EMBED_BATCH_SIZE", "4")
    monkeypatch.setenv("GEMINI_BATCH_DELAY_SECONDS", "1.25")
    monkeypatch.setenv("GEMINI_REQUESTS_PER_MINUTE", "70")
    service = GeminiEmbeddingService(client_factory=lambda *_: FakeGeminiClient())
    assert service.batch_size == 4
    assert service.batch_pause == 1.25
    assert service.requests_per_minute == 70
    with pytest.raises(ValueError, match="below the 100 RPM quota"):
        GeminiEmbeddingService(requests_per_minute=100, client_factory=lambda *_: FakeGeminiClient())


def test_embedding_surfaces_non_retryable_api_errors():
    class InvalidClient(FakeGeminiClient):
        def embed_documents(self, texts):
            self.calls += 1
            raise ValueError("invalid request")

    client = InvalidClient()
    service = GeminiEmbeddingService(
        client_factory=lambda *_: client,
        max_retries=4,
        batch_pause=0,
        sleep=lambda _: None,
    )
    with pytest.raises(GeminiEmbeddingError):
        service.embed_texts(["hello"], task_type="RETRIEVAL_DOCUMENT")
    assert client.calls == 1


class FakeEmbeddingService:
    def __init__(self):
        self.embed_calls = 0

    def embed_chunks(self, chunks):
        self.embed_calls += len(chunks)
        result = []
        for chunk in chunks:
            vector = [1.0, 0.0] if "alpha" in chunk["text"].lower() else [0.0, 1.0]
            result.append({**chunk, "embedding": vector})
        return result

    def embed_query(self, query):
        return [1.0, 0.0] if "alpha" in query.lower() else [0.0, 1.0]


def test_chroma_persists_chunks_and_skips_reembedding(tmp_path):
    embedder = FakeEmbeddingService()
    chunks = [
        {"chunk_id": "aws-1", "text": "Alpha service details", "page_number": 7,
         "metadata": {"page": 6, "page_number": 7, "source": "aws-overview.pdf"}},
        {"chunk_id": "aws-2", "text": "Beta service details", "page_number": 12,
         "metadata": {"page": 11, "page_number": 12, "source": "aws-overview.pdf"}},
    ]
    store = ChromaStore(embedder, persist_directory=tmp_path / "chroma-test", collection_name="persist_test")
    first = store.index_chunks(chunks)
    second = store.index_chunks(chunks)

    assert first == {
        "total": 2, "embedded": 2, "skipped": 0, "metadata_updated": 0, "removed": 0,
        "processed_chunks": 2, "remaining_chunks": 0, "failed_batches": 0,
    }
    assert second == {
        "total": 2, "embedded": 0, "skipped": 2, "metadata_updated": 0, "removed": 0,
        "processed_chunks": 2, "remaining_chunks": 0, "failed_batches": 0,
    }
    assert embedder.embed_calls == 2

    reopened = ChromaStore(
        FakeEmbeddingService(),
        persist_directory=tmp_path / "chroma-test",
        collection_name="persist_test",
    )
    results = VectorSearch(reopened).search("alpha", top_k=1)
    assert results[0]["chunk_id"] == "aws-1"
    assert results[0]["page_number"] == 7
    assert results[0]["metadata"]["page"] == 6
    assert reopened.count() == 2


def test_local_pipeline_uses_isolated_collection_and_preserves_gemini(tmp_path):
    persist = tmp_path / "multi-provider-chroma"
    chunks = [
        {"chunk_id": "local-alpha", "text": "Alpha service details", "page_number": 7,
         "metadata": {"page_number": 7, "source": "aws.pdf"}},
        {"chunk_id": "local-beta", "text": "Beta database details", "page_number": 8,
         "metadata": {"page_number": 8, "source": "aws.pdf"}},
    ]

    legacy_gemini = ChromaStore(
        FakeEmbeddingService(), persist_directory=persist, provider="gemini",
        model_name="gemini-embedding-001", collection_name="aws_content",
    )
    legacy_gemini.index_chunks([chunks[0]])
    old_count = legacy_gemini.count()

    model = FakeSentenceTransformer()
    service = SentenceTransformerEmbeddingService(batch_size=2, model=model)
    local_store = ChromaStore(service, persist_directory=persist)
    assert local_store.collection_name == "aws_content_local_sentence_transformers_all_minilm_l6_v2"
    assert local_store.collection_name != legacy_gemini.collection_name
    assert local_store.index_chunks(chunks)["embedded"] == 2
    assert legacy_gemini.count() == old_count

    query_vector = service.embed_query("alpha")
    vector_results = local_store.query_by_embedding(query_vector, top_k=2)
    assert vector_results[0]["chunk_id"] == "local-alpha"

    hybrid = HybridSearch(local_store, BM25Search(chunks))
    retrieval = hybrid.search("alpha service", top_k=2)
    assert retrieval["keyword"][0]["chunk_id"] == "local-alpha"
    assert retrieval["hybrid"][0]["chunk_id"] == "local-alpha"
    assert retrieval["hybrid"][0]["vector_rank"] == 1
    assert retrieval["hybrid"][0]["keyword_rank"] == 1

    calls_before_reindex = list(model.batch_sizes)
    reopened = ChromaStore(
        SentenceTransformerEmbeddingService(batch_size=2, model=model),
        persist_directory=persist,
    )
    assert reopened.index_chunks(chunks)["skipped"] == 2
    assert model.batch_sizes == calls_before_reindex

    class FakeAnswerClient:
        def invoke(self, messages):
            return SimpleNamespace(
                content="**Answer:** Alpha service details [1]"
            )

    answer = GeminiAnswerGenerator(hybrid, client=FakeAnswerClient()).generate(
        "What are alpha service details?", top_k=2
    )
    assert answer["sources"][0]["chunk_id"] == "local-alpha"
    assert "Document: aws.pdf" in answer["answer"]
    assert "Page: 7" in answer["answer"]
    assert "Chunk ID: local-alpha" in answer["answer"]


def test_chroma_reembeds_changed_text_but_only_updates_changed_metadata(tmp_path):
    embedder = FakeEmbeddingService()
    store = ChromaStore(embedder, persist_directory=tmp_path / "chroma-test", collection_name="update_test")
    chunk = {"chunk_id": "aws-1", "text": "Alpha service", "page_number": 7,
             "metadata": {"page_number": 7, "source": "aws.pdf"}}
    store.index_chunks([chunk])
    metadata_change = {**chunk, "metadata": {"page_number": 8, "source": "aws.pdf"}}
    result = store.index_chunks([metadata_change])
    assert result["embedded"] == 0
    assert result["metadata_updated"] == 1
    assert embedder.embed_calls == 1

    text_change = {**metadata_change, "text": "Alpha service changed"}
    result = store.index_chunks([text_change])
    assert result["embedded"] == 1
    assert embedder.embed_calls == 2


def test_chroma_can_force_reembed_and_remove_stale_ids(tmp_path):
    embedder = FakeEmbeddingService()
    store = ChromaStore(embedder, persist_directory=tmp_path / "chroma-test", collection_name="stale_test")
    chunks = [
        {"chunk_id": "aws-1", "text": "Alpha service", "page_number": 1,
         "metadata": {"page_number": 1}},
        {"chunk_id": "aws-2", "text": "Beta service", "page_number": 2,
         "metadata": {"page_number": 2}},
    ]
    store.index_chunks(chunks)
    forced = store.index_chunks(chunks, force=True, remove_stale=True)
    assert forced["embedded"] == 2
    assert forced["skipped"] == 0
    assert forced["removed"] == 0
    assert embedder.embed_calls == 4

    stale = store.index_chunks(chunks[:1], remove_stale=True)
    assert stale["embedded"] == 0
    assert stale["skipped"] == 1
    assert stale["removed"] == 1
    assert store.count() == 1
    assert store.collection.get(ids=["aws-2"])["ids"] == []


def test_backup_chroma_copies_existing_directory_before_rebuild(tmp_path):
    persist = tmp_path / "chroma"
    persist.mkdir()
    (persist / "chroma.sqlite3").write_text("state", encoding="utf-8")
    backup = backup_chroma(persist, tmp_path / "backups")
    assert backup is not None
    assert backup.parent == tmp_path / "backups"
    assert (backup / "chroma.sqlite3").read_text(encoding="utf-8") == "state"
    assert backup_chroma(tmp_path / "missing", tmp_path / "backups") is None


def test_indexing_resumes_from_last_persisted_batch_after_failure(tmp_path):
    class FailOnSecondBatch(FakeEmbeddingService):
        batch_size = 2

        def __init__(self):
            super().__init__()
            self.calls, self.seen_ids, self.fail = 0, [], True

        def embed_chunks(self, chunks):
            self.calls += 1
            self.seen_ids.extend(chunk["chunk_id"] for chunk in chunks)
            if self.fail and self.calls == 2:
                raise GeminiEmbeddingError("simulated 429 after first batch")
            return super().embed_chunks(chunks)

    service = FailOnSecondBatch()
    store = ChromaStore(service, persist_directory=tmp_path / "resume-test", collection_name="resume_test")
    chunks = [
        {"chunk_id": f"chunk-{i}", "text": f"Alpha content {i}", "page_number": i + 1,
         "metadata": {"page_number": i + 1}}
        for i in range(5)
    ]
    progress = []
    with pytest.raises(GeminiEmbeddingError):
        store.index_chunks(chunks, progress_callback=progress.append)

    assert store.count() == 2
    assert progress[-1] == {
        "total_chunks": 5, "processed_chunks": 2, "remaining_chunks": 3, "failed_batches": 1,
    }
    first_run_ids = list(service.seen_ids)
    service.fail = False
    resumed = store.index_chunks(chunks)
    assert resumed["embedded"] == 3
    assert resumed["skipped"] == 2
    assert resumed["processed_chunks"] == 5
    assert store.count() == 5
    assert service.seen_ids[len(first_run_ids):] == ["chunk-2", "chunk-3", "chunk-4"]


def test_bm25_exact_aws_service_name_query():
    search = BM25Search.from_json()
    results = search.search("Amazon S3", top_k=3)
    assert results
    assert any("s3" in item["text"].casefold() for item in results)
    assert all(item["page_number"] == item["metadata"]["page_number"] for item in results)


def test_bm25_conceptual_question_retrieves_cloud_computing_content():
    search = BM25Search.from_json()
    results = search.search(
        "What is cloud computing on-demand delivery of compute power database storage applications pay-as-you-go pricing",
        top_k=5,
    )
    assert results
    assert 12 in {item["page_number"] for item in results}
    assert all(item["score"] > 0 for item in results)


def test_rrf_fuses_rankings_deduplicates_and_uses_weights():
    vector = [
        {"chunk_id": "a", "text": "A", "page_number": 1, "score": 0.9},
        {"chunk_id": "b", "text": "B", "page_number": 2, "score": 0.8},
        {"chunk_id": "a", "text": "duplicate", "page_number": 1, "score": 0.7},
    ]
    keyword = [
        {"chunk_id": "b", "text": "B", "page_number": 2, "score": 4.0},
        {"chunk_id": "c", "text": "C", "page_number": 3, "score": 3.0},
    ]
    fused = reciprocal_rank_fusion(vector, keyword, top_k=3, vector_weight=1.0, keyword_weight=2.0)
    assert len({item["chunk_id"] for item in fused}) == 3
    assert fused[0]["chunk_id"] == "b"
    assert fused[0]["vector_rank"] == 2
    assert fused[0]["keyword_rank"] == 1
    assert fused[0]["page_number"] == 2


def test_rrf_accumulates_dense_and_bm25_scores_for_same_chunk_id():
    vector_shared = {
        "chunk_id": "shared", "text": "original dense text", "page_number": 17,
        "metadata": {"source": "aws.pdf", "page_number": 17}, "score": 0.8,
    }
    keyword_shared = {
        "chunk_id": "shared", "text": "different keyword object text", "page_number": 99,
        "metadata": {"source": "other.pdf", "page_number": 99}, "score": 7.0,
    }
    vector = [vector_shared, {"chunk_id": "dense-only", "text": "dense", "metadata": {}}]
    keyword = [keyword_shared, {"chunk_id": "bm25-only", "text": "keyword", "metadata": {}}]

    fused = reciprocal_rank_fusion(
        vector, keyword, top_k=10, vector_weight=0.5, keyword_weight=0.5, rrf_k=60,
    )
    shared = next(item for item in fused if item["chunk_id"] == "shared")

    assert shared["score"] == pytest.approx(0.5 / 61 + 0.5 / 61)
    assert shared["vector_rank"] == 1
    assert shared["keyword_rank"] == 1
    # The first retrieved representation retains its text and source metadata.
    assert shared["text"] == "original dense text"
    assert shared["page_number"] == 17
    assert shared["metadata"] == {"source": "aws.pdf", "page_number": 17}


def test_rrf_unique_ids_are_ranked_by_aggregated_score_and_keep_distinct_chunks():
    vector = [
        {"chunk_id": "dense-first", "text": "D1", "metadata": {"origin": "dense"}},
        {"chunk_id": "shared", "text": "shared body", "metadata": {"origin": "dense"}},
        {"chunk_id": "dense-third", "text": "D3", "metadata": {"origin": "dense"}},
    ]
    keyword = [
        {"chunk_id": "shared", "text": "duplicate body", "metadata": {"origin": "bm25"}},
        {"chunk_id": "keyword-second", "text": "K2", "metadata": {"origin": "bm25"}},
    ]

    fused = reciprocal_rank_fusion(vector, keyword, top_k=10, rrf_k=60)

    assert len(fused) == 4
    assert len({item["chunk_id"] for item in fused}) == 4
    assert [item["chunk_id"] for item in fused] == [
        "shared", "dense-first", "dense-third", "keyword-second",
    ]
    scores = [item["score"] for item in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0]["chunk_id"] == "shared"
    assert fused[0]["score"] == pytest.approx(1 / 62 + 0.5 / 61)
    assert fused[0]["text"] == "shared body"
    assert fused[0]["metadata"] == {"origin": "dense"}


def test_rrf_identical_ids_across_different_objects_are_one_fused_chunk():
    dense_instance = {"chunk_id": "same-id", "text": "same passage", "metadata": {"a": 1}}
    bm25_instance = {"chunk_id": "".join(["same", "-id"]), "text": "same passage", "metadata": {"a": 1}}
    assert dense_instance is not bm25_instance

    fused = reciprocal_rank_fusion([dense_instance], [bm25_instance], top_k=10)

    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "same-id"
    assert fused[0]["score"] == pytest.approx(1 / 61 + 0.5 / 61)


def test_hybrid_search_returns_vector_bm25_and_fused_results():
    chunks = [
        {"chunk_id": "aws-a", "text": "AWS Lambda runs code without provisioning servers.",
         "page_number": 41, "metadata": {"page_number": 41}},
        {"chunk_id": "aws-b", "text": "Amazon EC2 provides resizable compute capacity.",
         "page_number": 30, "metadata": {"page_number": 30}},
    ]

    class FakeStore:
        def count(self):
            return 2

        @property
        def embedding_service(self):
            return FakeEmbeddingService()

        def query_by_embedding(self, query_embedding, top_k):
            return [{**chunks[1], "score": 0.9, "rank": 1}]

    searcher = HybridSearch(FakeStore(), BM25Search(chunks))
    results = searcher.search("AWS Lambda", top_k=2, vector_weight=1.0, keyword_weight=1.0)
    assert results["vector"][0]["chunk_id"] == "aws-b"
    assert results["keyword"][0]["chunk_id"] == "aws-a"
    assert {item["chunk_id"] for item in results["hybrid"]} == {"aws-a", "aws-b"}
    assert results["vector_only"] == results["vector"]
    assert all(item.get("page_number") is not None for item in results["hybrid"])
