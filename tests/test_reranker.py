import pytest

from src.retrieval import cli
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankedHybridSearch,
    RerankerError,
)


def make_candidates(count=10):
    return [
        {
            "chunk_id": f"chunk-{index}",
            "text": f"candidate text {index}",
            "rank": index + 1,
            "candidate_rank": index + 1,
            "page_number": index + 100,
            "metadata": {"source": "aws-overview.pdf", "page_number": index + 100},
            "score": 1 / (index + 1),
        }
        for index in range(count)
    ]


class FakeCrossEncoder:
    def __init__(self, scores=None):
        self.scores = scores or list(range(10))
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores[:len(pairs)]


def test_reranker_scores_original_query_and_candidate_text_and_reuses_model():
    model = FakeCrossEncoder([0.2, 0.9])
    loads = []
    reranker = CrossEncoderReranker(
        "test/model", model_factory=lambda name: (loads.append(name), model)[1]
    )
    ranked = reranker.rerank("original question", make_candidates(2), top_k=2)
    reranker.rerank("next question", make_candidates(2), top_k=1)

    assert loads == ["test/model"]
    assert model.calls[0][0] == [
        ("original question", "candidate text 0"),
        ("original question", "candidate text 1"),
    ]
    assert ranked[0]["chunk_id"] == "chunk-1"
    assert ranked[0]["reranker_score"] == 0.9


def test_reranker_returns_top_five_from_ten_and_preserves_chunk_metadata():
    model = FakeCrossEncoder(list(range(10)))
    result = CrossEncoderReranker(model=model).rerank("q", make_candidates(), top_k=5)

    assert [item["chunk_id"] for item in result] == [
        "chunk-9", "chunk-8", "chunk-7", "chunk-6", "chunk-5",
    ]
    assert [item["rank"] for item in result] == [1, 2, 3, 4, 5]
    assert result[0]["page_number"] == 109
    assert result[0]["metadata"] == {
        "source": "aws-overview.pdf", "page_number": 109,
    }
    assert result[0]["score"] == 0.1
    assert result[0]["candidate_rank"] == 10


def test_reranker_handles_empty_candidate_list_without_loading_model():
    reranker = CrossEncoderReranker(model_factory=lambda _: pytest.fail("must not load"))
    assert reranker.rerank("q", []) == []


def test_reranker_wraps_model_load_and_prediction_failures():
    broken_load = CrossEncoderReranker(model_factory=lambda _: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(RerankerError, match="Unable to load reranker"):
        broken_load.rerank("q", make_candidates(1))

    class BrokenPrediction:
        def predict(self, *_args, **_kwargs):
            raise ValueError("bad input")

    with pytest.raises(RerankerError, match="Unable to score candidates"):
        CrossEncoderReranker(model=BrokenPrediction()).rerank("q", make_candidates(1))


def test_hybrid_wrapper_retrieves_ten_fuses_then_returns_five_with_metadata():
    class BaseSearch:
        def __init__(self):
            self.calls = []

        def search(self, query, **kwargs):
            self.calls.append((query, kwargs))
            candidates = make_candidates(kwargs["top_k"])
            return {"hybrid": candidates, "vector": candidates, "keyword": candidates,
                    "vector_only": candidates}

    base = BaseSearch()
    search = RerankedHybridSearch(
        base, CrossEncoderReranker(model=FakeCrossEncoder()), candidate_count=10,
        rerank_top_k=5,
    )
    result = search.search("query", top_k=5, vector_weight=0.5, keyword_weight=0.5)

    assert base.calls == [("query", {
        "top_k": 10, "vector_weight": 0.5, "keyword_weight": 0.5,
    })]
    assert len(result["hybrid"]) == 5
    assert result["hybrid"][0]["chunk_id"] == "chunk-9"
    assert result["hybrid"][0]["metadata"]["source"] == "aws-overview.pdf"


def test_duplicate_rrf_chunks_take_one_slot_in_reranker_candidate_pool():
    vector_rows = [
        {"chunk_id": f"vector-{index}", "text": f"vector passage {index}", "metadata": {}}
        for index in range(10)
    ]
    vector_rows.insert(0, {"chunk_id": "shared", "text": "shared passage", "metadata": {}})
    keyword_rows = [
        {"chunk_id": "shared", "text": "shared passage", "metadata": {}},
        {"chunk_id": "keyword-only", "text": "keyword passage", "metadata": {}},
    ]

    class StaticVectorSearch:
        def search(self, _query, *, top_k):
            return vector_rows[:top_k]

    class StaticKeywordSearch:
        def search(self, _query, *, top_k):
            return keyword_rows[:top_k]

    base = object.__new__(HybridSearch)
    base.vector_search = StaticVectorSearch()
    base.keyword_search = StaticKeywordSearch()

    class CapturingCrossEncoder(FakeCrossEncoder):
        def predict(self, pairs, **kwargs):
            self.seen_pairs = pairs
            return [0.0] * len(pairs)

    model = CapturingCrossEncoder()
    search = RerankedHybridSearch(base, CrossEncoderReranker(model=model), candidate_count=10)
    result = search.search("question", top_k=5)

    seen_ids = [pair[1].replace(" passage", "") for pair in model.seen_pairs]
    assert len(model.seen_pairs) == 10
    assert len(result["hybrid"]) == 5
    assert "shared" in seen_ids
    assert seen_ids.count("shared") == 1
    assert len(set(seen_ids)) == 10


def test_disabled_cli_configuration_keeps_existing_hybrid_search(monkeypatch):
    class FakeStore:
        def count(self):
            return 1

    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setattr(cli, "ChromaStore", FakeStore)
    result = cli._build_searcher()

    assert isinstance(result, HybridSearch)


def test_enabled_cli_configuration_builds_lazy_minilm_reranked_search(monkeypatch):
    class FakeStore:
        def count(self):
            return 1

    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RERANKER_MODEL", "test/local-cross-encoder")
    monkeypatch.setenv("RERANKER_CANDIDATE_COUNT", "10")
    monkeypatch.setenv("RERANKER_TOP_K", "5")
    monkeypatch.setattr(cli, "ChromaStore", FakeStore)
    result = cli._build_searcher()

    assert isinstance(result, RerankedHybridSearch)
    assert result.candidate_count == 10
    assert result.rerank_top_k == 5
    assert result.reranker.model_name == "test/local-cross-encoder"
    assert result.reranker._model is None
