import json

import pytest

from src.evaluation.dataset import EvaluationDatasetError, load_evaluation_dataset
from src.evaluation.evaluator import (
    RAGEvaluator,
    RetrievalOnlyEvaluator,
    aggregate_results,
    format_evaluation_report,
    save_evaluation_results,
)
from src.evaluation.gemini_evaluator import AnswerEvaluationError, parse_evaluation_scores
from src.evaluation.metrics import (
    evaluate_retrieval,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics_are_deterministic():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "d"]

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1 / 2)
    assert reciprocal_rank(retrieved, relevant, 3) == pytest.approx(1 / 2)
    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0
    assert hit_rate_at_k(["x", "y"], relevant, 2) == 0.0
    assert evaluate_retrieval(retrieved, relevant, 2) == {
        "precision_at_k": pytest.approx(0.5),
        "recall_at_k": pytest.approx(0.5),
        "mrr": pytest.approx(0.5),
        "hit_rate_at_k": pytest.approx(1.0),
    }


def test_retrieval_metrics_return_none_without_relevant_ids():
    assert precision_at_k(["a"], [], 1) is None
    assert recall_at_k(["a"], [], 1) is None
    assert reciprocal_rank(["a"], [], 1) is None
    assert hit_rate_at_k(["a"], [], 1) is None


def test_default_evaluation_dataset_loads_and_covers_required_categories():
    dataset = load_evaluation_dataset()
    questions = dataset["questions"]

    assert len(questions) >= 15
    assert len({item["id"] for item in questions}) == len(questions)
    assert all(item["reference_answer"] for item in questions)
    assert all(item["relevant_pages"] for item in questions)
    assert all(item["relevant_chunk_ids"] for item in questions)
    categories = {item["category"] for item in questions}
    assert "conceptual" in categories
    assert "exact-service" in categories


def test_dataset_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "dataset.json"
    item = {
        "id": "q01",
        "question": "Question?",
        "reference_answer": "Answer.",
        "relevant_pages": [1],
        "relevant_chunk_ids": ["chunk-1"],
    }
    path.write_text(json.dumps({"questions": [item, item]}), encoding="utf-8")

    with pytest.raises(EvaluationDatasetError, match="Duplicate question id"):
        load_evaluation_dataset(path)


def test_evaluation_results_are_saved_as_json(tmp_path):
    result = {"metadata": {"dataset": "test"}, "questions": [], "aggregate": {"questions": 0}}
    path = save_evaluation_results(result, tmp_path / "nested" / "results.json")

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == result


class FakeAnswerGenerator:
    def generate(self, question, *, top_k=5, vector_weight=1.0, keyword_weight=1.0):
        chunks = [
            {
                "chunk_id": "chunk-a",
                "page_number": 12,
                "metadata": {"page_number": 12},
                "text": "Cloud computing is on-demand delivery.",
                "score": 0.9,
                "rank": 1,
            },
            {
                "chunk_id": "chunk-b",
                "page_number": 13,
                "metadata": {"page_number": 13},
                "text": "Other context.",
                "score": 0.5,
                "rank": 2,
            },
        ][:top_k]
        return {
            "question": question,
            "answer": "Cloud computing is on-demand delivery. Sources: page 12 | chunk-a",
            "model_answer": "Cloud computing is on-demand delivery.",
            "sources": [{"chunk_id": "chunk-a", "page_number": 12, "rank": 1, "score": 0.9}],
            "retrieved": chunks,
            "retrieval": {"hybrid": chunks},
            "top_k": top_k,
            "model": "fake",
        }


class FakeAnswerEvaluator:
    def evaluate(self, *, question, answer, contexts, reference_answer):
        return {
            "answer_relevance": 0.9,
            "faithfulness": 1.0,
            "context_relevance": 0.8,
            "correctness": 0.7,
            "reasoning": "Supported by the retrieved context.",
        }


def test_evaluator_runs_pipeline_and_aggregates_metrics():
    dataset = {
        "name": "test",
        "source": "test.pdf",
        "questions": [
            {
                "id": "q01",
                "category": "conceptual",
                "question": "What is cloud computing?",
                "reference_answer": "On-demand delivery.",
                "relevant_pages": [12],
                "relevant_chunk_ids": ["chunk-a"],
            }
        ],
    }
    runner = RAGEvaluator(FakeAnswerGenerator(), dataset, FakeAnswerEvaluator())
    result = runner.run(top_k=2)

    row = result["questions"][0]
    assert row["generation_error"] is None
    assert row["retrieved_chunk_ids"] == ["chunk-a", "chunk-b"]
    assert row["retrieval_metrics"]["precision_at_k"] == pytest.approx(0.5)
    assert row["retrieval_metrics"]["recall_at_k"] == pytest.approx(1.0)
    assert row["retrieval_metrics"]["mrr"] == pytest.approx(1.0)
    assert row["answer_metrics"]["faithfulness"] == 1.0

    aggregate = result["aggregate"]
    assert aggregate["questions"] == 1
    assert aggregate["retrieval_questions"] == 1
    assert aggregate["answer_questions"] == 1
    assert aggregate["retrieval"]["precision_at_k"] == pytest.approx(0.5)
    assert aggregate["answer_estimates"]["correctness"] == pytest.approx(0.7)
    assert "deterministic" in aggregate["notes"]
    assert "estimate" in aggregate["notes"]
    assert "Aggregate Metrics" in format_evaluation_report(result)


def test_evaluator_records_generation_errors_without_crashing():
    class FailingGenerator:
        def generate(self, *args, **kwargs):
            raise RuntimeError("Gemini unavailable")

    dataset = {
        "name": "test",
        "questions": [
            {
                "id": "q01",
                "question": "What is AWS?",
                "reference_answer": "A cloud provider.",
                "relevant_pages": [1],
                "relevant_chunk_ids": ["chunk-a"],
            }
        ],
    }
    result = RAGEvaluator(FailingGenerator(), dataset).run(top_k=1)

    assert result["questions"][0]["generation_error"] == "RuntimeError: Gemini unavailable"
    assert result["aggregate"]["generation_errors"] == 1
    assert result["aggregate"]["retrieval_questions"] == 0


def test_retrieval_only_evaluator_skips_answer_generation_and_judging():
    class FakeRetriever:
        def __init__(self):
            self.calls = 0

        def search(self, question, *, top_k=5, vector_weight=1.0, keyword_weight=1.0):
            self.calls += 1
            chunks = [
                {
                    "chunk_id": "chunk-a",
                    "page_number": 12,
                    "metadata": {"page_number": 12},
                    "text": "Cloud computing is on-demand delivery.",
                    "score": 0.9,
                    "rank": 1,
                },
                {
                    "chunk_id": "chunk-b",
                    "page_number": 13,
                    "metadata": {"page_number": 13},
                    "text": "Other context.",
                    "score": 0.5,
                    "rank": 2,
                },
            ][:top_k]
            return {"hybrid": chunks, "vector": chunks, "keyword": chunks}

    dataset = {
        "name": "retrieval-only-test",
        "questions": [
            {
                "id": "q01",
                "category": "conceptual",
                "question": "What is cloud computing?",
                "reference_answer": "On-demand delivery.",
                "relevant_pages": [12],
                "relevant_chunk_ids": ["chunk-a"],
            }
        ],
    }
    retriever = FakeRetriever()
    result = RetrievalOnlyEvaluator(retriever, dataset).run(top_k=2)
    row = result["questions"][0]

    assert retriever.calls == 1
    assert row["retrieved_chunk_ids"] == ["chunk-a", "chunk-b"]
    assert row["retrieved_pages"] == [12, 13]
    assert row["retrieval_metrics"]["precision_at_k"] == pytest.approx(0.5)
    assert row["retrieval_metrics"]["recall_at_k"] == pytest.approx(1.0)
    assert row["retrieval_metrics"]["mrr"] == pytest.approx(1.0)
    assert row["answer"] is None
    assert row["model_answer"] is None
    assert row["sources"] == []
    assert row["answer_metrics"] is None
    assert row["generation_error"] is None
    assert row["retrieval_error"] is None

    assert result["metadata"]["mode"] == "retrieval_only"
    assert result["metadata"]["answer_generation"] is False
    assert result["metadata"]["llm_answer_evaluation"] is False
    assert result["metadata"]["answer_metrics_type"] == "disabled"
    assert result["aggregate"]["retrieval_questions"] == 1
    assert result["aggregate"]["retrieval_errors"] == 0
    assert result["aggregate"]["answer_questions"] == 0

    report = format_evaluation_report(result)
    assert "Mode: retrieval_only" in report
    assert "Answer generation: disabled" in report
    assert "Answer:" not in report


def test_retrieval_only_evaluator_records_retrieval_errors():
    class FailingRetriever:
        def search(self, *args, **kwargs):
            raise RuntimeError("query embedding unavailable")

    dataset = {
        "name": "retrieval-only-test",
        "questions": [
            {
                "id": "q01",
                "question": "What is AWS?",
                "reference_answer": "A cloud provider.",
                "relevant_pages": [1],
                "relevant_chunk_ids": ["chunk-a"],
            }
        ],
    }
    result = RetrievalOnlyEvaluator(FailingRetriever(), dataset).run(top_k=1)
    row = result["questions"][0]

    assert row["retrieval_error"] == "RuntimeError: query embedding unavailable"
    assert row["retrieval_metrics"] is None
    assert result["aggregate"]["retrieval_questions"] == 0
    assert result["aggregate"]["retrieval_errors"] == 1
    assert "Retrieval error: RuntimeError: query embedding unavailable" in format_evaluation_report(result)


def test_parse_evaluation_scores_accepts_json_and_rejects_invalid_values():
    scores = parse_evaluation_scores(
        '{"answer_relevance": 0.9, "faithfulness": 1, '
        '"context_relevance": 0.8, "correctness": 0.7, "reasoning": "ok"}'
    )
    assert scores["faithfulness"] == 1.0
    assert scores["reasoning"] == "ok"

    with pytest.raises(AnswerEvaluationError, match="between 0 and 1"):
        parse_evaluation_scores(
            '{"answer_relevance": 2, "faithfulness": 1, '
            '"context_relevance": 0.8, "correctness": 0.7}'
        )


def test_aggregate_results_ignores_missing_metrics():
    results = [
        {"retrieval_metrics": None, "answer_metrics": None},
        {
            "retrieval_metrics": {
                "precision_at_k": 1.0,
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "hit_rate_at_k": 1.0,
            },
            "answer_metrics": None,
        },
    ]
    aggregate = aggregate_results(results)
    assert aggregate["retrieval_questions"] == 1
    assert aggregate["answer_questions"] == 0
    assert aggregate["retrieval"]["mrr"] == 1.0
    assert aggregate["answer_estimates"]["faithfulness"] is None
