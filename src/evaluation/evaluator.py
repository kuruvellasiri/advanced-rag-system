"""Run the RAG evaluation dataset through retrieval and answer generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.rag.prompts import chunk_page_number

from .metrics import evaluate_retrieval

DEFAULT_RESULTS_PATH = Path("data/evaluation/latest_results.json")
RETRIEVAL_KEYS = ("precision_at_k", "recall_at_k", "mrr", "hit_rate_at_k")
ANSWER_KEYS = ("answer_relevance", "faithfulness", "context_relevance", "correctness")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate deterministic retrieval and optional LLM-estimated metrics."""
    retrieval_questions = sum(
        1 for item in results
        if item.get("retrieval_metrics") is not None
        and any(value is not None for value in item["retrieval_metrics"].values())
    )
    answer_questions = sum(1 for item in results if item.get("answer_metrics") is not None)

    retrieval = {}
    for key in RETRIEVAL_KEYS:
        values = [
            float(item["retrieval_metrics"][key])
            for item in results
            if item.get("retrieval_metrics") and item["retrieval_metrics"].get(key) is not None
        ]
        retrieval[key] = _mean(values)

    answer = {}
    for key in ANSWER_KEYS:
        values = [
            float(item["answer_metrics"][key])
            for item in results
            if item.get("answer_metrics") and item["answer_metrics"].get(key) is not None
        ]
        answer[key] = _mean(values)

    return {
        "questions": len(results),
        "retrieval_questions": retrieval_questions,
        "answer_questions": answer_questions,
        "generation_errors": sum(1 for item in results if item.get("generation_error")),
        "retrieval_errors": sum(1 for item in results if item.get("retrieval_error")),
        "answer_evaluation_errors": sum(
            1 for item in results if item.get("answer_evaluation_error")
        ),
        "retrieval": retrieval,
        "answer_estimates": answer,
        "notes": (
            "Retrieval metrics are deterministic. Answer metrics are optional "
            "Gemini-generated estimates, not guaranteed factual judgments."
        ),
    }


def save_evaluation_results(
    result: dict[str, Any],
    path: str | Path = DEFAULT_RESULTS_PATH,
) -> Path:
    """Save detailed evaluation results as UTF-8 JSON and return the path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _format_metric(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _base_result_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": item["id"],
        "category": item.get("category"),
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "relevant_pages": item.get("relevant_pages", []),
        "relevant_chunk_ids": item.get("relevant_chunk_ids", []),
        "answer": None,
        "model_answer": None,
        "sources": [],
        "retrieved_chunk_ids": [],
        "retrieved_pages": [],
        "retrieval_metrics": None,
        "answer_metrics": None,
        "generation_error": None,
        "retrieval_error": None,
        "answer_evaluation_error": None,
    }


class RAGEvaluator:
    """Execute all dataset questions through the existing RAG pipeline."""

    def __init__(
        self,
        answer_generator: Any,
        dataset: dict[str, Any],
        answer_evaluator: Any | None = None,
    ) -> None:
        self.answer_generator = answer_generator
        self.dataset = dataset
        self.answer_evaluator = answer_evaluator

    def run(
        self,
        *,
        top_k: int = 5,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        limit: int | None = None,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run the dataset and return detailed plus aggregate results."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        questions = self.dataset["questions"]
        if limit is not None:
            questions = questions[:limit]

        result_rows: list[dict[str, Any]] = []
        for index, item in enumerate(questions, start=1):
            row = _base_result_row(item)
            try:
                generated = self.answer_generator.generate(
                    item["question"],
                    top_k=top_k,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                )
                retrieved = list(generated.get("retrieved") or [])
                retrieved_ids = [str(chunk.get("chunk_id")) for chunk in retrieved]
                row.update({
                    "answer": generated.get("answer"),
                    "model_answer": generated.get("model_answer"),
                    "sources": generated.get("sources", []),
                    "retrieved_chunk_ids": retrieved_ids,
                    "retrieved_pages": [chunk_page_number(chunk) for chunk in retrieved],
                    "retrieval_metrics": evaluate_retrieval(
                        retrieved_ids,
                        item.get("relevant_chunk_ids", []),
                        top_k,
                    ),
                })
                if self.answer_evaluator is not None and generated.get("model_answer"):
                    try:
                        row["answer_metrics"] = self.answer_evaluator.evaluate(
                            question=item["question"],
                            answer=generated["model_answer"],
                            contexts=retrieved,
                            reference_answer=item["reference_answer"],
                        )
                    except Exception as exc:
                        row["answer_evaluation_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                row["generation_error"] = f"{type(exc).__name__}: {exc}"
            result_rows.append(row)
            if progress_callback:
                progress_callback(index, len(questions), row)

        return {
            "metadata": {
                "dataset": self.dataset.get("name", "evaluation"),
                "source": self.dataset.get("source"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "top_k": top_k,
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight,
                "limited_to": limit,
                "mode": "full" if self.answer_evaluator is not None else "answer_generation_only",
                "answer_generation": True,
                "llm_answer_evaluation": self.answer_evaluator is not None,
                "retrieval_metrics_type": "deterministic",
                "answer_metrics_type": (
                    "optional_gemini_estimate" if self.answer_evaluator is not None else "disabled"
                ),
            },
            "questions": result_rows,
            "aggregate": aggregate_results(result_rows),
        }


class RetrievalOnlyEvaluator:
    """Evaluate deterministic retrieval without answer generation or judging."""

    def __init__(self, retriever: Any, dataset: dict[str, Any]) -> None:
        self.retriever = retriever
        self.dataset = dataset

    def run(
        self,
        *,
        top_k: int = 5,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        limit: int | None = None,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run retrieval and metrics only; no Gemini chat or judge calls."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        questions = self.dataset["questions"]
        if limit is not None:
            questions = questions[:limit]

        result_rows: list[dict[str, Any]] = []
        for index, item in enumerate(questions, start=1):
            row = _base_result_row(item)
            try:
                retrieval = self.retriever.search(
                    item["question"],
                    top_k=top_k,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                )
                retrieved = list(retrieval.get("hybrid") or [])[:top_k]
                retrieved_ids = [str(chunk.get("chunk_id")) for chunk in retrieved]
                row.update({
                    "retrieved_chunk_ids": retrieved_ids,
                    "retrieved_pages": [chunk_page_number(chunk) for chunk in retrieved],
                    "retrieval_metrics": evaluate_retrieval(
                        retrieved_ids,
                        item.get("relevant_chunk_ids", []),
                        top_k,
                    ),
                })
            except Exception as exc:
                row["retrieval_error"] = f"{type(exc).__name__}: {exc}"
            result_rows.append(row)
            if progress_callback:
                progress_callback(index, len(questions), row)

        return {
            "metadata": {
                "dataset": self.dataset.get("name", "evaluation"),
                "source": self.dataset.get("source"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "top_k": top_k,
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight,
                "limited_to": limit,
                "mode": "retrieval_only",
                "answer_generation": False,
                "llm_answer_evaluation": False,
                "retrieval_metrics_type": "deterministic",
                "answer_metrics_type": "disabled",
            },
            "questions": result_rows,
            "aggregate": aggregate_results(result_rows),
        }


def format_evaluation_report(result: dict[str, Any], *, answer_preview: int = 360) -> str:
    """Return a readable terminal report for evaluation results."""
    lines: list[str] = []
    metadata = result.get("metadata", {})
    lines.append("RAG Evaluation Report")
    lines.append("=" * 72)
    lines.append(f"Dataset: {metadata.get('dataset')}")
    lines.append(f"Mode: {metadata.get('mode', 'full')}")
    lines.append(f"Top K: {metadata.get('top_k')}")
    lines.append(
        "Answer metrics: optional Gemini estimates"
        if metadata.get("llm_answer_evaluation")
        else "Answer metrics: disabled"
    )
    if not metadata.get("answer_generation", True):
        lines.append("Answer generation: disabled")
    lines.append("")

    for index, row in enumerate(result.get("questions", []), start=1):
        lines.append(f"[{index}] {row['question_id']} ({row.get('category') or 'general'})")
        lines.append(f"Q: {row['question']}")
        lines.append(
            f"Expected pages: {row.get('relevant_pages')} | "
            f"Retrieved pages: {row.get('retrieved_pages')}"
        )
        metrics = row.get("retrieval_metrics") or {}
        lines.append(
            "Retrieval: "
            f"P@K={_format_metric(metrics.get('precision_at_k'))} "
            f"R@K={_format_metric(metrics.get('recall_at_k'))} "
            f"MRR={_format_metric(metrics.get('mrr'))} "
            f"Hit@K={_format_metric(metrics.get('hit_rate_at_k'))}"
        )
        if row.get("retrieval_error"):
            lines.append(f"Retrieval error: {row['retrieval_error']}")
        if row.get("generation_error"):
            lines.append(f"Generation error: {row['generation_error']}")
        elif metadata.get("answer_generation", True):
            answer = (row.get("answer") or "").replace("\n", " ")
            if len(answer) > answer_preview:
                answer = answer[:answer_preview] + "..."
            lines.append(f"Answer: {answer}")
        if row.get("sources"):
            source_text = ", ".join(
                f"page {source['page_number']} | {source['chunk_id']}"
                for source in row["sources"]
            )
            lines.append(f"Sources: {source_text}")
        if row.get("answer_metrics"):
            scores = row["answer_metrics"]
            lines.append(
                "Answer estimates: "
                f"relevance={_format_metric(scores.get('answer_relevance'))} "
                f"faithfulness={_format_metric(scores.get('faithfulness'))} "
                f"context={_format_metric(scores.get('context_relevance'))} "
                f"correctness={_format_metric(scores.get('correctness'))}"
            )
        if row.get("answer_evaluation_error"):
            lines.append(f"Answer evaluation error: {row['answer_evaluation_error']}")
        lines.append("")

    aggregate = result.get("aggregate", {})
    retrieval = aggregate.get("retrieval", {})
    answer = aggregate.get("answer_estimates", {})
    lines.append("Aggregate Metrics")
    lines.append("-" * 72)
    lines.append(
        f"Retrieval questions: {aggregate.get('retrieval_questions')} | "
        f"Answer-evaluated questions: {aggregate.get('answer_questions')} | "
        f"Retrieval errors: {aggregate.get('retrieval_errors', 0)} | "
        f"Generation errors: {aggregate.get('generation_errors')} | "
        f"Answer-evaluation errors: {aggregate.get('answer_evaluation_errors')}"
    )
    lines.append(
        "Retrieval: "
        f"P@K={_format_metric(retrieval.get('precision_at_k'))} "
        f"R@K={_format_metric(retrieval.get('recall_at_k'))} "
        f"MRR={_format_metric(retrieval.get('mrr'))} "
        f"Hit@K={_format_metric(retrieval.get('hit_rate_at_k'))}"
    )
    lines.append(
        "Answer estimates: "
        f"relevance={_format_metric(answer.get('answer_relevance'))} "
        f"faithfulness={_format_metric(answer.get('faithfulness'))} "
        f"context={_format_metric(answer.get('context_relevance'))} "
        f"correctness={_format_metric(answer.get('correctness'))}"
    )
    lines.append(str(aggregate.get("notes", "")))
    return "\n".join(lines).rstrip() + "\n"
