"""Load and validate the AWS RAG evaluation dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_PATH = Path("data/evaluation/aws_overview_eval.json")


class EvaluationDatasetError(ValueError):
    """The evaluation dataset is missing required fields or is malformed."""


def load_evaluation_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    """Load a JSON dataset and validate every question record."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise EvaluationDatasetError(f"Evaluation dataset not found: {dataset_path}")
    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError(f"Invalid JSON in {dataset_path}: {exc}") from exc

    if isinstance(data, list):
        data = {"name": dataset_path.stem, "questions": data}
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise EvaluationDatasetError("Dataset must contain a 'questions' list")

    questions = data["questions"]
    if not questions:
        raise EvaluationDatasetError("Dataset must contain at least one question")

    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise EvaluationDatasetError(f"Question {index} must be an object")
        question_id = str(item.get("id") or f"q{index:02d}")
        question = str(item.get("question") or "").strip()
        reference = str(item.get("reference_answer") or "").strip()
        if not question:
            raise EvaluationDatasetError(f"Question {index} is missing 'question'")
        if not reference:
            raise EvaluationDatasetError(f"Question {index} is missing 'reference_answer'")
        if question_id in seen_ids:
            raise EvaluationDatasetError(f"Duplicate question id: {question_id}")
        seen_ids.add(question_id)

        pages = item.get("relevant_pages") or []
        chunk_ids = item.get("relevant_chunk_ids") or []
        if not isinstance(pages, list) or not isinstance(chunk_ids, list):
            raise EvaluationDatasetError(
                f"Question {question_id} must use lists for relevant_pages and relevant_chunk_ids"
            )

        normalized.append({
            **item,
            "id": question_id,
            "question": question,
            "reference_answer": reference,
            "relevant_pages": [int(page) for page in pages],
            "relevant_chunk_ids": [str(chunk_id) for chunk_id in chunk_ids],
            "category": str(item.get("category") or "general"),
        })

    return {**data, "questions": normalized}
