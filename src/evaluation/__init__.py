"""Retrieval and answer evaluation for the AWS RAG project."""

from .dataset import DEFAULT_DATASET_PATH, EvaluationDatasetError, load_evaluation_dataset
from .evaluator import (
    DEFAULT_RESULTS_PATH,
    RAGEvaluator,
    RetrievalOnlyEvaluator,
    aggregate_results,
    format_evaluation_report,
    save_evaluation_results,
)
from .gemini_evaluator import (
    AnswerEvaluationError,
    GeminiAnswerEvaluator,
    parse_evaluation_scores,
)
from .metrics import (
    evaluate_retrieval,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "AnswerEvaluationError",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_RESULTS_PATH",
    "EvaluationDatasetError",
    "GeminiAnswerEvaluator",
    "RAGEvaluator",
    "RetrievalOnlyEvaluator",
    "aggregate_results",
    "evaluate_retrieval",
    "format_evaluation_report",
    "hit_rate_at_k",
    "load_evaluation_dataset",
    "parse_evaluation_scores",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "save_evaluation_results",
]
