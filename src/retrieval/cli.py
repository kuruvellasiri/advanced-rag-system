"""Terminal Q&A and evaluation interface for the AWS RAG project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.dataset import DEFAULT_DATASET_PATH, load_evaluation_dataset
from src.evaluation.evaluator import (
    DEFAULT_RESULTS_PATH,
    RAGEvaluator,
    RetrievalOnlyEvaluator,
    format_evaluation_report,
    save_evaluation_results,
)
from src.evaluation.gemini_evaluator import GeminiAnswerEvaluator
from src.rag.answer_generator import AnswerGenerationError, GeminiAnswerGenerator
from src.rag.gemini_client import ensure_api_key
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RerankedHybridSearch,
)
from src.retrieval.vector_search import VectorSearch
from src.vectorstore.chroma_store import ChromaStore

DEFAULT_RETRIEVAL_ONLY_RESULTS_PATH = Path("data/evaluation/retrieval_only_results.json")


def _print_results(title: str, records: list[dict], *, score_key: str = "score") -> None:
    print(f"\n{title}")
    if not records:
        print("  No matches")
        return
    for item in records:
        page = item.get("page_number") or item.get("metadata", {}).get("page_number", "?")
        score = item.get(score_key)
        print(f"  {item.get('rank', '?'):>2}. page {page} | score={score!s} | {item['chunk_id']}")
        print("     " + item.get("text", "").replace("\n", " ")[:280])


def _retry_notice(status: int | None, attempt: int, delay: float) -> None:
    label = "429 quota exhausted" if status == 429 else f"HTTP {status or 'transient'}"
    print(f"Gemini {label}; retry {attempt} after {delay:.1f}s", flush=True)


def _build_searcher() -> HybridSearch | RerankedHybridSearch:
    load_dotenv(Path(".env"))
    store = ChromaStore()
    if store.count() == 0:
        raise SystemExit("Chroma collection is empty. Run `python -m src.vectorstore.index_chunks` first.")
    searcher = HybridSearch(VectorSearch(store))
    if os.getenv("RERANKER_ENABLED", "false").strip().casefold() not in {"1", "true", "yes", "on"}:
        return searcher
    model_name = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL).strip()
    try:
        candidate_count = int(os.getenv("RERANKER_CANDIDATE_COUNT", "10"))
        rerank_top_k = int(os.getenv("RERANKER_TOP_K", "5"))
    except ValueError as exc:
        raise SystemExit("RERANKER_CANDIDATE_COUNT and RERANKER_TOP_K must be integers") from exc
    if candidate_count < 1 or rerank_top_k < 1:
        raise SystemExit("RERANKER_CANDIDATE_COUNT and RERANKER_TOP_K must be positive")
    return RerankedHybridSearch(
        searcher,
        CrossEncoderReranker(model_name),
        candidate_count=candidate_count,
        rerank_top_k=rerank_top_k,
    )


def _run_evaluation(args: argparse.Namespace, searcher: HybridSearch) -> None:
    dataset = load_evaluation_dataset(args.dataset)
    if args.retrieval_only:
        runner = RetrievalOnlyEvaluator(searcher, dataset)
        mode_label = "retrieval-only"
    else:
        generator = GeminiAnswerGenerator(
            searcher,
            model=args.model,
            top_k=args.top_k,
            retry_callback=_retry_notice,
        )
        answer_evaluator = (
            None if args.no_llm_eval
            else GeminiAnswerEvaluator(model=args.model, retry_callback=_retry_notice)
        )
        runner = RAGEvaluator(generator, dataset, answer_evaluator)
        mode_label = "answer generation" if args.no_llm_eval else "full"

    total = len(dataset["questions"]) if args.limit is None else min(args.limit, len(dataset["questions"]))
    print(f"Evaluating {total} questions ({mode_label}, top_k={args.top_k})...", flush=True)

    def progress(index: int, count: int, row: dict) -> None:
        status = "error" if row.get("retrieval_error") or row.get("generation_error") else "ok"
        print(f"[{index}/{count}] {row['question_id']} {status}", flush=True)

    result = runner.run(
        top_k=args.top_k,
        vector_weight=args.vector_weight,
        keyword_weight=args.keyword_weight,
        limit=args.limit,
        progress_callback=progress,
    )
    output_path = args.output
    if args.retrieval_only and output_path == DEFAULT_RESULTS_PATH:
        output_path = DEFAULT_RETRIEVAL_ONLY_RESULTS_PATH
    output_path = save_evaluation_results(result, output_path)
    print()
    print(format_evaluation_report(result), end="")
    print(f"\nDetailed results saved to: {output_path}")


def _ask(generator: GeminiAnswerGenerator, args: argparse.Namespace, query: str) -> None:
    try:
        result = generator.generate(
            query,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight,
        )
    except AnswerGenerationError as exc:
        print(f"\nUnable to answer this question: {exc}")
        return

    if args.answer_only:
        print(result["answer"])
        return

    print("\nAnswer:\n" + result["answer"])
    if args.compare:
        _print_results("Vector search", result["retrieval"]["vector"])
        _print_results("BM25 keyword search", result["retrieval"]["keyword"])


def _run_interactive(args: argparse.Namespace, searcher: HybridSearch) -> None:
    generator = GeminiAnswerGenerator(
        searcher,
        model=args.model,
        top_k=args.top_k,
        retry_callback=_retry_notice,
    )
    if args.question is not None:
        _ask(generator, args, args.question)
        return

    print("Ask about AWS services or concepts. Enter an empty question to quit.")
    while True:
        try:
            query = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        _ask(generator, args, query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the AWS PDF and run RAG evaluation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--keyword-weight", type=float, default=1.0)
    parser.add_argument("--model", default=None, help="Gemini chat model override")
    parser.add_argument("--question", default=None, help="Ask one question and exit")
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Print only the generated answer, without sources or retrieval details",
    )
    parser.add_argument("--compare", action="store_true", help="Also print vector and BM25 rankings")
    parser.add_argument("--evaluate", action="store_true", help="Run the evaluation dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval metrics without Gemini answer generation or LLM judging",
    )
    parser.add_argument("--no-llm-eval", action="store_true", help="Skip optional LLM answer scoring")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.question is not None and not args.question.strip():
        parser.error("--question must not be empty")

    try:
        ensure_api_key()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    searcher = _build_searcher()
    if args.evaluate:
        _run_evaluation(args, searcher)
    else:
        _run_interactive(args, searcher)


if __name__ == "__main__":
    main()
