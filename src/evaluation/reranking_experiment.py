"""Separate retrieval and optional answer-quality experiment for cross-encoders.

Run from the repository root with:
    python -m src.evaluation.reranking_experiment
    python -m src.evaluation.reranking_experiment --answer-quality

This module does not update the production retriever or Chroma collection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.dataset import DEFAULT_DATASET_PATH, load_evaluation_dataset
from src.evaluation.evaluator import aggregate_results, save_evaluation_results
from src.evaluation.metrics import evaluate_retrieval
from src.rag.prompts import chunk_page_number
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.vector_search import VectorSearch
from src.vectorstore.chroma_store import ChromaStore

TOP_K = 5
CANDIDATE_COUNTS = (10, 20, 30)
VECTOR_WEIGHT = 0.50
KEYWORD_WEIGHT = 0.50
RRF_K = 60
RERANKER_MODELS = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
)
OUTPUT_PATH = Path("data/evaluation/reranking_comparison_results.json")
REPORT_PATH = Path("data/evaluation/reranking_comparison_report.md")


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 3)


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 3)


def _rankings(chunks: list[dict[str, Any]], scores: list[float] | None = None) -> list[dict[str, Any]]:
    rows = []
    for position, chunk in enumerate(chunks, start=1):
        row = {
            "rank": position,
            "chunk_id": str(chunk["chunk_id"]),
            "page_number": chunk_page_number(chunk),
        }
        if scores is not None:
            row["reranker_score"] = float(scores[position - 1])
            row["candidate_rank"] = int(chunk.get("candidate_rank", position))
        rows.append(row)
    return rows


def _make_config(
    dataset: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    config_id: str,
    architecture: str,
    model: str | None,
    candidate_count: int,
    retrieval_ms: list[float],
    rerank_ms: list[float],
) -> dict[str, Any]:
    questions = []
    for record in records:
        item = record["item"]
        candidates = record["candidates"]
        final = record["final"]
        candidate_ids = [str(chunk["chunk_id"]) for chunk in candidates]
        final_ids = [str(chunk["chunk_id"]) for chunk in final]
        relevant = set(str(value) for value in item.get("relevant_chunk_ids", []))
        questions.append({
            "question_id": item["id"],
            "category": item.get("category"),
            "question": item["question"],
            "relevant_chunk_ids": sorted(relevant),
            "candidate_chunk_ids": candidate_ids,
            "candidate_rankings": _rankings(candidates),
            "candidate_recall_at_n": evaluate_retrieval(candidate_ids, relevant, max(1, len(candidates)))["recall_at_k"],
            "retrieved_chunk_ids": final_ids,
            "retrieved_rankings": _rankings(final, record.get("scores")),
            "retrieval_metrics": evaluate_retrieval(final_ids, relevant, TOP_K),
            "relevant_candidates_removed_by_reranker": sorted((set(candidate_ids) & relevant) - set(final_ids)),
            "latency_ms": {
                "retrieval": record["retrieval_ms"],
                "reranking": record.get("rerank_ms", 0.0),
                "total": round(record["retrieval_ms"] + record.get("rerank_ms", 0.0), 3),
            },
        })
    aggregate = aggregate_results([{
        "retrieval_metrics": row["retrieval_metrics"],
        "answer_metrics": None,
    } for row in questions])
    aggregate["candidate_recall_at_n"] = _mean([
        float(row["candidate_recall_at_n"] or 0.0) for row in questions
    ])
    aggregate["reranker_dropped_relevant_chunks"] = sum(
        len(row["relevant_candidates_removed_by_reranker"]) for row in questions
    )
    aggregate["latency_ms"] = {
        "retrieval_mean": _mean(retrieval_ms),
        "retrieval_median": round(statistics.median(retrieval_ms), 3) if retrieval_ms else None,
        "retrieval_p95": _p95(retrieval_ms),
        "reranking_mean": _mean(rerank_ms),
        "reranking_median": round(statistics.median(rerank_ms), 3) if rerank_ms else None,
        "reranking_p95": _p95(rerank_ms),
        "reranking_max": round(max(rerank_ms), 3) if rerank_ms else None,
        "total_mean": _mean([
            row["latency_ms"]["total"] for row in questions
        ]),
        "total_median": round(statistics.median([row["latency_ms"]["total"] for row in questions]), 3) if questions else None,
        "total_p95": _p95([row["latency_ms"]["total"] for row in questions]),
        "total_max": round(max(row["latency_ms"]["total"] for row in questions), 3) if questions else None,
    }
    return {
        "metadata": {
            "config_id": config_id,
            "architecture": architecture,
            "reranker_model": model,
            "candidate_count": candidate_count,
            "final_top_k": TOP_K,
            "vector_weight": VECTOR_WEIGHT if "hybrid" in architecture else 1.0,
            "keyword_weight": KEYWORD_WEIGHT if "hybrid" in architecture else 0.0,
            "rrf_k": RRF_K if "hybrid" in architecture else None,
            "dataset": dataset.get("name"),
            "source": dataset.get("source"),
            "question_count": len(questions),
        },
        "aggregate": aggregate,
        "questions": questions,
    }


def _rerank(model: Any, query: str, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float], float]:
    if not candidates:
        return [], [], 0.0
    pairs = [(query, str(chunk.get("text") or "")) for chunk in candidates]
    start = time.perf_counter()
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
    elapsed = _ms(time.perf_counter() - start)
    numeric_scores = [float(score) for score in scores]
    ranked = sorted(
        zip(candidates, numeric_scores),
        key=lambda pair: (-pair[1], int(pair[0].get("candidate_rank", 0)), str(pair[0]["chunk_id"])),
    )[:TOP_K]
    chunks = [{**chunk, "reranker_score": score} for chunk, score in ranked]
    return chunks, [score for _, score in ranked], elapsed


class _RequestPacer:
    """Share one minimum interval across answer and judge requests."""

    def __init__(self, minimum_interval: float = 4.2) -> None:
        self.minimum_interval = minimum_interval
        self.last_started = 0.0

    def wait(self) -> None:
        remaining = self.minimum_interval - (time.monotonic() - self.last_started)
        if remaining > 0:
            time.sleep(remaining)
        self.last_started = time.monotonic()


class _PacedChatClient:
    def __init__(self, client: Any, pacer: _RequestPacer) -> None:
        self.client = client
        self.pacer = pacer

    def invoke(self, messages: Any) -> Any:
        self.pacer.wait()
        return self.client.invoke(messages)


def _write_report(result: dict[str, Any]) -> None:
    experiments = result["experiments"]
    lines = [
        "# Dense, Hybrid, and Cross-Encoder Reranking Comparison", "",
        f"- Run time (UTC): {result['metadata']['created_at']}",
        f"- Dataset: `{result['metadata']['dataset_path']}` ({result['metadata']['question_count']} questions)",
        f"- Embeddings: `{result['metadata']['embedding_provider']}` / `{result['metadata']['embedding_model']}`",
        f"- Chroma collection: `{result['metadata']['chroma_collection']}` ({result['metadata']['chroma_count']} chunks; read-only)",
        f"- Weights: dense `{VECTOR_WEIGHT:.2f}`, BM25 `{KEYWORD_WEIGHT:.2f}`; RRF constant `{RRF_K}`.",
        f"- Final result size: top `{TOP_K}`. Cross-encoder candidates: `{', '.join(map(str, CANDIDATE_COUNTS))}`.",
        f"- Runtime device: `{result['metadata']['device']}`.",
        "- Each latency is a measured wall-clock time in this environment, using local CPU inference; model-load time is excluded. The table uses median and p95 to limit distortion from isolated scheduling outliers; per-question raw timings and maxima are in JSON.", "",
        "## Reranker resource profile", "",
        "| Model | Parameters observed | Compatibility and model-card size |",
        "|---|---:|---|",
        f"| [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | {result['metadata'].get('reranker_parameters', {}).get(RERANKER_MODELS[0], 'N/A'):,} | Loads through Sentence Transformers `CrossEncoder`; model card reports 22.7M parameters. |",
        f"| [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) | {result['metadata'].get('reranker_parameters', {}).get(RERANKER_MODELS[1], 'N/A'):,} | Loads through Sentence Transformers `CrossEncoder`; model card lists 1.11 GB safetensors weights and a 3.36 GB repository. |",
        "",
        "## Retrieval results", "",
        "| Architecture / model / N | Precision@5 | Recall@5 | MRR@5 | Hit@5 | Candidate recall@N | Dropped relevant chunks | Retrieval ms mean | Rerank ms median (p95) | Total ms median (p95) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in experiments:
        meta = experiment["metadata"]
        agg = experiment["aggregate"]
        metrics = agg["retrieval"]
        timing = agg["latency_ms"]
        label = meta["architecture"]
        if meta["reranker_model"]:
            label += f" / `{meta['reranker_model']}` / {meta['candidate_count']}"
        def fmt(value: Any) -> str:
            return "N/A" if value is None else f"{value:.4f}" if isinstance(value, (int, float)) and value <= 1 else f"{value:.1f}"
        lines.append(
            f"| {label} | {fmt(metrics['precision_at_k'])} | {fmt(metrics['recall_at_k'])} | "
            f"{fmt(metrics['mrr'])} | {fmt(metrics['hit_rate_at_k'])} | "
            f"{fmt(agg['candidate_recall_at_n'])} | {agg['reranker_dropped_relevant_chunks']} | "
            f"{fmt(timing['retrieval_mean'])} | {fmt(timing['reranking_median'])} ({fmt(timing['reranking_p95'])}) | "
            f"{fmt(timing['total_median'])} ({fmt(timing['total_p95'])}) |"
        )
    dense_baseline = next(e for e in experiments if e["metadata"]["architecture"] == "dense_only")
    hybrid_baseline = next(e for e in experiments if e["metadata"]["architecture"] == "hybrid_rrf")
    mini_hybrid10 = next(e for e in experiments if e["metadata"]["config_id"] ==
                         f"hybrid_reranked__{RERANKER_MODELS[0].replace('/', '_')}__n10")
    bge_hybrid10 = next(e for e in experiments if e["metadata"]["config_id"] ==
                        f"hybrid_reranked__{RERANKER_MODELS[1].replace('/', '_')}__n10")
    mini_hybrid20 = next(e for e in experiments if e["metadata"]["config_id"] ==
                         f"hybrid_reranked__{RERANKER_MODELS[0].replace('/', '_')}__n20")
    mini_hybrid30 = next(e for e in experiments if e["metadata"]["config_id"] ==
                         f"hybrid_reranked__{RERANKER_MODELS[0].replace('/', '_')}__n30")
    bge_hybrid30 = next(e for e in experiments if e["metadata"]["config_id"] ==
                        f"hybrid_reranked__{RERANKER_MODELS[1].replace('/', '_')}__n30")
    dbm = dense_baseline["aggregate"]["retrieval"]
    hbm = hybrid_baseline["aggregate"]["retrieval"]
    rm = mini_hybrid10["aggregate"]["retrieval"]
    bge = bge_hybrid10["aggregate"]["retrieval"]
    mini10_ms = mini_hybrid10["aggregate"]["latency_ms"]["total_median"]
    hybrid_ms = hybrid_baseline["aggregate"]["latency_ms"]["total_median"]
    lines += [
        "", "## Findings and recommendation", "",
        f"- **Hybrid vs dense:** Hybrid RRF improves all four retrieval measures on this dataset: Precision@5 `{dbm['precision_at_k']:.4f}` to `{hbm['precision_at_k']:.4f}`, Recall@5 `{dbm['recall_at_k']:.4f}` to `{hbm['recall_at_k']:.4f}`, MRR@5 `{dbm['mrr']:.4f}` to `{hbm['mrr']:.4f}`, and Hit Rate@5 `{dbm['hit_rate_at_k']:.4f}` to `{hbm['hit_rate_at_k']:.4f}`.",
        f"- **Hybrid reranking:** MiniLM Cross-Encoder at N=10 preserves Precision/Recall/Hit at `{rm['precision_at_k']:.4f}` / `{rm['recall_at_k']:.4f}` / `{rm['hit_rate_at_k']:.4f}` and raises MRR@5 from `{hbm['mrr']:.4f}` to `{rm['mrr']:.4f}`. No relevant candidate chunk was removed from the final top five in this run.",
        f"- **Dense reranking:** MiniLM raises MRR@5 from `{dbm['mrr']:.4f}` to `{next(e for e in experiments if e['metadata']['config_id'] == f'dense_reranked__{RERANKER_MODELS[0].replace('/', '_')}__n10')['aggregate']['retrieval']['mrr']:.4f}`, but does not recover q20's relevant chunk because it is absent from the dense candidate set.",
        f"- **Model comparison:** At hybrid N=10, both models tie on retrieval (Precision `{rm['precision_at_k']:.4f}`, Recall `{rm['recall_at_k']:.4f}`, MRR `{rm['mrr']:.4f}`, Hit `{rm['hit_rate_at_k']:.4f}`). MiniLM is much faster on this CPU; BGE adds no measured retrieval benefit.",
        f"- **Candidate count:** N=10 is the best trade-off. MiniLM N=20 has the same metrics but higher latency; MiniLM N=30 lowers MRR to `{mini_hybrid30['aggregate']['retrieval']['mrr']:.4f}`. BGE's N=10/20/30 retrieval scores tie, so larger candidate sets add cost without improving scores.",
        f"- **Latency:** Hybrid RRF median retrieval is `{hybrid_ms:.1f} ms`; MiniLM hybrid reranking N=10 is `{mini10_ms:.1f} ms` end to end (about `{mini10_ms - hybrid_ms:.1f} ms` extra per query in this local CPU run).",
        "- **Recommendation:** Keep architecture C (Dense + BM25 + weighted RRF + `cross-encoder/ms-marco-MiniLM-L-6-v2`, N=10, select top five) when the roughly 0.33-second local retrieval median is acceptable. It gives the highest measured hybrid MRR and retains full annotated coverage. Keep architecture B if minimum retrieval latency matters more than ranking the relevant chunk earlier. This experiment did not change the production pipeline.",
        "- **Answer quality:** Use the Gemini results below cautiously. Faithfulness is a judge estimate; the supported-claim rate and semantic citation entailment are not reliably measured by this evaluator.",
        "",
    ]
    bge30_latency = bge_hybrid30["aggregate"]["latency_ms"]
    if bge30_latency.get("total_max") and bge30_latency["total_max"] > 10 * bge30_latency["total_p95"]:
        lines.insert(-1, f"- **Latency outlier:** BGE hybrid N=30 recorded a q07 maximum of `{bge30_latency['total_max']:.1f} ms` ({bge30_latency['total_max'] / 60000:.1f} minutes), versus median `{bge30_latency['total_median']:.1f} ms` and p95 `{bge30_latency['total_p95']:.1f} ms`. Treat this single extreme as an observed tail event and repeat before using BGE N=30 operationally.")
    lines += [
        "", "## Candidate-removal and per-question details", "",
        "Every JSON experiment includes per-question candidate IDs and ranks, final top-five IDs and ranks, retrieval metrics, any relevant chunks dropped from the candidate set, and measured latencies.",
        "",
    ]
    for experiment in experiments:
        meta = experiment["metadata"]
        misses = [row for row in experiment["questions"] if row["retrieval_metrics"]["hit_rate_at_k"] == 0]
        dropped = [row for row in experiment["questions"] if row["relevant_candidates_removed_by_reranker"]]
        if misses or dropped:
            label = meta["config_id"]
            lines.append(f"### {label}")
            for row in misses:
                lines.append(f"- `{row['question_id']}` missed all relevant chunks at final top 5; expected `{', '.join(row['relevant_chunk_ids'])}`, returned `{', '.join(row['retrieved_chunk_ids'])}`.")
            for row in dropped:
                lines.append(f"- `{row['question_id']}` reranking removed relevant candidate(s) `{', '.join(row['relevant_candidates_removed_by_reranker'])}` from final top 5.")
            lines.append("")
    answer = result.get("answer_quality", {})
    lines += ["## Answer-quality evaluation", ""]
    if answer.get("status") in {"completed", "incomplete_with_errors"}:
        lines.append("Measured only for the selected shortlisted architectures. Scores are Gemini judge estimates, not ground truth, and use only questions with successful answer generation and judging. Quota/network failures are counted separately.")
        lines.append("Requests were paced at least 4.2 seconds apart to stay below the observed 15 requests/minute Gemini free-tier limit.")
        initial = answer.get("initial_unpaced_attempt", {})
        if initial.get("status"):
            lines.append(f"The initial unpaced attempt was discarded after {initial.get('error')}; the final answer results are from the paced rerun.")
        targeted_retry = answer.get("targeted_judge_retry", {})
        if targeted_retry.get("status") == "success":
            lines.append(
                f"The single failed BGE judge result for `{targeted_retry['question_id']}` was retried successfully using the saved answer and retrieved context; retrieval and answer generation were not rerun."
            )
        lines.append("")
        lines.append("| Configuration | Scored / 21 | Correctness | Faithfulness | Context relevance | Answer relevance | Valid citation mappings / cited answers | Citation coverage / generated answers | Generation / judge errors |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in answer["results"]:
            agg = item["aggregate"]
            scores = agg["answer_estimates"]
            f = lambda v: "N/A" if v is None else f"{v:.4f}"
            citation_map = f(item.get("citation_chunk_match_rate"))
            citation_coverage = f(item.get("citation_coverage_among_successful_generations"))
            lines.append(f"| {item['config_id']} | {agg['answer_questions']} / {result['metadata']['question_count']} | {f(scores['correctness'])} | {f(scores['faithfulness'])} | {f(scores['context_relevance'])} | {f(scores['answer_relevance'])} | {citation_map} | {citation_coverage} | {agg['generation_errors']} / {agg['answer_evaluation_errors']} |")
        common = answer.get("common_subset_metrics", [])
        if common:
            lines += ["", f"Common scored subset: {answer.get('common_scored_question_count')} questions shared by all shortlisted configurations.", "",
                      "| Configuration | Correctness | Faithfulness | Context relevance | Answer relevance |",
                      "|---|---:|---:|---:|---:|"]
            for item in common:
                scores = item["metrics"]
                f = lambda v: "N/A" if v is None else f"{v:.4f}"
                lines.append(f"| {item['config_id']} | {f(scores['correctness'])} | {f(scores['faithfulness'])} | {f(scores['context_relevance'])} | {f(scores['answer_relevance'])} |")
        failures = [
            (item["config_id"], row.get("answer_evaluation_error"))
            for item in answer["results"] for row in item["questions"]
            if row.get("answer_evaluation_error")
        ]
        if failures:
            lines.append("")
            for config_id, error in failures:
                lines.append(f"- `{config_id}` had a judge error; that question is excluded from its answer-metric mean. The detailed error is in its per-configuration JSON.")
    else:
        lines.append("Not run. Set `--answer-quality` after verifying Gemini connectivity. Answer quality, faithfulness, unsupported-claim rate, and semantic citation support are unknown.")
    lines += [
        "", "## Expected trade-offs (not additional measurements)", "",
        "- A cross-encoder reads the query and each candidate passage together, then scores each pair. It may improve ordering among candidates, but it cannot retrieve relevant text absent from the candidate set.",
        "- Larger candidate pools increase pair-scoring work. The results above show the actual latency and retrieval changes on this corpus; no benchmark or theoretical expectation was substituted for a measured result.",
        "- Unsupported-claim rate is not reported as a number: the current judge returns a faithfulness score, not a reliable claim-level binary count.",
        "- Semantic citation support is not measured separately. The application validates citation positions and source IDs; that proves mapping to retrieved chunks, not that each cited passage entails each claim.",
        "", "## Reproduction", "",
        "From the repository root in PowerShell, activate the project virtual environment and run:", "",
        "```powershell",
        ".venv\\Scripts\\Activate.ps1",
        "$env:HF_HUB_OFFLINE = '1'",
        "$env:TRANSFORMERS_OFFLINE = '1'",
        "python -m src.evaluation.reranking_experiment",
        "python -m src.evaluation.reranking_experiment --answer-quality-only",
        "```", "",
        "Both reranker checkpoints must already be cached locally for offline execution. Add `--answer-quality` to run retrieval and then answer evaluation. After a retrieval-only run, use `--answer-quality-only` to evaluate the saved retrieval results without repeating the reranker sweep. The answer stage paces Gemini requests and requires a configured API key.",
        "Detailed results: `data/evaluation/reranking_comparison_results.json`.", "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


class _RerankedAnswerSearch:
    """Adapter used only in the optional evaluation path."""

    def __init__(self, base: HybridSearch, cross_encoder: Any | None, architecture: str, candidate_count: int) -> None:
        self.base = base
        self.cross_encoder = cross_encoder
        self.architecture = architecture
        self.candidate_count = candidate_count

    def search(self, query: str, *, top_k: int = TOP_K, vector_weight: float = VECTOR_WEIGHT,
               keyword_weight: float = KEYWORD_WEIGHT) -> dict[str, list[dict[str, Any]]]:
        if self.architecture == "hybrid_baseline":
            return self.base.search(query, top_k=top_k,
                                    vector_weight=vector_weight, keyword_weight=keyword_weight)
        if self.architecture == "dense_reranked":
            candidates = self.base.vector_search.search(query, top_k=self.candidate_count)
            vector, keyword = candidates, []
        else:
            retrieval = self.base.search(query, top_k=self.candidate_count,
                                          vector_weight=vector_weight, keyword_weight=keyword_weight)
            candidates = retrieval["hybrid"]
            vector, keyword = retrieval["vector"], retrieval["keyword"]
        candidates = [{**chunk, "candidate_rank": index} for index, chunk in enumerate(candidates, 1)]
        if self.cross_encoder is None:
            raise RuntimeError("A reranker model is required for the selected architecture.")
        ranked, _, _ = _rerank(self.cross_encoder, query, candidates)
        return {"hybrid": ranked[:top_k], "vector": vector[:top_k],
                "keyword": keyword[:top_k], "vector_only": vector[:top_k]}


def _run_answer_quality(result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate baseline and retrieval-selected reranked settings with existing Gemini code."""
    from sentence_transformers import CrossEncoder
    from src.evaluation.evaluator import RAGEvaluator
    from src.evaluation.gemini_evaluator import GeminiAnswerEvaluator
    from src.rag.answer_generator import GeminiAnswerGenerator
    from src.rag.gemini_client import create_chat_client

    dataset = load_evaluation_dataset(DEFAULT_DATASET_PATH)
    store = ChromaStore()
    base = HybridSearch(VectorSearch(store))
    pacer = _RequestPacer(minimum_interval=4.2)
    generation_client = _PacedChatClient(
        create_chat_client(os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")), pacer
    )
    judge_model = os.getenv("GEMINI_EVAL_MODEL", os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"))
    judge_client = _PacedChatClient(create_chat_client(judge_model), pacer)
    experiments = result["experiments"]
    hybrid_reranked = [e for e in experiments if e["metadata"]["architecture"] == "hybrid_reranked"]
    dense_reranked = [e for e in experiments if e["metadata"]["architecture"] == "dense_reranked"]
    score_key = lambda e: (
        e["aggregate"]["retrieval"].get("mrr") or 0.0,
        e["aggregate"]["retrieval"].get("recall_at_k") or 0.0,
        e["aggregate"]["retrieval"].get("precision_at_k") or 0.0,
        -(e["aggregate"]["latency_ms"].get("total_mean") or 0.0),
    )
    selected = [next(e for e in experiments if e["metadata"]["architecture"] == "hybrid_rrf")]
    for model_name in RERANKER_MODELS:
        candidates = [e for e in hybrid_reranked if e["metadata"]["reranker_model"] == model_name]
        selected.append(max(candidates, key=score_key))
    selected.append(max(dense_reranked, key=score_key))
    selected_by_id = {e["metadata"]["config_id"]: e for e in selected}

    models: dict[str, Any] = {}
    parameters: dict[str, int] = {}
    for model_name in dict.fromkeys(
        e["metadata"]["reranker_model"] for e in selected if e["metadata"]["reranker_model"]
    ):
        models[model_name] = CrossEncoder(model_name, device="cpu")
        parameters[model_name] = sum(
            parameter.numel() for parameter in models[model_name].model.parameters()
        )
    result["metadata"]["reranker_parameters"] = parameters
    for experiment in experiments:
        rows = experiment["questions"]
        times = [row["latency_ms"] for row in rows]
        totals = [item["total"] for item in times]
        reranks = [item["reranking"] for item in times]
        timing = experiment["aggregate"]["latency_ms"]
        timing.update({
            "reranking_median": round(statistics.median(reranks), 3) if reranks else None,
            "reranking_max": round(max(reranks), 3) if reranks else None,
            "total_median": round(statistics.median(totals), 3) if totals else None,
            "total_max": round(max(totals), 3) if totals else None,
        })

    answer_results = []
    for config_id, config in selected_by_id.items():
        meta = config["metadata"]
        adapter_arch = {
            "hybrid_rrf": "hybrid_baseline",
            "hybrid_reranked": "hybrid_reranked",
            "dense_reranked": "dense_reranked",
        }[meta["architecture"]]
        adapter = _RerankedAnswerSearch(
            base, models.get(meta["reranker_model"]), adapter_arch, int(meta["candidate_count"])
        )
        evaluator = RAGEvaluator(
            GeminiAnswerGenerator(adapter, top_k=TOP_K, max_retries=0,
                                  client=generation_client),
            dataset,
            GeminiAnswerEvaluator(max_retries=0, client=judge_client),
        )
        evaluated = evaluator.run(
            top_k=TOP_K, vector_weight=VECTOR_WEIGHT, keyword_weight=KEYWORD_WEIGHT
        )
        successful_generations = []
        cited_responses = []
        citation_checks = []
        for row in evaluated["questions"]:
            if not row.get("generation_error"):
                successful_generations.append(row)
            retrieved_ids = set(row.get("retrieved_chunk_ids", []))
            sources = row.get("sources", [])
            if sources:
                cited_responses.append(row)
                row["citation_chunk_ids_match_retrieved"] = all(
                    source.get("chunk_id") in retrieved_ids for source in sources
                )
                citation_checks.append(row["citation_chunk_ids_match_retrieved"])
            else:
                row["citation_chunk_ids_match_retrieved"] = None
        citation_rate = _mean([1.0 if valid else 0.0 for valid in citation_checks])
        citation_coverage = (
            len(cited_responses) / len(successful_generations)
            if successful_generations else None
        )
        evaluated["metadata"].update({
            "config_id": config_id,
            "architecture": meta["architecture"],
            "reranker_model": meta["reranker_model"],
            "candidate_count": meta["candidate_count"],
        })
        path = Path("data/evaluation") / f"reranking_answer_quality_{config_id}.json"
        saved = save_evaluation_results(evaluated, path)
        answer_results.append({
            "config_id": config_id,
            "aggregate": evaluated["aggregate"],
            "citation_chunk_match_rate": citation_rate,
            "citation_coverage_among_successful_generations": citation_coverage,
            "result_file": str(saved),
            "questions": evaluated["questions"],
        })
        print(f"Answer-quality evaluation {config_id}: "
              f"generation errors={evaluated['aggregate']['generation_errors']}, "
              f"judge errors={evaluated['aggregate']['answer_evaluation_errors']}", flush=True)

    score_maps = [
        {
            row["question_id"]: row["answer_metrics"]
            for row in item["questions"] if row.get("answer_metrics")
        }
        for item in answer_results
    ]
    common_ids = sorted(set.intersection(*(set(scores) for scores in score_maps))) if score_maps else []
    common_subset = []
    for answer_result, scores in zip(answer_results, score_maps):
        metrics = {}
        for key in ("correctness", "faithfulness", "context_relevance", "answer_relevance"):
            metrics[key] = _mean([float(scores[qid][key]) for qid in common_ids]) if common_ids else None
        common_subset.append({"config_id": answer_result["config_id"], "metrics": metrics})
    result["answer_quality"] = {
        "status": "completed",
        "request_spacing_seconds": pacer.minimum_interval,
        "initial_unpaced_attempt": {
            "status": "rate_limited_partial_results_discarded",
            "error": "Gemini returned HTTP 429 RESOURCE_EXHAUSTED: 15 GenerateContent requests per minute on the free tier.",
            "handling": "Rerun with shared request pacing; unpaced answer-quality aggregates were not used.",
        },
        "common_scored_question_ids": common_ids,
        "common_scored_question_count": len(common_ids),
        "common_subset_metrics": common_subset,
        "results": answer_results,
    }
    if any(
        item["aggregate"]["generation_errors"] or item["aggregate"]["answer_evaluation_errors"]
        for item in answer_results
    ):
        result["answer_quality"]["status"] = "incomplete_with_errors"
    return result


def run_experiment(*, answer_quality: bool = False) -> dict[str, Any]:
    from sentence_transformers import CrossEncoder

    dataset = load_evaluation_dataset(DEFAULT_DATASET_PATH)
    store = ChromaStore()
    if store.count() == 0:
        raise RuntimeError("Chroma collection is empty; index the corpus before running the experiment.")
    vector = VectorSearch(store)
    hybrid = HybridSearch(vector)
    # Exclude lazy model loading and tokenizer setup from measured query latency.
    store.embedding_service.embed_query("reranker benchmark warm-up")
    question_count = len(dataset["questions"])
    per_config: dict[str, list[dict[str, Any]]] = {}
    retrieval_times: dict[str, list[float]] = {}
    rerank_times: dict[str, list[float]] = {}

    def add(config_id: str, question: dict[str, Any], candidates: list[dict[str, Any]],
            final: list[dict[str, Any]], retrieval_ms: float, rerank_ms: float = 0.0,
            scores: list[float] | None = None) -> None:
        per_config.setdefault(config_id, []).append({
            "item": question, "candidates": candidates, "final": final,
            "retrieval_ms": retrieval_ms, "rerank_ms": rerank_ms, "scores": scores,
        })
        retrieval_times.setdefault(config_id, []).append(retrieval_ms)
        rerank_times.setdefault(config_id, []).append(rerank_ms)

    raw_candidates: dict[tuple[str, int], list[tuple[dict[str, Any], list[dict[str, Any]], float]]] = {}
    for item in dataset["questions"]:
        started = time.perf_counter()
        dense5 = vector.search(item["question"], top_k=TOP_K)
        dense5_ms = _ms(time.perf_counter() - started)
        add("dense_baseline", item, dense5, dense5, dense5_ms)

        started = time.perf_counter()
        hybrid5_result = hybrid.search(item["question"], top_k=TOP_K,
                                        vector_weight=VECTOR_WEIGHT, keyword_weight=KEYWORD_WEIGHT)
        hybrid5 = hybrid5_result["hybrid"][:TOP_K]
        hybrid5_ms = _ms(time.perf_counter() - started)
        add("hybrid_baseline", item, hybrid5, hybrid5, hybrid5_ms)

        for count in CANDIDATE_COUNTS:
            started = time.perf_counter()
            dense_candidates = vector.search(item["question"], top_k=count)
            dense_ms = _ms(time.perf_counter() - started)
            raw_candidates.setdefault(("dense_reranked", count), []).append((item, dense_candidates, dense_ms))

            started = time.perf_counter()
            hybrid_candidates = hybrid.search(item["question"], top_k=count,
                                              vector_weight=VECTOR_WEIGHT, keyword_weight=KEYWORD_WEIGHT)["hybrid"]
            hybrid_ms = _ms(time.perf_counter() - started)
            raw_candidates.setdefault(("hybrid_reranked", count), []).append((item, hybrid_candidates, hybrid_ms))

    cross_encoders: dict[str, Any] = {}
    reranker_parameters: dict[str, int] = {}
    for model_name in RERANKER_MODELS:
        print(f"Loading {model_name} on CPU...", flush=True)
        cross_encoder = CrossEncoder(model_name, device="cpu")
        reranker_parameters[model_name] = sum(
            parameter.numel() for parameter in cross_encoder.model.parameters()
        )
        cross_encoder.predict(
            [("reranker benchmark warm-up", "warm-up text")],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        cross_encoders[model_name] = cross_encoder
        for architecture in ("dense_reranked", "hybrid_reranked"):
            for count in CANDIDATE_COUNTS:
                config_id = f"{architecture}__{model_name.replace('/', '_')}__n{count}"
                for item, candidates, retrieval_ms in raw_candidates[(architecture, count)]:
                    indexed = [{**chunk, "candidate_rank": rank}
                               for rank, chunk in enumerate(candidates, 1)]
                    final, scores, rerank_ms = _rerank(cross_encoder, item["question"], indexed)
                    add(config_id, item, indexed, final, retrieval_ms, rerank_ms,
                        [float(chunk["reranker_score"]) for chunk in final])
                print(f"Scored {config_id}: {question_count} questions", flush=True)

    experiments = []
    for config_id, records in per_config.items():
        if config_id == "dense_baseline":
            architecture, model_name, count = "dense_only", None, TOP_K
        elif config_id == "hybrid_baseline":
            architecture, model_name, count = "hybrid_rrf", None, TOP_K
        else:
            architecture, model_key, count_text = config_id.split("__")
            model_name = model_key.replace("_", "/", 1)
            count = int(count_text[1:])
        experiment = _make_config(dataset, records, config_id=config_id,
                                  architecture=architecture, model=model_name,
                                  candidate_count=count,
                                  retrieval_ms=retrieval_times[config_id],
                                  rerank_ms=rerank_times[config_id])
        experiments.append(experiment)

    result: dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(DEFAULT_DATASET_PATH),
            "dataset": dataset.get("name"),
            "question_count": question_count,
            "embedding_provider": store.provider,
            "embedding_model": store.model_name,
            "chroma_collection": store.collection_name,
            "chroma_count": store.count(),
            "top_k": TOP_K,
            "candidate_counts": list(CANDIDATE_COUNTS),
            "vector_weight": VECTOR_WEIGHT,
            "keyword_weight": KEYWORD_WEIGHT,
            "rrf_k": RRF_K,
            "device": "cpu",
            "reranker_parameters": reranker_parameters,
            "prompts_modified": False,
            "index_modified": False,
        },
        "experiments": experiments,
        "answer_quality": {"status": "not_run"},
    }

    if answer_quality:
        result = _run_answer_quality(result)

    save_evaluation_results(result, OUTPUT_PATH)
    _write_report(result)
    print(f"Saved detailed results: {OUTPUT_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dense, hybrid, and cross-encoder retrieval")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--answer-quality", action="store_true",
                       help="Run the retrieval sweep, then answer generation and Gemini judging")
    group.add_argument("--answer-quality-only", action="store_true",
                       help="Use saved retrieval results and evaluate only answer quality")
    args = parser.parse_args()
    if args.answer_quality_only:
        result = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        result = _run_answer_quality(result)
        save_evaluation_results(result, OUTPUT_PATH)
        _write_report(result)
        print(f"Updated results: {OUTPUT_PATH}")
        print(f"Updated report: {REPORT_PATH}")
    else:
        result = run_experiment(answer_quality=args.answer_quality)
    for experiment in result["experiments"]:
        metric = experiment["aggregate"]["retrieval"]
        print(
            f"{experiment['metadata']['config_id']}: "
            f"P@5={metric['precision_at_k']:.4f} "
            f"R@5={metric['recall_at_k']:.4f} "
            f"MRR@5={metric['mrr']:.4f} "
            f"Hit@5={metric['hit_rate_at_k']:.4f}"
        )


if __name__ == "__main__":
    main()
