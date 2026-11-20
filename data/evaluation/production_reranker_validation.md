# Production MiniLM Reranker Validation

- Dataset: `data\evaluation\aws_overview_eval.json` (21 questions).
- Collection: `aws_content_local_sentence_transformers_all_minilm_l6_v2` (512 chunks; queried only).
- Settings: dense/BM25 weights 0.50/0.50, RRF k=60, top 5 from 10, model `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU.
- No index rebuild and no Gemini calls were performed during this production validation.

## Measured production retrieval vs saved baseline

| Metric | Hybrid RRF baseline (saved) | Integrated production reranker (measured) |
|---|---:|---:|
| Precision@5 | 0.2190 | 0.2190 |
| Recall@5 | 1.0000 | 1.0000 |
| MRR@5 | 0.8413 | 0.9762 |
| Hit Rate@5 | 1.0000 | 1.0000 |
| Median query retrieval ms | 15.7 (saved) | 367.7 |
| Mean query retrieval ms | 16.4 (saved) | 385.4 |

The baseline figures are reused from the earlier experiment. Production integrated reranker figures above were freshly measured using the deployed wrapper and all evaluation questions. Latency excludes first-time model and embedding load, and captures the production search wrapper including candidate retrieval and reranking.

## Answer quality

No answer generation or Gemini judging was executed in this validation. The existing saved evaluation reports answer-quality estimates for the same hybrid + MiniLM + N=10 architecture: correctness, faithfulness, context relevance, and answer relevance are listed in the JSON under `answer_quality_reused`. Those scores are reused experiment results, not newly measured production-path output.

## Per-question rankings

Per-question candidate and final chunk rankings, relevant IDs, and measured latency are stored in `production_reranker_validation.json`.

## Reproduction

```powershell
.venv\Scripts\Activate.ps1
$env:RERANKER_ENABLED = 'true'
$env:RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
$env:RERANKER_CANDIDATE_COUNT = '10'
$env:RERANKER_TOP_K = '5'
python -m src.retrieval.cli --evaluate --retrieval-only --top-k 5 --vector-weight 0.5 --keyword-weight 0.5 --output data/evaluation/production_reranker_retrieval.json
```
