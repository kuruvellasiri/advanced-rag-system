# Dense and BM25 Weight Comparison

- Run time (UTC): 2026-11-20T07:42:15.302445+00:00
- Dataset: `data\evaluation\aws_overview_eval.json` (21 questions)
- Top-K: 5
- RRF constant: 60
- Embeddings: `local` / `sentence-transformers/all-MiniLM-L6-v2`
- Chroma collection: `aws_content_local_sentence_transformers_all_minilm_l6_v2` (512 records)
- Method: weighted RRF adds `weight / (60 + rank)` for each ranking; weights are multipliers, not percentages.
- Index and source data were only read; no rebuild or re-embedding was performed.

## Retrieval results

| Experiment | Dense | BM25 | Precision@5 | Recall@5 | MRR@5 | Hit Rate@5 | Errors | Results JSON |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.00 | 0.00 | 0.2095 | 0.9524 | 0.7492 | 0.9524 | 0 | `weight_experiment_01_dense_1.00_bm25_0.00.json` |
| 2 | 0.75 | 0.25 | 0.2095 | 0.9524 | 0.8175 | 0.9524 | 0 | `weight_experiment_02_dense_0.75_bm25_0.25.json` |
| 3 | 0.50 | 0.50 | 0.2190 | 1.0000 | 0.8413 | 1.0000 | 0 | `weight_experiment_03_dense_0.50_bm25_0.50.json` |
| 4 | 0.25 | 0.75 | 0.2000 | 0.9524 | 0.9286 | 0.9524 | 0 | `weight_experiment_04_dense_0.25_bm25_0.75.json` |
| 5 | 0.00 | 1.00 | 0.2000 | 0.9524 | 0.8135 | 0.9524 | 0 | `weight_experiment_05_dense_0.00_bm25_1.00.json` |
| 6 | 1.00 | 1.00 | 0.2190 | 1.0000 | 0.8413 | 1.0000 | 0 | `weight_experiment_06_dense_1.00_bm25_1.00.json` |
| 7 | 1.50 | 1.00 | 0.2095 | 0.9524 | 0.8175 | 0.9524 | 0 | `weight_experiment_07_dense_1.50_bm25_1.00.json` |
| 8 | 1.00 | 1.50 | 0.2000 | 0.9524 | 0.8571 | 0.9524 | 0 | `weight_experiment_08_dense_1.00_bm25_1.50.json` |

## Recommended overall setting

- Recommended: dense `0.50`, BM25 `0.50` (Experiment 3). It is the strongest balanced setting: Precision@5 `0.2190`, Recall@5 `1.0000`, MRR@5 `0.8413`, and Hit Rate@5 `1.0000`.
- Experiment 6 (`1.00`, `1.00`) produces the same rankings and metrics. RRF weights are multipliers; scaling both equally does not change their relative influence.
- Experiment 4 (`0.25`, `0.75`) has the highest MRR@5 (`0.9286`), but it misses both annotated S3 chunks for q04. The recommended equal-weight setting retrieved relevant context for all 21 questions, so it is safer for answer coverage.
- This recommendation is based on retrieval only. Answer quality could not be measured in this environment (see below).

## Answer-quality evaluation

- Shortlisted Experiment 4 (highest MRR) and Experiment 3 (best precision/recall/hit rate).
- Attempted full answer generation for all 21 questions with Experiment 4. All 21 failed before producing an answer because outbound socket access to Gemini was denied (`WinError 10013`). This was not an HTTP 429 quota response.
- No answer-generation or Gemini judge scores were produced. Correctness, faithfulness, context relevance, unsupported-information rate, and semantic citation support are **unknown**, not zero and not successful.
- Experiment 3 was not sent to Gemini after the same environment/network failure; its per-question status is explicitly recorded as `not_run`.
- The application validates each citation number against the retrieved context list and maps it to that actual chunk. The current judge reports answer relevance, faithfulness, context relevance, and correctness, but has no separate per-citation entailment metric. No semantic citation-support result is claimed here.
- Files: `answer_quality_dense_0.25_bm25_0.75.json` (21 generation errors) and `answer_quality_dense_0.50_bm25_0.50.json` (explicitly not run).

## Per-question retrieval failures

### Experiment 1 (dense 1.00, BM25 0.00)
- `q20`: no relevant chunk in top 5; expected `page-0152-chunk-0001`; got `page-0044-chunk-0001, page-0043-chunk-0001, page-0154-chunk-0001, page-0038-chunk-0002, page-0011-chunk-0001`.

### Experiment 2 (dense 0.75, BM25 0.25)
- `q20`: no relevant chunk in top 5; expected `page-0152-chunk-0001`; got `page-0044-chunk-0001, page-0043-chunk-0001, page-0154-chunk-0001, page-0038-chunk-0002, page-0011-chunk-0001`.

### Experiment 3 (dense 0.50, BM25 0.50)
- No retrieval errors or hit-rate misses.

### Experiment 4 (dense 0.25, BM25 0.75)
- `q04`: no relevant chunk in top 5; expected `page-0149-chunk-0003, page-0149-chunk-0004`; got `page-0150-chunk-0001, page-0028-chunk-0002, page-0150-chunk-0003, page-0150-chunk-0002, page-0084-chunk-0005`.

### Experiment 5 (dense 0.00, BM25 1.00)
- `q04`: no relevant chunk in top 5; expected `page-0149-chunk-0003, page-0149-chunk-0004`; got `page-0028-chunk-0002, page-0150-chunk-0003, page-0150-chunk-0002, page-0150-chunk-0001, page-0084-chunk-0005`.

### Experiment 6 (dense 1.00, BM25 1.00)
- No retrieval errors or hit-rate misses.

### Experiment 7 (dense 1.50, BM25 1.00)
- `q20`: no relevant chunk in top 5; expected `page-0152-chunk-0001`; got `page-0044-chunk-0001, page-0043-chunk-0001, page-0154-chunk-0001, page-0038-chunk-0002, page-0011-chunk-0001`.

### Experiment 8 (dense 1.00, BM25 1.50)
- `q04`: no relevant chunk in top 5; expected `page-0149-chunk-0003, page-0149-chunk-0004`; got `page-0150-chunk-0001, page-0028-chunk-0002, page-0150-chunk-0003, page-0150-chunk-0002, page-0084-chunk-0005`.

## Reproduction

From the project root in PowerShell, run retrieval-only evaluation for any individual setting:

```powershell
python -m src.retrieval.cli --evaluate --retrieval-only --top-k 5 --vector-weight 1.00 --keyword-weight 0.00 --output data/evaluation/weight_experiment_01_dense_1.00_bm25_0.00.json
```

Change the two weights and output filename for the other rows. The existing CLI accepts both flags; its RRF constant remains the implementation default of 60.

Detailed rankings for every question and each experiment are in the individual JSON files. Combined metrics are in `data/evaluation/weight_comparison_results.json`.
