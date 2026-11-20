# Dense, Hybrid, and Cross-Encoder Reranking Comparison

- Run time (UTC): 2026-11-20T08:32:52.505054+00:00
- Dataset: `data\evaluation\aws_overview_eval.json` (21 questions)
- Embeddings: `local` / `sentence-transformers/all-MiniLM-L6-v2`
- Chroma collection: `aws_content_local_sentence_transformers_all_minilm_l6_v2` (512 chunks; read-only)
- Weights: dense `0.50`, BM25 `0.50`; RRF constant `60`.
- Final result size: top `5`. Cross-encoder candidates: `10, 20, 30`.
- Runtime device: `cpu`.
- Each latency is a measured wall-clock time in this environment, using local CPU inference; model-load time is excluded. The table uses median and p95 to limit distortion from isolated scheduling outliers; per-question raw timings and maxima are in JSON.

## Reranker resource profile

| Model | Parameters observed | Compatibility and model-card size |
|---|---:|---|
| [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 22,713,601 | Loads through Sentence Transformers `CrossEncoder`; model card reports 22.7M parameters. |
| [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) | 278,044,417 | Loads through Sentence Transformers `CrossEncoder`; model card lists 1.11 GB safetensors weights and a 3.36 GB repository. |

## Retrieval results

| Architecture / model / N | Precision@5 | Recall@5 | MRR@5 | Hit@5 | Candidate recall@N | Dropped relevant chunks | Retrieval ms mean | Rerank ms median (p95) | Total ms median (p95) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_only | 0.2095 | 0.9524 | 0.7492 | 0.9524 | 0.9520 | 0 | 15.1 | 0.0000 (0.0000) | 14.7 (19.5) |
| hybrid_rrf | 0.2190 | 1.0000 | 0.8413 | 1.0000 | 1.0000 | 0 | 16.4 | 0.0000 (0.0000) | 15.7 (20.5) |
| dense_reranked / `cross-encoder/ms-marco-MiniLM-L-6-v2` / 10 | 0.2095 | 0.9524 | 0.9286 | 0.9524 | 0.9520 | 0 | 15.3 | 316.6 (359.5) | 332.4 (375.7) |
| dense_reranked / `cross-encoder/ms-marco-MiniLM-L-6-v2` / 20 | 0.2095 | 0.9524 | 0.9286 | 0.9524 | 0.9520 | 0 | 15.7 | 582.4 (649.5) | 595.5 (665.8) |
| dense_reranked / `cross-encoder/ms-marco-MiniLM-L-6-v2` / 30 | 0.2095 | 0.9524 | 0.9286 | 0.9524 | 0.9520 | 0 | 17.9 | 920.5 (994.0) | 937.8 (1011.4) |
| hybrid_reranked / `cross-encoder/ms-marco-MiniLM-L-6-v2` / 10 | 0.2190 | 1.0000 | 0.9762 | 1.0000 | 1.0000 | 0 | 16.7 | 324.8 (429.7) | 341.9 (445.9) |
| hybrid_reranked / `cross-encoder/ms-marco-MiniLM-L-6-v2` / 20 | 0.2190 | 1.0000 | 0.9762 | 1.0000 | 1.0000 | 0 | 17.9 | 589.1 (695.0) | 608.0 (713.4) |
| hybrid_reranked / `cross-encoder/ms-marco-MiniLM-L-6-v2` / 30 | 0.2190 | 1.0000 | 0.9524 | 1.0000 | 1.0000 | 0 | 20.1 | 910.7 (1037.6) | 928.3 (1059.3) |
| dense_reranked / `BAAI/bge-reranker-base` / 10 | 0.2095 | 0.9524 | 0.9286 | 0.9524 | 0.9520 | 0 | 15.3 | 2161.1 (2310.5) | 2176.1 (2326.0) |
| dense_reranked / `BAAI/bge-reranker-base` / 20 | 0.2095 | 0.9524 | 0.9286 | 0.9524 | 0.9520 | 0 | 15.7 | 4189.8 (4825.1) | 4206.4 (4841.4) |
| dense_reranked / `BAAI/bge-reranker-base` / 30 | 0.2095 | 0.9524 | 0.9286 | 0.9524 | 0.9520 | 0 | 17.9 | 6442.7 (7101.8) | 6462.8 (7119.8) |
| hybrid_reranked / `BAAI/bge-reranker-base` / 10 | 0.2190 | 1.0000 | 0.9762 | 1.0000 | 1.0000 | 0 | 16.7 | 2508.1 (3103.1) | 2524.5 (3120.3) |
| hybrid_reranked / `BAAI/bge-reranker-base` / 20 | 0.2190 | 1.0000 | 0.9762 | 1.0000 | 1.0000 | 0 | 17.9 | 4792.2 (5667.9) | 4812.0 (5685.7) |
| hybrid_reranked / `BAAI/bge-reranker-base` / 30 | 0.2190 | 1.0000 | 0.9762 | 1.0000 | 1.0000 | 0 | 20.1 | 6768.8 (8064.6) | 6786.2 (8084.9) |

## Findings and recommendation

- **Hybrid vs dense:** Hybrid RRF improves all four retrieval measures on this dataset: Precision@5 `0.2095` to `0.2190`, Recall@5 `0.9524` to `1.0000`, MRR@5 `0.7492` to `0.8413`, and Hit Rate@5 `0.9524` to `1.0000`.
- **Hybrid reranking:** MiniLM Cross-Encoder at N=10 preserves Precision/Recall/Hit at `0.2190` / `1.0000` / `1.0000` and raises MRR@5 from `0.8413` to `0.9762`. No relevant candidate chunk was removed from the final top five in this run.
- **Dense reranking:** MiniLM raises MRR@5 from `0.7492` to `0.9286`, but does not recover q20's relevant chunk because it is absent from the dense candidate set.
- **Model comparison:** At hybrid N=10, both models tie on retrieval (Precision `0.2190`, Recall `1.0000`, MRR `0.9762`, Hit `1.0000`). MiniLM is much faster on this CPU; BGE adds no measured retrieval benefit.
- **Candidate count:** N=10 is the best trade-off. MiniLM N=20 has the same metrics but higher latency; MiniLM N=30 lowers MRR to `0.9524`. BGE's N=10/20/30 retrieval scores tie, so larger candidate sets add cost without improving scores.
- **Latency:** Hybrid RRF median retrieval is `15.7 ms`; MiniLM hybrid reranking N=10 is `341.9 ms` end to end (about `326.2 ms` extra per query in this local CPU run).
- **Recommendation:** Keep architecture C (Dense + BM25 + weighted RRF + `cross-encoder/ms-marco-MiniLM-L-6-v2`, N=10, select top five) when the roughly 0.33-second local retrieval median is acceptable. It gives the highest measured hybrid MRR and retains full annotated coverage. Keep architecture B if minimum retrieval latency matters more than ranking the relevant chunk earlier. This experiment did not change the production pipeline.
- **Answer quality:** Use the Gemini results below cautiously. Faithfulness is a judge estimate; the supported-claim rate and semantic citation entailment are not reliably measured by this evaluator.
- **Latency outlier:** BGE hybrid N=30 recorded a q07 maximum of `1034229.0 ms` (17.2 minutes), versus median `6786.2 ms` and p95 `8084.9 ms`. Treat this single extreme as an observed tail event and repeat before using BGE N=30 operationally.


## Candidate-removal and per-question details

Every JSON experiment includes per-question candidate IDs and ranks, final top-five IDs and ranks, retrieval metrics, any relevant chunks dropped from the candidate set, and measured latencies.

### dense_baseline
- `q20` missed all relevant chunks at final top 5; expected `page-0152-chunk-0001`, returned `page-0044-chunk-0001, page-0043-chunk-0001, page-0154-chunk-0001, page-0038-chunk-0002, page-0011-chunk-0001`.

### dense_reranked__cross-encoder_ms-marco-MiniLM-L-6-v2__n10
- `q20` missed all relevant chunks at final top 5; expected `page-0152-chunk-0001`, returned `page-0080-chunk-0001, page-0154-chunk-0001, page-0011-chunk-0003, page-0044-chunk-0001, page-0011-chunk-0001`.

### dense_reranked__cross-encoder_ms-marco-MiniLM-L-6-v2__n20
- `q20` missed all relevant chunks at final top 5; expected `page-0152-chunk-0001`, returned `page-0080-chunk-0001, page-0154-chunk-0001, page-0011-chunk-0003, page-0011-chunk-0002, page-0144-chunk-0001`.

### dense_reranked__cross-encoder_ms-marco-MiniLM-L-6-v2__n30
- `q20` missed all relevant chunks at final top 5; expected `page-0152-chunk-0001`, returned `page-0142-chunk-0005, page-0080-chunk-0001, page-0138-chunk-0003, page-0154-chunk-0001, page-0011-chunk-0003`.

### dense_reranked__BAAI_bge-reranker-base__n10
- `q20` missed all relevant chunks at final top 5; expected `page-0152-chunk-0001`, returned `page-0011-chunk-0003, page-0001-chunk-0001, page-0043-chunk-0001, page-0154-chunk-0001, page-0011-chunk-0001`.

### dense_reranked__BAAI_bge-reranker-base__n20
- `q20` missed all relevant chunks at final top 5; expected `page-0152-chunk-0001`, returned `page-0011-chunk-0003, page-0011-chunk-0002, page-0001-chunk-0001, page-0043-chunk-0001, page-0042-chunk-0001`.

### dense_reranked__BAAI_bge-reranker-base__n30
- `q20` missed all relevant chunks at final top 5; expected `page-0152-chunk-0001`, returned `page-0011-chunk-0003, page-0011-chunk-0002, page-0142-chunk-0005, page-0001-chunk-0001, page-0013-chunk-0001`.

## Answer-quality evaluation

Measured only for the selected shortlisted architectures. Scores are Gemini judge estimates, not ground truth, and use only questions with successful answer generation and judging. Quota/network failures are counted separately.
Requests were paced at least 4.2 seconds apart to stay below the observed 15 requests/minute Gemini free-tier limit.
The initial unpaced attempt was discarded after Gemini returned HTTP 429 RESOURCE_EXHAUSTED: 15 GenerateContent requests per minute on the free tier.; the final answer results are from the paced rerun.
The single failed BGE judge result for `q07` was retried successfully using the saved answer and retrieved context; retrieval and answer generation were not rerun.

| Configuration | Scored / 21 | Correctness | Faithfulness | Context relevance | Answer relevance | Valid citation mappings / cited answers | Citation coverage / generated answers | Generation / judge errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_baseline | 21 / 21 | 0.9762 | 1.0000 | 1.0000 | 0.9810 | 1.0000 | 1.0000 | 0 / 0 |
| hybrid_reranked__cross-encoder_ms-marco-MiniLM-L-6-v2__n10 | 21 / 21 | 0.9833 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 / 0 |
| hybrid_reranked__BAAI_bge-reranker-base__n10 | 21 / 21 | 0.9833 | 1.0000 | 1.0000 | 0.9905 | 1.0000 | 1.0000 | 0 / 0 |
| dense_reranked__cross-encoder_ms-marco-MiniLM-L-6-v2__n10 | 21 / 21 | 0.9667 | 1.0000 | 0.9524 | 0.9905 | 1.0000 | 0.9524 | 0 / 0 |

Common scored subset: 21 questions shared by all shortlisted configurations.

| Configuration | Correctness | Faithfulness | Context relevance | Answer relevance |
|---|---:|---:|---:|---:|
| hybrid_baseline | 0.9762 | 1.0000 | 1.0000 | 0.9810 |
| hybrid_reranked__cross-encoder_ms-marco-MiniLM-L-6-v2__n10 | 0.9833 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_reranked__BAAI_bge-reranker-base__n10 | 0.9833 | 1.0000 | 1.0000 | 0.9905 |
| dense_reranked__cross-encoder_ms-marco-MiniLM-L-6-v2__n10 | 0.9667 | 1.0000 | 0.9524 | 0.9905 |

## Expected trade-offs (not additional measurements)

- A cross-encoder reads the query and each candidate passage together, then scores each pair. It may improve ordering among candidates, but it cannot retrieve relevant text absent from the candidate set.
- Larger candidate pools increase pair-scoring work. The results above show the actual latency and retrieval changes on this corpus; no benchmark or theoretical expectation was substituted for a measured result.
- Unsupported-claim rate is not reported as a number: the current judge returns a faithfulness score, not a reliable claim-level binary count.
- Semantic citation support is not measured separately. The application validates citation positions and source IDs; that proves mapping to retrieved chunks, not that each cited passage entails each claim.

## Reproduction

From the repository root in PowerShell, activate the project virtual environment and run:

```powershell
.venv\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m src.evaluation.reranking_experiment
python -m src.evaluation.reranking_experiment --answer-quality-only
```

Both reranker checkpoints must already be cached locally for offline execution. Add `--answer-quality` to run retrieval and then answer evaluation. After a retrieval-only run, use `--answer-quality-only` to evaluate the saved retrieval results without repeating the reranker sweep. The answer stage paces Gemini requests and requires a configured API key.
Detailed results: `data/evaluation/reranking_comparison_results.json`.
