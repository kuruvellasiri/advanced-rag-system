# AWS RAG Project

This repository is a single-PDF retrieval augmented generation (RAG) application for `documents/aws-overview.pdf`. The code implements PDF extraction and cleanup, page routing, recursive chunking, configurable local Sentence Transformers or Gemini embeddings, persistent Chroma indexing, BM25 plus vector retrieval fused with reciprocal rank fusion (RRF), Gemini answer generation, and retrieval/answer evaluation. It has no reranker and no query rewriting.

## Architecture and query flow

```text
PDF -> PyMuPDFLoader -> whitespace cleanup -> page/section routing
    -> recursive chunks -> configured embeddings -> provider-specific Chroma collection

Question -> query embedding from the same model -> Chroma vector ranking --+
          -> local BM25 ranking -----------------------> weighted RRF
          -> retrieved excerpts + source metadata -> Gemini chat -> answer + cited references
```

### Ingestion and indexing

`src/preprocess.py` runs `src/ingestion/pdf_loader.py`, `text_cleaner.py`, `section_router.py`, then `src/chunking/recursive_chunker.py`. PyMuPDFLoader creates one record per physical PDF page; the loader stores a 1-based `page_number` alongside original loader metadata (whose `page` remains zero-based). Cleanup normalizes whitespace and preserves paragraph and list line boundaries. Physical pages 3â€“10 are routed to navigation; the standalone Document History section is routed to history; the AWS Glossary remains in content. RecursiveCharacterTextSplitter uses 800 characters and 150 overlap by default. IDs are deterministic per physical page and chunk order.

Preprocessing writes extracted and cleaned page JSON plus three chunk files under `data/`. Only `content_chunks.json` is embedded/indexed for Q&A; navigation and history chunks are retained as separate output files and are not queried. `src/embeddings/factory.py` selects either the default local `sentence-transformers/all-MiniLM-L6-v2` model (384 dimensions) or optional Gemini `gemini-embedding-001`. The local service uses the same model and normalization for document and query vectors; its model downloads on first use and is cached by Sentence Transformers. `src/vectorstore/chroma_store.py` stores vectors, text, and scalar metadata in persistent Chroma at `data/chroma/`. Chroma collections are separated by provider and model: local defaults to `aws_content_local_sentence_transformers_all_minilm_l6_v2`; the existing Gemini default remains `aws_content`. The new local collection begins empty and indexes all content chunks once; later runs skip unchanged texts and update metadata. Existing Gemini collections are not migrated, overwritten, or deleted. The index CLI retains stale IDs rather than deleting records; `--rebuild` only forces embedding for current chunks.

### Query to answer

`src/retrieval/cli.py` initializes the provider-specific Chroma collection and loads content chunks for local BM25. For a question, `HybridSearch` embeds the query using the configured embedding provider and asks Chroma for cosine-distance ranked results, independently gets BM25 ranked results, then fuses rankings with weighted RRF (constant 60; default weights 1.0 each). There is no query rewrite, reranker, or separate context compression step. Up to `top_k` fused chunks and their page/chunk metadata are formatted as prompt context for Gemini chat. Answer generation is independent of the embedding provider and still requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`. The default chat model is `gemini-3.5-flash-lite` (overridable with `GEMINI_CHAT_MODEL` or `--model`); temperature is 0. The prompt directs the model to use only context and cite claims. The app accepts only exact page/chunk citations matching retrieved chunks, renders references from those cited chunks, and falls back to the insufficient-information message if the model gives no valid citation. Empty retrieval skips the model call.

This makes references traceable but does not prove the model's claims are entailed by the cited text. Similarity retrieval can return irrelevant context, and a model can still make a false claim while citing a real chunk. The model's insufficient-context judgment is not a separately calibrated confidence test. Retrieval evaluation and human review remain important.

## Implemented, partial, and absent components

| Component | Status | Implementation |
|---|---|---|
| PDF ingestion | Implemented for one configured PDF | `src/ingestion/pdf_loader.py` |
| Text cleanup | Implemented | `src/ingestion/text_cleaner.py` |
| Page/section routing | Implemented; page ranges/heading rules are document-specific | `src/ingestion/section_router.py` |
| Chunking | Implemented; recursive character chunks, page-scoped IDs | `src/chunking/recursive_chunker.py` |
| Embeddings | Implemented through local Sentence Transformers by default; Gemini remains optional | `src/embeddings/` |
| Vector database | Implemented with provider/model-specific persistent Chroma collections | `src/vectorstore/chroma_store.py` |
| BM25 | Implemented in-process against content chunk JSON | `src/retrieval/bm25_search.py` |
| Vector retrieval | Implemented against Chroma | `src/retrieval/vector_search.py` |
| Hybrid search / RRF | Implemented, weighted reciprocal-rank fusion | `src/retrieval/hybrid_search.py`, `rrf.py` |
| Reranking | Missing | No cross-encoder/LLM reranker in code |
| Query rewriting | Missing | The raw user question is sent to both retrievers |
| Context construction | Implemented as retrieved chunk text plus identifiers | `src/rag/prompts.py` |
| Answer generation | Implemented with Gemini and deterministic source rendering | `src/rag/answer_generator.py` |
| Evaluation | Implemented; default dataset is present locally under `data/evaluation/` | `src/evaluation/` |

## Setup

Run commands from the project root in Windows PowerShell. Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) for Gemini chat answers/evaluation. Local embedding and indexing require no Gemini key. The Sentence Transformers model is downloaded at first local use, which requires network access, and is cached for subsequent runs.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`requirements.txt` installs LangChain community/text splitters/Google GenAI, Sentence Transformers, ChromaDB, python-dotenv, PyMuPDF, and pytest. If PowerShell prevents script activation, use the environment's Python directly, for example `.venv\Scripts\python.exe -m src.preprocess`.

Embedding settings are `EMBEDDING_PROVIDER=local` and `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`; `EMBEDDING_BATCH_SIZE` controls local batches (default 32). To choose Gemini embeddings, set both to `EMBEDDING_PROVIDER=gemini` and `EMBEDDING_MODEL=gemini-embedding-001`. Chat defaults to `gemini-3.5-flash-lite`; it is configured separately using `GEMINI_CHAT_MODEL`. Gemini embedding rate/retry settings only apply to that optional provider.

## PowerShell command reference

No separate command exists for each internal stage. `python -m src.preprocess` performs extraction, cleaning, routing, and chunking in one run; `python -m src.vectorstore.index_chunks` performs embedding and indexing together. The Python modules listed below are the actual entry points. Unless noted, commands are optional for a use case but prerequisite commands in the table must have been run first.

| Purpose | Command | What it does, prerequisite, and expected output |
|---|---|---|
| Create environment | `python -m venv .venv` | Creates `.venv`; once per environment. |
| Activate environment | `.venv\Scripts\Activate.ps1` | Activates the environment in this PowerShell session. |
| Install dependencies | `python -m pip install -r requirements.txt` | Installs dependencies; required before running modules/tests. |
| Extract, clean, route, chunk | `python -m src.preprocess` | Runs `src/preprocess.py`; required before first indexing and after changing the PDF/chunk settings. Prints page and chunk counts; writes JSON under `data/extracted_text/` and `data/chunks/`. |
| Set chunk parameters | `python -m src.preprocess --chunk-size 800 --chunk-overlap 150` | Same `src/preprocess.py` pipeline with supported overrides; overlap must be less than positive chunk size. |
| Embed and create/update Chroma | `python -m src.vectorstore.index_chunks` | Runs `src/vectorstore/index_chunks.py`; requires preprocessing output. With local embeddings, no API key is needed; downloads the model on first use. Prints provider/model/collection and indexing counts. |
| Force re-embedding | `python -m src.vectorstore.index_chunks --rebuild` | Same indexer; re-embeds every content chunk into the selected provider/model collection. Optional maintenance operation; does not modify other collections. |
| BM25 retrieval | No standalone CLI command | BM25 is available via `BM25Search.from_json()` in Python and `--compare` in the query CLI; no dedicated command is implemented. |
| Vector retrieval | No standalone CLI command | `VectorSearch.search()` is used by hybrid search; no dedicated command is implemented. Requires populated Chroma; local query embedding has no API dependency. |
| Hybrid retrieval / RRF / answer | `python -m src.retrieval.cli --question "What is Amazon EC2?"` | Runs `src/retrieval/cli.py`; requires indexed Chroma, chunk JSON, and Gemini API key for answer generation. Prints answer and cited references. The first query requires prior preprocessing and indexing. |
| Answer only | `python -m src.retrieval.cli --question "What is Amazon EC2?" --answer-only` | Same module; prints answer text only (including source references if supported). |
| Interactive Q&A | `python -m src.retrieval.cli` | Same module; prompts until blank input/exit. Requires populated index, chunk JSON, and API key. |
| Tune retrieval/model | `python -m src.retrieval.cli --question "What is Amazon EC2?" --top-k 5 --vector-weight 1 --keyword-weight 1 --model gemini-3.5-flash-lite` | Same module; optional top-k, RRF weights, and Gemini model overrides. Prints answer and sources. |
| Compare rankings | `python -m src.retrieval.cli --question "What is Amazon EC2?" --compare` | Same module; prints answer/sources plus vector-only and BM25 rankings. |
| Reranking | No command | Not implemented. |
| Query rewriting | No command | Not implemented. |
| Full answer-generation evaluation | `python -m src.retrieval.cli --evaluate --no-llm-eval` | Runs dataset retrieval, generation, and deterministic retrieval metrics, without LLM judging. Requires the dataset path, populated index, and API key. Saves `data/evaluation/latest_results.json`. |
| Full evaluation with judge | `python -m src.retrieval.cli --evaluate` | Same module; also asks Gemini to estimate answer relevance, faithfulness, context relevance, and correctness. Saves latest results JSON. |
| Limit evaluation | `python -m src.retrieval.cli --evaluate --limit 5` | Runs first five dataset records; same prerequisites/output as selected evaluation mode. |
| Retrieval-only evaluation | `python -m src.retrieval.cli --evaluate --retrieval-only` | Runs retrieval and deterministic metrics only (no Gemini chat/judge; uses the configured embedding provider). Default output is `data/evaluation/retrieval_only_results.json`. The current CLI still checks for a Gemini API key on startup. |
| Set dataset/output | `python -m src.retrieval.cli --evaluate --dataset data/evaluation/my_dataset.json --output data/evaluation/my_run.json` | Uses the specified dataset and results paths. Dataset must follow the schema below. |
| Individual test file | `python -m pytest tests/test_rag.py -q` | Runs answer-generation/prompt tests with mocked Gemini calls. No API key needed. Other files: `tests/test_preprocessing.py`, `tests/test_retrieval.py`, `tests/test_evaluation.py`. |
| Individual test | `python -m pytest tests/test_rag.py::test_answer_generation_without_context_skips_llm -q` | Runs one named test; replace file and function with an implemented test. |
| All tests | `python -m pytest -q` | Runs all files under `tests/`; Gemini calls are mocked. |

Commands for extraction-only, cleaning-only, chunking-only, embedding-only, vector-only search, BM25-only, reranking, or query rewriting are **not** implemented. These stages must not be invoked with invented module names.

See [COMMANDS.md](COMMANDS.md) for the provider setup and end-to-end PowerShell command sequence.

## Evaluation data and metrics

The CLI's default dataset is `data/evaluation/aws_overview_eval.json`, and that file is present in the current workspace. The loader accepts either a list of records or an object containing `questions`. Each question has `id` (optional), `question`, `reference_answer`, `relevant_pages` (list), `relevant_chunk_ids` (list), and optional `category`. Reference answers are plain text in each record. Relevant page/chunk annotations are manually supplied expected targets; retrieval scoring currently uses `relevant_chunk_ids`, while pages are retained in output/report for inspection. If the dataset is not present in another checkout, provide it with `--dataset`.

For top K hybrid chunk IDs, deterministic metrics are precision@K (relevant fraction in returned unique results), recall@K (fraction of annotated relevant chunks retrieved), MRR (reciprocal rank of first relevant chunk), and hit rate@K (whether any relevant chunk appeared). Metrics are averaged over questions with annotated relevant chunk IDs; absent annotations yield N/A. In answer modes, optional Gemini judging estimates answer relevance, faithfulness to retrieved context, context relevance, and correctness against the reference. These estimates can be wrong and are not deterministic metrics. Each JSON result stores metadata (dataset, timestamp, top K, weights, mode), per-question reference/answer/sources/retrieved IDs/pages/errors/metrics, and aggregates. Full and retrieval-only runs use different default result files; `--output` overrides either.

Evaluation gaps: metrics do not score relevant pages directly, answer citations/source correctness, abstention quality, latency, cost, or statistical uncertainty. There is no reranker comparison, rewriting ablation, or per-category aggregate. Add labeled questions and verify LLM judged answers manually before using scores as quality claims.

## Answer format and source behavior

Answerable responses begin with **Answer:**. Gemini may add **Detailed Explanation:** and **Key Points:** for complex questions; simple questions stay short. The application appends **Source References:** from valid cited chunks only, including document filename, physical PDF page, and chunk ID. If there is no retrieved context, the model's exact insufficient-information response is returned without a chat request. If the model does not provide an exact citation matching a retrieved page/chunk, the app returns that same limitation response rather than presenting an uncited claim as grounded.

The document name comes from source metadata; if unavailable, it is shown as `AWS document`. Page numbers are 1-based physical PDF page numbers. Source references verify which retrieved text the model cited, not whether the answer is correct.


