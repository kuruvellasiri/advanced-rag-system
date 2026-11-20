# Windows PowerShell Command Reference

Run commands from the project root. The default embedding configuration is local Sentence Transformers. Gemini remains available by switching the embedding settings in `.env`. The answer-generation model is independently configured with `GEMINI_CHAT_MODEL` and still requires a Gemini API key.

## Environment and install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`python -m pip install -r requirements.txt` installs Sentence Transformers and the existing RAG dependencies. Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env` for answer generation and Gemini evaluation. Local embeddings do not need this key. `sentence-transformers/all-MiniLM-L6-v2` downloads the first time it is loaded and is then cached locally.

Set the default embedding provider in `.env`:

```text
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
```

To use Gemini embeddings instead, set both provider and model:

```text
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001
```

Do not change the embedding provider/model after indexing and expect the existing collection to be reused. Each provider/model resolves to its own Chroma collection. The default local collection is `aws_content_local_sentence_transformers_all_minilm_l6_v2`; the existing Gemini collection `aws_content` remains preserved. A new provider/model collection is populated once; unchanged chunks are skipped on subsequent runs.

## Prepare and index

```powershell
python -m src.preprocess
python -m src.vectorstore.index_chunks
```

- `python -m src.preprocess` runs `src/preprocess.py`: extract the configured PDF, clean text, route document sections, and create content/navigation/history chunks. It must run before indexing if chunk JSON is missing or stale.
- `python -m src.vectorstore.index_chunks` runs `src/vectorstore/index_chunks.py`: embed content chunks with the selected provider and upsert them in that provider/model's Chroma collection. It prints provider, model, collection, progress, and indexing totals. On the first local collection run it generates vectors for all current chunks. Later runs only embed new or changed text. This uses no Gemini embedding API when `EMBEDDING_PROVIDER=local`.
- `python -m src.preprocess --chunk-size 800 --chunk-overlap 150` changes chunking parameters while running the same combined preprocessing stages.
- `python -m src.vectorstore.index_chunks --rebuild` forces vectors for all chunks in the selected collection. Avoid this for routine updates.
- The index CLI retains IDs that no longer occur in the chunk JSON; it does not delete existing collection records. `--rebuild` re-embeds current chunks without deleting stale records.

No standalone commands exist for extraction-only, cleanup-only, chunking-only, or embedding-only. They are parts of the combined commands above.

## Ask and retrieve

```powershell
python -m src.retrieval.cli
python -m src.retrieval.cli --question "What is Amazon Lightsail?"
python -m src.retrieval.cli --question "What is Amazon Lightsail?" --answer-only
python -m src.retrieval.cli --question "What is Amazon Lightsail?" --top-k 5 --vector-weight 1 --keyword-weight 1 --compare
```

All commands run `src/retrieval/cli.py`. They require preprocessing, a populated selected Chroma collection, and a Gemini API key for answer generation. The first run requires network access to download the local model if it is not cached. The CLI combines vector results with BM25 and weighted RRF; `--compare` prints the separate vector and keyword rankings. No standalone BM25, vector-search, hybrid-search, or RRF command is implemented. There is no reranking command or reranker implementation.

The answer-generation model remains Gemini regardless of the selected embedding provider. The embedding provider only generates document and query vectors; it does not configure or replace the chat model.

### Optional MiniLM reranking

Set these values in `.env` to enable the measured production reranker:

```text
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_CANDIDATE_COUNT=10
RERANKER_TOP_K=5
```

Then use the same CLI commands above, for example:

```powershell
python -m src.retrieval.cli --question "What is Amazon Lightsail?" --vector-weight 0.5 --keyword-weight 0.5
python -m src.retrieval.cli --evaluate --retrieval-only --top-k 5 --vector-weight 0.5 --keyword-weight 0.5 --output data/evaluation/production_reranker_retrieval.json
```

The CLI first retrieves ten hybrid candidates, applies weighted RRF using the supplied weights, and the CrossEncoder returns the top five for answer context. The default candidate pool is ten; the model is loaded only when reranking is enabled and reused while the CLI process is running. The first enabled run downloads the model if it is not cached.

To disable reranking, set `RERANKER_ENABLED=false` in `.env` and run the same CLI command. This uses the existing hybrid RRF results without loading the CrossEncoder. If the setting is absent, reranking is also disabled.

The experiment command `python -m src.evaluation.reranking_experiment` remains a separate comparison path and does not change production settings or the Chroma index.

## Evaluation

```powershell
python -m src.retrieval.cli --evaluate
python -m src.retrieval.cli --evaluate --no-llm-eval
python -m src.retrieval.cli --evaluate --retrieval-only
python -m src.retrieval.cli --evaluate --limit 5 --output data/evaluation/quick_run.json
```

These commands run `src/retrieval/cli.py` against `data/evaluation/aws_overview_eval.json`. Full evaluation generates answers and uses Gemini as an optional judge. `--no-llm-eval` skips judging but still generates answers. `--retrieval-only` skips both generation and judging. Results default to `data/evaluation/latest_results.json` or `data/evaluation/retrieval_only_results.json`; `--output` overrides the destination. The CLI currently checks for a Gemini API key at startup even in retrieval-only mode.

## Tests

```powershell
python -m pytest tests/test_retrieval.py -q
python -m pytest tests/test_rag.py -q
python -m pytest -q
```

These run retrieval/storage tests, answer-generation tests, and the full test suite. Provider behavior is tested with injected local model doubles, so tests do not download the model or call Gemini APIs.
