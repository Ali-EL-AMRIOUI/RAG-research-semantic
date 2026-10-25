# Backend — Semantic RAG API

FastAPI service: document ingestion, hybrid retrieval, grounded LLM generation. See the [root README](../README.md) for the full picture and [`docs/`](../docs/) for architecture/API details.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY; set QDRANT_HOST=localhost if Qdrant isn't in Docker
uvicorn app.main:app --reload
```

Requires a running Qdrant instance (`docker run -d --name qdrant -p 6333:6333 -v "$(pwd)/../qdrant_storage:/qdrant/storage" qdrant/qdrant`) and Tesseract (`apt install tesseract-ocr tesseract-ocr-fra`) if not using the provided Dockerfile.

## Structure
app/
├── core/ # config, logging, X-API-Key auth
├── db/ # Qdrant client wrapper (hybrid search, CRUD)
├── models/ # Pydantic schemas
├── services/ # embedding, ingestion, rerank, LLM, OCR, the RAG orchestrator
├── utils/ # chunking, document loading
└── main.py # all routes
scripts/
└── ingest_data.py # CLI ingestion, shares IngestionService with the API
tests/
├── unit/ # mocked, no external services
├── integration/ # needs live Qdrant + GROQ_API_KEY, auto-skips otherwise
└── benchmark/ # timing


## Testing

```bash
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v   # needs Qdrant + GROQ_API_KEY
```

## Linting

```bash
ruff check .
```

Project-specific ignores (broad exception handling, FastAPI's `Depends()`/`File()` default-argument pattern) are documented in `ruff.toml`.

## Key environment variables

See [root README](../README.md#environment-variables) for the full list. The ones most likely to need changing:

- `QDRANT_HOST` — `qdrant` inside Docker Compose, `localhost` when running the backend outside Docker
- `EMBEDDING_MODEL` — changing this requires `POST /clear` + re-ingesting (vector dimension is fixed at collection creation)
- `API_KEY` — empty disables auth (local dev only); must be set before exposing the API

## Notes

- `main.py` is the single entrypoint; there's intentionally no separate router module — an earlier `api/routes/search.py` defined a second, never-mounted RAG pipeline and was removed to avoid loading the embedding model and Qdrant connections twice.
- Image uploads go through two paths: Groq Vision captioning (indexed as searchable text) and OpenCLIP embedding (separate `_images` collection, visual similarity only).