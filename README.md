# Semantic RAG - Advanced Document Search and Retrieval

## Overview

Semantic RAG is a Retrieval-Augmented Generation system that enables intelligent semantic search and question answering across multiple document formats. The system combines vector embeddings, multimodal understanding (text + images), and large language models to provide accurate, source-cited responses.

## Key Features

- Multi-format document support (PDF, DOCX, TXT, MD, RTF, HTML, JPG, PNG, TIFF, and more — see Supported File Formats)
- Multimodal search with OpenCLIP embeddings (text and images)
- OCR extraction for images and scanned documents (Tesseract)
- Real hybrid search (dense + sparse BM25, server-side RRF fusion) with Qdrant — not a placeholder
- LLM integration with Groq (Llama 3.3, Qwen3.6-27B for vision)
- Cross-encoder reranking for improved result relevance
- FastAPI backend with async support
- API key authentication on write/costly endpoints
- Next.js frontend
- Docker containerization
- Kubernetes manifests (namespace, backend, frontend, Qdrant, ingress, configmap, secrets)
- Retrieval and generation evaluation framework with tracked metrics over time
- Response citations with source filename, page, and relevance score

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python |
| Vector Database | Qdrant (dense + sparse named vectors, native RRF fusion) |
| Text Embeddings | Sentence Transformers — default `all-MiniLM-L6-v2`; `BAAI/bge-m3` recommended for multilingual (FR/EN) corpora |
| Sparse Embeddings | fastembed (`Qdrant/bm25`) |
| Image Embeddings | OpenCLIP (LAION checkpoints) |
| Vision LLM | Groq (`qwen/qwen3.6-27b`, preview) |
| Reranking | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` default; `BAAI/bge-reranker-v2-m3` for multilingual) |
| Text LLM | Groq (Llama 3.3 70B) |
| OCR | Tesseract |
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Deployment | Docker, Docker Compose, Kubernetes |

## Architecture

```
Document Upload
    ↓
Document Loader (Unstructured + Tesseract OCR)
    ↓
Chunking (paragraph-based, title-aware, real overlap between chunks)
    ↓
Embeddings (Text: Sentence Transformers / Images: OpenCLIP)
    ↓
Vector Storage (Qdrant — dense + sparse, real hybrid search)
    ↓
Query Processing
    ↓
Retrieval + Reranking
    ↓
LLM Generation (Groq)
    ↓
Response with Citations
```

See `docs/architecture.md` for component-level detail and design rationale.

## Project Structure

```
semantic-search-rag/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── .dockerignore
├── README.md
├── docker-compose.yml
├── LICENSE
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   └── security.py        # X-API-Key auth dependency
│   │   ├── db/
│   │   │   └── vector_store.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   ├── services/
│   │   │   ├── embedding_service.py
│   │   │   ├── image_embedding_service.py
│   │   │   ├── ingestion_service.py
│   │   │   ├── llm_service.py
│   │   │   ├── ocr_service.py
│   │   │   ├── rag_service.py
│   │   │   ├── rerank_service.py
│   │   │   └── retrieval_service.py
│   │   ├── utils/
│   │   │   ├── chunking.py
│   │   │   └── document_loader.py
│   │   └── main.py                # single API entrypoint — all routes live here
│   ├── scripts/
│   │   └── ingest_data.py
│   ├── data/
│   │   ├── raw/                   # gitignored
│   │   ├── processed/
│   │   └── embeddings/
│   └── tests/
│       ├── unit/                  # mocked, no external services required
│       ├── integration/           # requires live Qdrant + GROQ_API_KEY, auto-skips otherwise
│       └── benchmark/             # timing, auto-skips if dependencies/services unavailable
├── evaluation/
│   ├── dataset/
│   │   ├── questions.json
│   │   ├── expected_answers.json
│   │   └── documents/
│   ├── scripts/
│   │   ├── evaluate_retrieval.py
│   │   └── evaluate_generation.py
│   ├── reports/
│   │   └── history/
│   └── README.md
├── docs/
│   ├── api.md
│   ├── architecture.md
│   └── evaluation.md
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── next.config.ts
│   ├── postcss.config.js          # Tailwind v3
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── app/
│   │   └── page.tsx               # uses lib/api.ts (X-API-Key attached)
│   ├── lib/
│   │   └── api.ts                 # centralized axios client
│   └── public/
└── kubernetes/
    ├── namespace.yaml
    ├── backend.yaml
    ├── frontend.yaml
    ├── qdrant.yaml
    ├── ingress.yaml                        # includes per-IP rate limiting
    ├── configmap.yaml
    ├── secret.yaml.example                 # Groq API key template
    └── backend-auth-secret.yaml.example     # backend X-API-Key template
```

**Note:** `app/api/routes/search.py` was removed — it defined a second, parallel RAG entrypoint (its own `IndustrialRAGService` instance behind `@lru_cache`) that was never mounted in `main.py`. Mounting it as-is would have loaded the embedding model, reranker, and Qdrant connections a second time in memory. `main.py` already exposes everything it offered, backed by a single shared instance.

## Supported File Formats

| Category | Formats |
|----------|---------|
| Documents | PDF, DOCX, TXT, MD, RTF, HTML, ODT |
| Spreadsheets | XLS, XLSX, CSV |
| Presentations | PPT, PPTX |
| Images | JPG, JPEG, PNG, GIF, BMP, TIFF |
| Data | JSON, YAML |

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/Ali-EL-AMRIOUI/RAG-research-semantic.git
cd RAG-research-semantic

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in real values (GROQ_API_KEY required; API_KEY optional for local dev)
uvicorn app.main:app --reload

# Frontend setup (new terminal)
cd frontend
npm install
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL and NEXT_PUBLIC_API_KEY if backend API_KEY is set
npm run dev

# Qdrant (Docker)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### Docker Deployment

```bash
docker-compose up --build
curl http://localhost:8000/health   # should return a real JSON status, not hang
```

### Kubernetes Deployment

```bash
# 1. Generate a real backend API key (not the placeholder)
openssl rand -hex 32

# 2. Create real secrets from the templates (never commit the results)
cp kubernetes/secret.yaml.example kubernetes/secret.yaml                       # fill in real GROQ_API_KEY
cp kubernetes/backend-auth-secret.yaml.example kubernetes/backend-auth-secret.yaml   # fill in the key from step 1

# 3. Apply in order
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secret.yaml
kubectl apply -f kubernetes/backend-auth-secret.yaml
kubectl apply -f kubernetes/qdrant.yaml
kubectl apply -f kubernetes/backend.yaml
kubectl apply -f kubernetes/frontend.yaml
kubectl apply -f kubernetes/ingress.yaml

# 4. Verify
kubectl get pods -n rag
```

## API Endpoints

| Method | Endpoint | Auth required | Description |
|--------|----------|:---:|-------------|
| GET | `/health` | – | Health check (verifies real Qdrant connectivity) |
| POST | `/upload` | ✅ `X-API-Key` | Upload and index a document |
| GET | `/documents` | – | List indexed documents |
| DELETE | `/documents` | ✅ `X-API-Key` | Delete a document and its chunks |
| POST | `/ask` | ✅ `X-API-Key` | Ask a question (retrieval + LLM generation) |
| GET | `/stats` | – | Collection statistics |
| POST | `/clear` | ✅ `X-API-Key` | Clear all indexed documents |

Auth is enforced only when `API_KEY` is set in the backend environment — unset locally by default so `localhost` development needs no key. **Must be set before this API is reachable outside your own machine.** Full request/response schemas: `docs/api.md`.

### A note on the frontend and the API key

`NEXT_PUBLIC_API_KEY` is inlined into the browser bundle at build time — it stops anonymous bots/scanners hitting the API directly, but it is not hidden from someone inspecting the frontend's network requests. `kubernetes/ingress.yaml` adds a per-IP rate limit as a complementary layer. For real protection against an authenticated app's own users, route sensitive calls through a server-side proxy or move to per-user login instead of a shared key.

## Usage Examples

### Upload Document

```bash
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: your_api_key" \
  -F "file=@document.pdf"
```

### Ask Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"question": "What is this document about?"}'
```

### Response Format

```json
{
  "answer": "This document discusses...",
  "source_documents": [
    {
      "content": "Extracted text chunk...",
      "filename": "document.pdf",
      "page": 5,
      "score": 0.89
    }
  ],
  "processing_time_ms": null
}
```

## Testing

```bash
cd backend

# Unit tests — mocked, no external services needed
python -m pytest tests/unit/ -v

# Integration tests — requires Qdrant running and GROQ_API_KEY set;
# auto-skips individual tests when a dependency is missing
python -m pytest tests/integration/ -v

# Benchmarks — timing, same auto-skip behavior (not yet implemented, TODO)
python -m pytest tests/benchmark/ -v
```

## Evaluation

Retrieval and generation quality are tracked over time, not assumed:

```bash
cd backend
python ../evaluation/scripts/evaluate_retrieval.py
python ../evaluation/scripts/evaluate_generation.py
```

Reports land in `evaluation/reports/` (`*_latest.json` plus a timestamped
`history/`). Metric definitions and methodology: `docs/evaluation.md`.
Usage details: `evaluation/README.md`.

**Note:** these scripts run against whatever collection `QDRANT_COLLECTION_NAME` currently points to — typically the same collection as production. Re-running them after the indexed documents have changed will produce metrics that no longer match `evaluation/reports/*_latest.json`; re-index the documents referenced in `evaluation/dataset/expected_answers.json` first if you need comparable numbers.

## Environment Variables

### Backend (`backend/.env`)

```env
# LLM
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=documents

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
EMBEDDING_USE_CACHE=True

# Retrieval
TOP_K=10
TOP_N=3
SCORE_THRESHOLD=0.4
USE_HYBRID_SEARCH=True
USE_RERANKER=True
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_N=3

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# API
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Auth — leave empty for local dev (auth disabled with a startup warning)
API_KEY=

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=standard
```

### Frontend (`frontend/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=
```

Switching `EMBEDDING_MODEL` to a model with a different output dimension
(e.g. `BAAI/bge-m3`, 1024 dims vs MiniLM's 384) requires re-ingesting
already-indexed documents — the Qdrant collection is created with a fixed
dimension at first use.

## Future Improvements

- Liveness/readiness probes on `kubernetes/backend.yaml` and `kubernetes/qdrant.yaml`, tied to `/health`
- Qdrant healthcheck + `depends_on` in `docker-compose.yml`
- Confirm or implement the Prometheus/Grafana monitoring referenced in earlier planning — not yet verified as present
- Implement `tests/benchmark/` (currently TODO stubs — `embed_batch()` throughput, `search_hybrid()` vs `search_vector()` latency)
- Move from shared-secret API key to per-user auth (OAuth2/JWT) if this becomes a multi-user product
- Server-side proxy for frontend API calls, so the API key never reaches the browser
- LLM-judge option for `evaluate_generation.py` when answer-quality nuance matters more than evaluation cost
- Document versioning
- Multi-vector-database support (Weaviate, Pinecone) if a second backend is ever needed
- Populate `docs/api.md`, `docs/architecture.md`, `docs/evaluation.md` (currently empty placeholders)

## License

MIT License

## Author

ALI EL AMRIOUI - GitHub: https://github.com/Ali-EL-AMRIOUI