# Semantic RAG — Multilingual Document Search & Q&A

Retrieval-Augmented Generation system for semantic search and multi-turn question answering across documents in Arabic, French, and English. Combines multilingual embeddings, multimodal understanding (text + images), and Groq-hosted LLMs to give accurate, source-cited answers — and explicitly says "not found" rather than guessing when the indexed documents don't contain the answer.

## Why this exists

General-purpose assistants (ChatGPT, Claude, Gemini) handle a handful of uploaded files well, but don't scale to hundreds of permanent documents, don't guarantee Arabic/French retrieval quality, and don't keep your documents off third-party servers. This project targets that gap: organizations (schools, administrations, law practices) with large, bilingual document archives who need fast, cited, private search.

## Key Features

- **Multilingual retrieval** (`BAAI/bge-m3`) — correctly retrieves Arabic content for French queries and vice versa, unlike English-centric embedding models
- **Multi-turn chat** — conversation history is passed to the LLM on every turn, not just the latest question
- **Multi-file upload** with per-file success/failure reporting
- **Grounded answers** — the LLM must quote the supporting passage before stating a fact, and refuses rather than guesses when context is missing
- **Multimodal** — images are both OCR/captioned (Groq Vision, `qwen/qwen3.8-27b`) into searchable text and embedded visually (OpenCLIP)
- **Real hybrid search** — dense + sparse BM25 fusion (RRF), server-side in Qdrant
- **Cross-encoder reranking** for precision on the chunks actually shown to the LLM
- Docker (CPU by default, optional GPU profile), Kubernetes manifests with liveness/readiness probes, CI, evaluation harness with tracked metrics

## Quick Start

```bash
git clone https://github.com/Ali-EL-AMRIOUI/RAG-research-semantic.git
cd RAG-research-semantic
docker compose up -d --build
```

Then open `http://localhost:3000`. First start loads the embedding model (~10–15 min on CPU, ~30s with `--profile gpu`); subsequent starts are fast (cached weights, persistent volume).

Detailed setup for each part: [`backend/README.md`](backend/README.md) · [`frontend/README.md`](frontend/README.md)

## Kubernetes Deployment

```bash
# 1. Build images and load them into the cluster (minikube example — adjust for your cluster)
docker build -t rag-backend:latest -f backend/Dockerfile .
docker build -t rag-frontend:latest ./frontend
minikube image load rag-backend:latest
minikube image load rag-frontend:latest

# 2. Create real secrets from the committed templates — never commit the results
cp kubernetes/secret.yaml.example kubernetes/secret.yaml                         # fill in GROQ_API_KEY
cp kubernetes/backend-auth-secret.yaml kubernetes/backend-auth-secret.real.yaml   # fill in: openssl rand -hex 32

# 3. Apply in order
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secret.yaml
kubectl apply -f kubernetes/backend-auth-secret.real.yaml
kubectl apply -f kubernetes/qdrant.yaml
kubectl apply -f kubernetes/backend.yaml
kubectl apply -f kubernetes/frontend.yaml
kubectl apply -f kubernetes/ingress.yaml

# 4. Verify
kubectl get pods -n rag
kubectl logs -n rag -l app=backend -f   # first start: ~10-15 min while bge-m3 loads
```

See [`docs/architecture.md`](docs/architecture.md#deployment-notes) for why the backend needs 4Gi of memory and what the readiness probe's delay accounts for.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (Python) |
| Vector DB | Qdrant — dense + sparse named vectors, native RRF |
| Text Embeddings | `BAAI/bge-m3` (1024-dim, multilingual) |
| Sparse Embeddings | fastembed (`Qdrant/bm25`) |
| Image Embeddings | OpenCLIP (LAION) |
| Vision LLM | Groq `qwen/qwen3.8-27b` |
| Text LLM | Groq `openai/gpt-oss-120b` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| OCR | Tesseract |
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Deployment | Docker Compose, Kubernetes |

## Architecture

Full detail: [`docs/architecture.md`](docs/architecture.md)

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — component-level design, why bge-m3, deployment notes, known limitations
- [`docs/api.md`](docs/api.md) — endpoint reference
- [`docs/evaluation.md`](docs/evaluation.md) — retrieval/generation metrics methodology
- [`backend/README.md`](backend/README.md) — backend setup, env vars, testing
- [`frontend/README.md`](frontend/README.md) — frontend setup, scripts

## Project Structure

semantic-search-rag/
├── backend/ # FastAPI service — see backend/README.md
├── frontend/ # Next.js UI — see frontend/README.md
├── evaluation/ # Retrieval/generation quality tracking
├── docs/ # Architecture, API, evaluation docs
├── kubernetes/ # K8s manifests
└── docker-compose.yml


## Supported File Formats

PDF, DOCX, TXT, MD, RTF, HTML, ODT · XLS, XLSX, CSV · PPT, PPTX · JPG, PNG, GIF, BMP, TIFF · JSON, YAML

## Future Improvements

- Route system/meta questions ("how many files?") to a real tool call instead of a hardcoded refusal
- Multi-tenancy (per-organization data isolation) for a multi-customer deployment
- Automated grounding check (verify a cited quote actually appears in context) rather than relying on prompt instructions alone
- Per-user auth (OAuth2/JWT) instead of a shared API key
- Evaluate retrieval quality specifically on Arabic-language queries
- Back the Kubernetes HuggingFace cache with a PersistentVolumeClaim instead of `emptyDir`

## License

MIT — see [`LICENSE`](LICENSE)

## Author

Ali El Amrioui — [github.com/Ali-EL-AMRIOUI](https://github.com/Ali-EL-AMRIOUI)