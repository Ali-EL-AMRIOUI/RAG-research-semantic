# Architecture

## Pipeline

Upload → Document Loader (Unstructured + Tesseract OCR) / Groq Vision captioning for images
→ Chunking (paragraph-based, title-aware, real overlap)
→ Embedding (BAAI/bge-m3 for all text, including image captions)
→ Qdrant (dense + sparse named vectors)

Query → Embedding → Hybrid search (dense+sparse, server-side RRF fusion)
→ Cross-encoder rerank
→ LLM (Groq, grounded prompt + last 6 turns of history)
→ Answer + cited sources


## Ingestion

**Text documents** (PDF, DOCX, TXT, MD, HTML, ...): loaded via `unstructured`, with Tesseract OCR fallback for scanned content. Chunked by paragraph, tracking a markdown title-path per chunk, with real character overlap carried between consecutive chunks (not just stored and ignored).

**Images**: two parallel, independent paths.
1. Groq Vision (`qwen/qwen3.8-27b`) generates a factual caption / OCR transcription. This text is chunked and embedded exactly like a document, so an image becomes findable through ordinary semantic queries ("what does the photo of X show").
2. The raw image is also embedded via OpenCLIP into a separate `documents_images` collection, for pure visual similarity (not currently used by the chat flow — reserved for a future "find similar image" feature).

If Groq Vision captioning fails, ingestion fails loudly (no silent fallback to a placeholder string) — a previous version stored `"Image file: {filename}"` as the indexed text on failure, which made the image permanently unsearchable with no visible error.

## Embedding model choice

`BAAI/bge-m3` (1024-dim) replaced the original `all-MiniLM-L6-v2` (384-dim) default. MiniLM is effectively English-only; for an Arabic/French corpus it failed to retrieve Arabic-language chunks for queries in either language. bge-m3 is natively multilingual and solves this without a separate translation step.

Changing `EMBEDDING_MODEL` to a model with a different output dimension requires wiping the Qdrant collection (`POST /clear`) and re-ingesting — the collection's vector size is fixed when first created.

## Retrieval

Each point in Qdrant carries two named vectors: `dense` (bge-m3) and `sparse` (BM25 via `fastembed`). Hybrid search issues a single `query_points` call with both as prefetch branches, fused server-side via Reciprocal Rank Fusion — not merged client-side.

Top candidates are reranked with a cross-encoder before being handed to the LLM, trading a small latency cost for materially better precision on what the LLM actually sees.

## Generation

The LLM prompt (see `rag_service.py::run()`):
- Receives up to the last 6 conversation turns as context (not just the current question)
- Must answer in the language of the question, regardless of the source document's language
- Must quote the exact supporting passage before stating a specific fact (date, name, number)
- Must say explicitly that the information wasn't found, rather than guess, use outside knowledge, or restate a claim the user has just disputed
- Is told to redirect system/meta questions ("how many documents") to the sidebar UI rather than answering from retrieved context

This grounding is currently enforced by prompt instructions only — there is no separate automated check that a model-cited quote genuinely appears in the retrieved context. See Known Limitations.

## Deployment Notes

**Memory.** The backend needs at least `4Gi` of memory, in both `docker-compose.yml` and `kubernetes/backend.yaml`. `bge-m3` alone uses roughly 1.5Gi once loaded; together with the cross-encoder reranker and FastAPI's own overhead, a `2Gi` limit gets the container OOM-killed partway through startup.

**Readiness/liveness probes (Kubernetes).** `backend.yaml`'s probes target `/health` with an `initialDelaySeconds` of 120–180s, because `bge-m3`'s cold load on CPU takes up to that long on first start — a shorter delay causes Kubernetes to restart a pod that was simply still loading, not actually unhealthy.

**Model cache.** Both Docker Compose (a named volume) and the Kubernetes manifest (an `emptyDir`) persist the downloaded HuggingFace weights so a restart doesn't re-download `bge-m3` from scratch. The Kubernetes `emptyDir` survives a pod restart on the same node, but not a reschedule to a different node — see Known Limitations.

## Known Limitations

- **No tool-routing for meta questions.** "How many files do you have" is hardcoded to a static refusal message rather than querying `/documents` directly — a real implementation would route this to a tool call.
- **No automated grounding verification.** The prompt asks the model to quote evidence; nothing programmatically checks the quote is real.
- **Single global collection.** No per-user or per-organization data isolation (no multi-tenancy).
- **`docker-compose` v1 quirk.** The legacy Python `docker-compose` (not the `docker compose` plugin) occasionally fails to recreate containers after certain config changes (`KeyError: 'ContainerConfig'`). Workaround: `docker compose down`, then targeted `docker rm` of the affected container, then `up` again.
- **CPU cold start is slow.** Loading `bge-m3` + the reranker takes roughly 10–15 minutes on CPU on first start; the optional `gpu` Compose profile reduces this to under a minute on a machine with an NVIDIA GPU.
- **Kubernetes model cache isn't node-independent.** The `emptyDir` cache volume in `backend.yaml` is lost if the pod is rescheduled to a different node (node failure, cluster autoscaling) — the next start pays the full 10–15 minute cold load again. A `PersistentVolumeClaim`, as already used for Qdrant's storage, would fix this for a real deployment.