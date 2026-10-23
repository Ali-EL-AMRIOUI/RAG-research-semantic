"""
Core RAG service: retrieval + LLM generation.

Changes vs the previous version:
- ingest() used to be a stub (`print(...); return 1`) that never loaded,
  chunked, or indexed anything, while IngestionService (a separate file)
  did the real work. Two ingestion paths, only one of them real. Fixed:
  ingest() now delegates to IngestionService (text/PDF) or
  ImageEmbeddingService (images).
- The reranker was loaded into memory in __init__ but never called in
  search()/run() - paying the load cost for nothing. Fixed: called in
  search() when use_reranker=True.
- run()/search() did not accept a per-request use_hybrid/use_reranker
  override; added as optional parameters.
- OpenAI CLIP (2021) was hardcoded here, duplicating
  image_embedding_service.py, which did nothing. Replaced by a single
  real service: ImageEmbeddingService (OpenCLIP/LAION), used here by
  composition.
- print() -> logger everywhere, consistent with the rest of the repo.
- _store_image_point() sent an unnamed vector (`vector=embedding`) to a
  Qdrant collection configured with named vectors -> 400 Bad Request
  "Not existing vector name" on every image upload. Fixed:
  `vector={"dense": embedding}`.
- NEW: the image payload had no semantic content
  (`"text": f"Image file: {filename}"`), so it was invisible to the
  reranker and to normal hybrid search - images were never retrieved by
  a text question. A real description is now generated at ingestion time
  via Groq Vision (qwen/qwen3.6-27b, OCR + caption) and indexed in the
  main TEXT collection, benefiting from the same hybrid search + reranker
  pipeline already validated for PDFs. CLIP is kept as a secondary
  channel (pure visual similarity, not text-based search).
"""
import base64
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from qdrant_client.http import models

from app.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    QDRANT_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    RERANKER_MODEL,
    SCORE_THRESHOLD,
    TOP_K,
    TOP_N,
    USE_RERANKER,
)
from app.core.logging import PerformanceLogger
from app.db.vector_store import IndustrialVectorStore
from app.services.embedding_service import IndustrialEmbeddingService
from app.services.image_embedding_service import ImageEmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.rerank_service import IndustrialReranker

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger("rag_service")

# Previously hardcoded here (TOP_K=10, TOP_N=5) instead of imported from
# config.py, which already defines TOP_K=10, TOP_N=3 - silent divergence:
# .env had no effect on these values.
DEFAULT_TOP_K = TOP_K
DEFAULT_TOP_N = TOP_N
DEFAULT_SCORE_THRESHOLD = SCORE_THRESHOLD

VISION_MODEL = "qwen/qwen3.8-27b"  # Groq multimodal model (OCR + caption + JSON mode)

load_dotenv()


class IndustrialRAGService:

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        collection_name: str = QDRANT_COLLECTION_NAME,
        use_reranker: bool = USE_RERANKER,
        use_cache: bool = True
    ):
        logger.info("Initializing RAG service...")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_collection_name = collection_name
        self.image_collection_name = f"{collection_name}_images"
        self.use_reranker = use_reranker
        self.default_top_k = DEFAULT_TOP_K
        self.default_top_n = DEFAULT_TOP_N

        # Text pipeline: a single ingestion/search path, shared with the
        # CLI script and the upload API.
        self.ingestion_service = IngestionService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            use_cache=use_cache,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=self.text_collection_name,
        )
        self.embedder: IndustrialEmbeddingService = self.ingestion_service.embedder
        self.vector_store: IndustrialVectorStore = self.ingestion_service.vector_store
        logger.info(f"Text pipeline ready: model={embedding_model}, dim={self.embedder.get_dimension()}")

        # Image pipeline: lazily loaded on first real use, to avoid paying
        # the GPU/memory cost if the app only ever serves text.
        self._image_embedder: ImageEmbeddingService | None = None
        self.image_vector_store = IndustrialVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=self.image_collection_name,
            vector_dim=512,  # OpenCLIP ViT-B-32 dimension, see image_embedding_service.py
            use_hybrid_search=False
        )

        # Groq Vision client, lazily loaded (same pattern as image_embedder
        # above) - no cost if no image is ever uploaded.
        self._vision_client: Groq | None = None

        self.llm_service = LLMService(model_name=llm_model, temperature=LLM_TEMPERATURE)
        if not self.llm_service.is_available:
            logger.warning("LLM unavailable - check GROQ_API_KEY")

        self.reranker: IndustrialReranker | None = None
        if use_reranker:
            try:
                self.reranker = IndustrialReranker(model_name=RERANKER_MODEL)
                logger.info(f"Reranker loaded: {RERANKER_MODEL}")
            except Exception as e:
                logger.warning(f"Reranker not available: {e}")
                self.use_reranker = False

        logger.info("RAG service ready")

    @property
    def image_embedder(self) -> ImageEmbeddingService:
        if self._image_embedder is None:
            self._image_embedder = ImageEmbeddingService()
        return self._image_embedder

    @property
    def vision_client(self) -> Groq:
        if self._vision_client is None:
            self._vision_client = Groq(api_key=GROQ_API_KEY)
        return self._vision_client

    def _is_image_file(self, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    def _caption_image(self, file_stream: BytesIO, filename: str) -> str:
        """
        Generate a real text description of the image via Groq Vision
        (OCR + caption). Replaces the "Image file: {filename}" placeholder,
        which had no semantic content comparable to a query - hence
        invisible to the reranker regardless of score_threshold.

        Falls back to the old placeholder if the Vision call fails, rather
        than failing the whole upload.
        """
        ext = Path(filename).suffix.lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext

        file_stream.seek(0)
        b64_image = base64.b64encode(file_stream.read()).decode("utf-8")
        file_stream.seek(0)

        try:
            completion = self.vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe precisely what this image contains, in French. "
                                    "If it contains text (document, screenshot, poster, form), "
                                    "transcribe it fully and faithfully. "
                                    "Be factual and thorough, no speculation."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/{mime};base64,{b64_image}"}
                            }
                        ]
                    }
                ],
                temperature=0.2,
                max_completion_tokens=1024,
            )
            caption = (completion.choices[0].message.content or "").strip()
            if not caption:
                raise ValueError("Empty caption returned by vision model")
            logger.info(f"Vision caption generated for {filename} ({len(caption)} characters)")
            return caption
        except Exception as e:
            logger.error(f"Vision captioning failed for {filename}: {e}")
            raise 

    def _store_image_point(self, filename: str, embedding: list[float]) -> int:
        point_id = str(uuid.uuid4())
        try:
            self.image_vector_store.client.upsert(
                collection_name=self.image_collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={"dense": embedding},
                        payload={"filename": filename, "modality": "image", "text": f"Image file: {filename}"}
                    )
                ]
            )
            logger.info(f"Image point stored in {self.image_collection_name}: {filename}")
            return 1
        except Exception as e:
            logger.error(f"Qdrant upsert failed for image {filename}: {e}")
            return 0


    def ingest(
        self,
        file_stream: BytesIO,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest an image or a document."""
        try:
            logger.info("Ingestion started for %s", filename)

            if self._is_image_file(filename):
                caption = self._caption_image(file_stream, filename)

                indexed = self.ingestion_service.ingest_text(
                    text=caption,
                    metadata={**(metadata or {}), "filename": filename, "modality": "image"},
                    source=filename
                )

                file_stream.seek(0)
                embedding = self.image_embedder.embed_image(file_stream)
                self._store_image_point(filename, embedding)

                return indexed

            return self.ingestion_service.ingest_pdf_stream(
                file_stream,
                filename,
                metadata=metadata,
            )

        except Exception as e:
            logger.error("Ingestion error for %s: %s", filename, e)
            return 0


    def search(
        self,
        query: str,
        limit: int | None = None,
        use_hybrid: bool | None = None,
        use_reranker: bool | None = None
    ) -> list[dict[str, Any]]:
        limit = limit or self.default_top_k
        apply_hybrid = use_hybrid if use_hybrid is not None else self.vector_store.use_hybrid_search
        apply_reranker = (use_reranker if use_reranker is not None else self.use_reranker) and self.reranker is not None

        import time
        search_start = time.perf_counter()

        query_vector = self.embedder.embed_query(query)

        if apply_hybrid:
            text_results = self.vector_store.search_hybrid(
                query_vector=query_vector, query_text=query, limit=limit, score_threshold=DEFAULT_SCORE_THRESHOLD
            )
        else:
            text_results = self.vector_store.search_vector(
                query_vector=query_vector, limit=limit, score_threshold=DEFAULT_SCORE_THRESHOLD
            )

        image_results = []
        try:
            image_query_vector = self.image_embedder.embed_text_query(query)
            image_results = self.image_vector_store.search_vector(
                query_vector=image_query_vector, limit=max(limit // 2, 1), score_threshold=DEFAULT_SCORE_THRESHOLD
            )
        except Exception as e:
            logger.warning(f"Image search failed: {e}")

        all_results = []
        for r in text_results:
            all_results.append({
                "text": r.content,
                "filename": r.metadata.get("filename", "unknown"),
                "score": r.score,
                "page": r.metadata.get("page", 0),
                "metadata": r.metadata
            })
        for r in image_results:
            all_results.append({
                "text": r.metadata.get("text", "Image file"),
                "filename": r.metadata.get("filename", "image"),
                "score": r.score,
                "page": 0,
                "metadata": r.metadata
            })

        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:limit]

        search_duration_ms = (time.perf_counter() - search_start) * 1000
        perf_logger.log_search_time(query=query, results=len(all_results), duration_ms=search_duration_ms)

        # The reranker used to be instantiated in __init__ but never
        # actually called here - paying the model load cost for nothing.
        if apply_reranker and all_results:
            rerank_start = time.perf_counter()
            all_results = self.reranker.rerank_search_results(
                query=query, search_results=all_results, top_n=min(len(all_results), self.default_top_n)
            )
            rerank_duration_ms = (time.perf_counter() - rerank_start) * 1000
            perf_logger.log_rerank_time(count=len(all_results), duration_ms=rerank_duration_ms)

        return all_results

    def run(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
        use_hybrid: bool | None = None,
        use_reranker: bool | None = None,
        conversation_history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        logger.info(f"Processing query: {query}")

        top_k = top_k or self.default_top_k
        top_n = top_n or self.default_top_n

        chunks = self.search(query, limit=top_k, use_hybrid=use_hybrid, use_reranker=use_reranker)

        sources_for_response = []
        has_context = bool(chunks)
        if has_context:
            top_chunks = chunks[:top_n]
            context_parts = [f"[Source: {c.get('filename', 'Unknown')}] {c.get('text', '')}" for c in top_chunks]
            context = "\n\n---\n\n".join(context_parts)

            for c in top_chunks:
                sources_for_response.append({
                    "content": c.get('text', ''),
                    "filename": c.get('filename', 'Unknown'),
                    "page": c.get('page', 0),
                    "score": c.get('score', 0)
                })
        else:
            context = ""

        history_block = ""
        if conversation_history:
            history_lines = []
            for turn in conversation_history[-6:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                history_lines.append(f"{role.upper()}: {content}")
            history_block = "CONVERSATION HISTORY (for context only, do not treat as source documents):\n" + "\n".join(history_lines) + "\n\n"

        prompt = f"""You are a precise document search assistant. Your ONLY source of factual information is the CONTEXT below, extracted from the user's indexed documents.

STRICT RULES — follow exactly, no exceptions:
1. Always respond in the same language as the user's question (a French question gets a French answer, an English question gets an English answer), regardless of the language of the CONTEXT below.
2. If the user sends a greeting or casual small talk (hello, thanks, how are you), respond briefly and naturally. Do not mention documents.
3. For any factual question, answer ONLY using the CONTEXT below. Before stating a precise fact (a date, a place, a name, a number), quote in quotation marks the exact passage from the CONTEXT that proves it. If you cannot produce that exact quote, say the information is not clearly available rather than stating it anyway. Cite the source filename for every claim.
4. If the CONTEXT does not contain the answer, say (in the user's language): "Je n'ai pas trouvé cette information dans les documents indexés." / "I couldn't find this information in the indexed documents."
5. If the user asks about the system itself (how many files, what documents exist, system capabilities) rather than document content, say (in the user's language): "Je ne peux répondre qu'aux questions portant sur le contenu de vos documents indexés. Pour le nombre de documents ou leur gestion, utilisez la liste de documents dans la barre latérale." / "I can only answer questions about the content of your indexed documents. For document counts or management, please use the document list in the sidebar."
6. Never fabricate a source filename or page number that is not in the CONTEXT.
7. If the user indicates that a previous answer was wrong, never repeat that claim as stated. Say clearly that you cannot confirm that information with certainty from the documents, rather than repeating the contested claim.

{history_block}CONTEXT FROM KNOWLEDGE BASE:
{context if has_context else "(no relevant documents found for this query)"}

USER QUESTION: {query}
"""

        try:
            import time
            llm_start = time.perf_counter()
            answer = self.llm_service.generate(prompt)
            llm_duration_ms = (time.perf_counter() - llm_start) * 1000

            usage = self.llm_service.last_usage
            perf_logger.log_llm_time(
                prompt_tokens=usage.get("input_tokens", 0),
                response_tokens=usage.get("output_tokens", 0),
                duration_ms=llm_duration_ms
            )
            logger.info(f"Response generated ({len(answer)} characters, {llm_duration_ms:.0f}ms)")
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = f"I'm sorry, I encountered an error generating a response: {e}"

        return {
            "answer": answer,
            "source_documents": sources_for_response,
            "sources_details": chunks[:top_n] if chunks else [],
            "processing_time_ms": None
        }
    def get_all_documents(self) -> list[dict[str, Any]]:
        documents = {}

        def process_collection(collection_name: str, vector_store: IndustrialVectorStore):
            try:
                next_offset = None
                while True:
                    scroll_result = vector_store.client.scroll(
                        collection_name=collection_name, limit=1000, offset=next_offset,
                        with_payload=True, with_vectors=False
                    )
                    batch, next_offset = scroll_result
                    for point in batch:
                        filename = point.payload.get("filename") or point.payload.get("source") or "unknown"
                        if filename not in documents:
                            documents[filename] = {"name": filename, "chunks": 0}
                        documents[filename]["chunks"] += 1
                    if next_offset is None:
                        break
            except Exception as e:
                logger.error(f"Error scrolling {collection_name}: {e}")

        process_collection(self.text_collection_name, self.vector_store)
        process_collection(self.image_collection_name, self.image_vector_store)

        result = list(documents.values())
        logger.info(f"Documents found: {len(result)}")
        return result

    def delete_document(self, filename: str) -> int:
        deleted_count = self.vector_store.delete_documents(filter_conditions={"filename": filename})
        deleted_count += self.image_vector_store.delete_documents(filter_conditions={"filename": filename})
        if deleted_count == 0:
            logger.info(f"No chunks found for: {filename}")
        return deleted_count

    def get_stats(self) -> dict[str, Any]:
        try:
            text_stats = self.vector_store.get_collection_stats()
            image_stats = self.image_vector_store.get_collection_stats()
            return {
                "text_collection": {
                    "name": self.text_collection_name,
                    "total_points": text_stats.get("points_count"),
                },
                "image_collection": {
                    "name": self.image_collection_name,
                    "total_points": image_stats.get("points_count"),
                },
                "status": "healthy"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    def clear_collection(self) -> int:
        deleted_count = self.vector_store.clear_collection()
        deleted_count += self.image_vector_store.clear_collection()
        return deleted_count