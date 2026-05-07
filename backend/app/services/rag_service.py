import os
import tempfile
import uuid
import traceback
from io import BytesIO
from typing import List, Optional, Dict, Any
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import clip
import torch
from PIL import Image

from app.utils.document_loader import IndustrialDocumentLoader
from app.utils.chunking import IndustrialChunker
from app.services.embedding_service import IndustrialEmbeddingService
from app.db.vector_store import IndustrialVectorStore
from app.services.rerank_service import IndustrialReranker
from app.core.config import (
    GROQ_API_KEY, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP,
    USE_RERANKER, RERANKER_MODEL
)

TOP_K = 10
TOP_N = 5

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
        print("Initializing RAG service...")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_collection_name = collection_name
        self.image_collection_name = f"{collection_name}_images"
        self.use_reranker = use_reranker
        self.top_k = TOP_K
        self.top_n = TOP_N
        
        self.embedder = IndustrialEmbeddingService(
            model_name=embedding_model,
            use_cache=use_cache,
            batch_size=32
        )
        print(f"Text embedder loaded: {embedding_model}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        print(f"CLIP image embedder loaded on {self.device}, dimension 512")
        
        try:
            self.llm = ChatGroq(
                model_name=llm_model,
                temperature=LLM_TEMPERATURE,
                groq_api_key=GROQ_API_KEY
            )
            print(f"LLM Groq loaded: {llm_model}")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.llm = None
        
        self.vector_store = IndustrialVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=self.text_collection_name,
            vector_dim=self.embedder.get_dimension(),
            use_hybrid_search=True
        )
        
        self.image_vector_store = IndustrialVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=self.image_collection_name,
            vector_dim=512,
            use_hybrid_search=False
        )
        
        if use_reranker:
            try:
                self.reranker = IndustrialReranker(model_name=RERANKER_MODEL)
                print(f"Reranker loaded: {RERANKER_MODEL}")
            except Exception as e:
                print(f"Reranker not available: {e}")
                self.use_reranker = False
        
        self.doc_loader = IndustrialDocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_cache=use_cache
        )
        
        self.chunker = IndustrialChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_tables=True,
            preserve_code_blocks=True
        )
        
        print("RAG service ready\n")
    
    def _embed_image(self, image_bytes: bytes) -> List[float]:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_input)
                embedding = embedding.cpu().numpy().flatten().tolist()
            return embedding
        except Exception as e:
            print(f"Error embedding image: {e}")
            return [0.0] * 512
    
    def _is_image_file(self, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def _store_image_point(self, filename: str, image_bytes: bytes, embedding: List[float]) -> int:
        point_id = str(uuid.uuid4())
        try:
            self.image_vector_store.client.upsert(
                collection_name=self.image_collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "filename": filename,
                            "modality": "image",
                            "text": f"Image file: {filename}",
                        }
                    )
                ]
            )
            print(f"Image point stored in {self.image_collection_name}: {filename}")
            return 1
        except Exception as e:
            print(f"Qdrant upsert failed for image {filename}: {e}")
            return 0
    

    def ingest(self, file_stream: BytesIO, filename: str) -> int:
        try:
            print(f"Ingestion started for {filename}")
            return 1
        except Exception as e:
            print(f"Error: {e}")
            return 0
    
    def search(self, query: str, limit: int = TOP_K) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)

        text_results = self.vector_store.search_vector(
            query_vector=query_vector,
            limit=limit,
            score_threshold=0.1
        )

        image_results = []
        try:
            image_results = self.image_vector_store.search_vector(
                query_vector=query_vector,
                limit=limit // 2,
                score_threshold=0.1
            )
        except Exception as e:
            print(f"Image search failed: {e}")

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

        return all_results[:limit]

    def run(self, query: str, limit: int = TOP_N) -> Dict[str, Any]:
        print(f"Processing Query: {query}")

        chunks = self.search(query, limit=self.top_k)

        sources_for_response = []
        if chunks:
            top_chunks = chunks[:limit]
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
            context = "No relevant documents found in the knowledge base."
            sources_for_response = []

        prompt = f"""You are a professional and friendly Industrial AI Assistant.

PROVIDED CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {query}

RESPONSE GUIDELINES:
1. If the user greets you or engages in small talk, respond politely as a chatbot.
2. If the question is about technical details found in the CONTEXT, provide a precise answer based on the documents.
3. If the information is NOT in the CONTEXT, clearly state that it is not in the indexed documents, but provide a helpful answer using your general knowledge if possible.
4. Always maintain a professional and supportive tone.
"""

        try:
            if self.llm is None:
                raise Exception("LLM not initialized")
                
            response = self.llm.invoke(prompt)
            answer = response.content
            print(f"Response generated ({len(answer)} characters)")
        except Exception as e:
            print(f"LLM error: {e}")
            answer = f"I'm sorry, I encountered an error: {str(e)}"

        return {
            "answer": answer,
            "source_documents": sources_for_response,
            "sources_details": chunks[:limit] if chunks else [],
            "processing_time_ms": None
        }
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        documents = {}

        def process_collection(collection_name: str):
            try:
                next_offset = None
                while True:
                    scroll_result = self.vector_store.client.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        offset=next_offset,
                        with_payload=True,
                        with_vectors=False
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
                print(f"Error scrolling {collection_name}: {e}")

        process_collection(self.text_collection_name)
        process_collection(self.image_collection_name)

        result = list(documents.values())
        print(f"Documents found: {len(result)}")
        for doc in result:
            print(f"   - {doc['name']}: {doc['chunks']} chunks")
        return result
    
    def delete_document(self, filename: str) -> int:
        deleted_count = 0
        
        try:
            scroll_result = self.vector_store.client.scroll(
                collection_name=self.text_collection_name,
                limit=10000,
                with_payload=True,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )
            
            point_ids = [point.id for point in scroll_result[0]]
            if point_ids:
                self.vector_store.client.delete(
                    collection_name=self.text_collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                deleted_count += len(point_ids)
                print(f"Deleted from text collection: {filename} ({len(point_ids)} chunks)")
        except Exception as e:
            print(f"Error deleting from text collection: {e}")
        
        try:
            scroll_result = self.image_vector_store.client.scroll(
                collection_name=self.image_collection_name,
                limit=10000,
                with_payload=True,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )
            
            point_ids = [point.id for point in scroll_result[0]]
            if point_ids:
                self.image_vector_store.client.delete(
                    collection_name=self.image_collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                deleted_count += len(point_ids)
                print(f"Deleted from image collection: {filename} ({len(point_ids)} chunks)")
        except Exception as e:
            print(f"Error deleting from image collection: {e}")
        
        if deleted_count == 0:
            print(f"No chunks found for: {filename}")
        
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            text_info = self.vector_store.client.get_collection(
                collection_name=self.text_collection_name
            )
            image_info = self.image_vector_store.client.get_collection(
                collection_name=self.image_collection_name
            )
            return {
                "text_collection": {
                    "name": self.text_collection_name,
                    "total_points": text_info.points_count,
                },
                "image_collection": {
                    "name": self.image_collection_name,
                    "total_points": image_info.points_count,
                },
                "status": "healthy"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def clear_collection(self) -> int:
        deleted_count = 0
        
        try:
            scroll_result = self.vector_store.client.scroll(
                collection_name=self.text_collection_name,
                limit=10000,
                with_payload=False
            )
            point_ids = [point.id for point in scroll_result[0]]
            if point_ids:
                self.vector_store.client.delete(
                    collection_name=self.text_collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                deleted_count += len(point_ids)
            print(f"Text collection cleared: {len(point_ids)} points deleted")
        except Exception as e:
            print(f"Error clearing text collection: {e}")
        
        try:
            scroll_result = self.image_vector_store.client.scroll(
                collection_name=self.image_collection_name,
                limit=10000,
                with_payload=False
            )
            point_ids = [point.id for point in scroll_result[0]]
            if point_ids:
                self.image_vector_store.client.delete(
                    collection_name=self.image_collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                deleted_count += len(point_ids)
            print(f"Image collection cleared: {len(point_ids)} points deleted")
        except Exception as e:
            print(f"Error clearing image collection: {e}")
        
        return deleted_count