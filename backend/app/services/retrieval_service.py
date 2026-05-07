import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.db.vector_store import IndustrialVectorStore
from app.services.embedding_service import IndustrialEmbeddingService

logger = logging.getLogger(__name__)


class RetrievalService:
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "phd_collection",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_hybrid: bool = True
    ):
        self.use_hybrid = use_hybrid
        
        self.vector_store = IndustrialVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name,
            vector_dim=384,
            use_hybrid_search=use_hybrid
        )
        
        self.embedding_service = IndustrialEmbeddingService(
            model_name=embedding_model,
            use_cache=True,
            batch_size=32
        )
        
        logger.info(f"RetrievalService initialized: collection={collection_name}, hybrid={use_hybrid}")
    
    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        use_hybrid: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            logger.warning("Empty query")
            return []
        
        logger.info(f"Search: '{query[:50]}...' (limit={limit})")
        
        query_vector = self.embedding_service.embed_query(query)
        
        hybrid_mode = use_hybrid if use_hybrid is not None else self.use_hybrid
        
        try:
            if hybrid_mode:
                results = self.vector_store.search_hybrid(
                    query_vector=query_vector,
                    query_text=query,
                    limit=limit,
                    score_threshold=score_threshold,
                    filter_conditions=filter_conditions
                )
            else:
                results = self.vector_store.search_vector(
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    filter_conditions=filter_conditions
                )
            
            logger.info(f"{len(results)} result(s) found")
            return [r.to_dict() for r in results]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def search_by_filename(
        self,
        filename: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        logger.info(f"Search by filename: {filename}")
        
        results = self.vector_store.get_documents_by_filename(filename)
        return [r.to_dict() for r in results[:limit]]
    
    def search_simple(
        self,
        query: str,
        limit: int = 5
    ) -> List[str]:
        results = self.search(query, limit=limit)
        return [r["content"] for r in results]
    
    def get_all_filenames(self) -> List[str]:
        return self.vector_store.get_all_filenames()
    
    def get_document_chunks(self, filename: str) -> List[Dict[str, Any]]:
        results = self.vector_store.get_documents_by_filename(filename)
        chunks = [r.to_dict() for r in results]
        chunks.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
        return chunks
    
    def get_collection_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_collection_stats()
    
    def search_with_context(
        self,
        query: str,
        context_chunks: int = 2,
        limit: int = 5
    ) -> Dict[str, Any]:
        results = self.search(query, limit=limit)
        
        enriched_results = []
        for result in results:
            chunk_index = result.get("metadata", {}).get("chunk_index", -1)
            filename = result.get("metadata", {}).get("filename", "")
            
            context_texts = [result["content"]]
            
            if chunk_index >= 0 and filename:
                all_chunks = self.get_document_chunks(filename)
                
                for i in range(1, context_chunks + 1):
                    prev_idx = chunk_index - i
                    if prev_idx >= 0 and prev_idx < len(all_chunks):
                        context_texts.insert(0, all_chunks[prev_idx]["content"])
                
                for i in range(1, context_chunks + 1):
                    next_idx = chunk_index + i
                    if next_idx < len(all_chunks):
                        context_texts.append(all_chunks[next_idx]["content"])
            
            enriched_results.append({
                **result,
                "context": context_texts,
                "full_context": "\n\n---\n\n".join(context_texts)
            })
        
        return {
            "query": query,
            "total_results": len(enriched_results),
            "results": enriched_results
        }