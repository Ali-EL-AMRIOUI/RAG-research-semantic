import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.services.rag_service import IndustrialRAGService
from app.services.retrieval_service import RetrievalService
from app.services.rerank_service import IndustrialReranker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=5000)
    limit: int = Field(default=3, description="Number of results", ge=1, le=20)
    use_hybrid: bool = Field(default=True, description="Use hybrid search")
    use_reranker: bool = Field(default=True, description="Use reranking")
    return_sources: bool = Field(default=True, description="Return sources")

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]] = []
    total_chunks: int = 0
    retrieval_mode: str = "hybrid"
    reranker_used: bool = False

class SimpleSearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total: int

def get_rag_service() -> IndustrialRAGService:
    try:
        return IndustrialRAGService()
    except Exception as e:
        logger.error(f"RAGService initialization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG service unavailable"
        )

def get_retrieval_service() -> RetrievalService:
    try:
        return RetrievalService()
    except Exception as e:
        logger.error(f"RetrievalService initialization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search service unavailable"
        )

def get_reranker() -> IndustrialReranker:
    try:
        return IndustrialReranker()
    except Exception as e:
        logger.error(f"Reranker initialization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Reranking service unavailable"
        )

@router.post("/rag", response_model=SearchResponse)
async def search_rag(
    request: SearchRequest,
    rag: IndustrialRAGService = Depends(get_rag_service)
):
    try:
        logger.info(f"RAG request: '{request.query[:100]}...'")
        
        result = rag.run(
            query=request.query,
            limit=request.limit,
            use_hybrid=request.use_hybrid,
            use_reranker=request.use_reranker
        )
        
        return SearchResponse(
            query=request.query,
            answer=result.get("answer", ""),
            sources=result.get("sources_details", []) if request.return_sources else [],
            total_chunks=len(result.get("source_documents", [])),
            retrieval_mode="hybrid" if request.use_hybrid else "vector",
            reranker_used=request.use_reranker
        )
        
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search error: {str(e)}"
        )

@router.post("/rag/simple", response_model=SearchResponse)
async def search_rag_simple(
    query: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(3, ge=1, le=10),
    rag: IndustrialRAGService = Depends(get_rag_service)
):
    request = SearchRequest(query=query, limit=limit)
    return await search_rag(request, rag)

@router.post("/retrieve", response_model=SimpleSearchResponse)
async def search_retrieve(
    request: SearchRequest,
    retrieval: RetrievalService = Depends(get_retrieval_service),
    reranker: Optional[IndustrialReranker] = Depends(get_reranker)
):
    try:
        logger.info(f"Retrieval request: '{request.query[:100]}...'")
        
        results = retrieval.search(
            query=request.query,
            limit=request.limit * 3 if request.use_reranker else request.limit,
            use_hybrid=request.use_hybrid
        )
        
        if request.use_reranker and results and reranker:
            results = reranker.rerank_search_results(
                query=request.query,
                search_results=results,
                top_n=request.limit
            )
        
        return SimpleSearchResponse(
            query=request.query,
            results=results[:request.limit],
            total=len(results)
        )
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/retrieve", response_model=SimpleSearchResponse)
async def search_retrieve_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=20),
    use_hybrid: bool = Query(True),
    retrieval: RetrievalService = Depends(get_retrieval_service)
):
    request = SearchRequest(query=q, limit=limit, use_hybrid=use_hybrid)
    return await search_retrieve(request, retrieval, None)

@router.get("/document/{filename}")
async def search_by_document(
    filename: str,
    retrieval: RetrievalService = Depends(get_retrieval_service)
):
    try:
        chunks = retrieval.get_document_chunks(filename)
        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def list_documents(
    retrieval: RetrievalService = Depends(get_retrieval_service)
):
    try:
        documents = retrieval.get_all_filenames()
        stats = retrieval.get_collection_stats()
        
        return {
            "total_documents": len(documents),
            "documents": documents,
            "collection_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_search_stats(
    retrieval: RetrievalService = Depends(get_retrieval_service)
):
    try:
        return retrieval.get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "search"}

@router.post("/test")
async def test_search(
    query: str = Query(..., description="Test query")
):
    try:
        retrieval = RetrievalService()
        results = retrieval.search_simple(query, limit=3)
        return {
            "query": query,
            "found": len(results),
            "preview": [r[:200] + "..." if len(r) > 200 else r for r in results]
        }
    except Exception as e:
        return {"error": str(e)}