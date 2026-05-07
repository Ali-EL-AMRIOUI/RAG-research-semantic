from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query", min_length=1, max_length=5000)
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1, le=50)
    top_n: int = Field(default=3, description="Number of chunks to use for response", ge=1, le=20)
    use_hybrid: bool = Field(default=True, description="Use hybrid search")
    use_reranker: bool = Field(default=True, description="Use reranking")

class DocumentUploadRequest(BaseModel):
    text: str = Field(..., description="Text to ingest", min_length=1)
    filename: Optional[str] = Field(default=None, description="Source filename")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Metadata")

class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., description="List of questions", min_items=1, max_items=20)
    top_k: int = Field(default=5, ge=1, le=20)

class SourceDocument(BaseModel):
    content: str = Field(..., description="Chunk content")
    filename: str = Field(default="unknown", description="Source filename")
    page: int = Field(default=0, description="Page number")
    score: float = Field(default=0.0, description="Similarity score")
    title_path: str = Field(default="", description="Title path")

class QueryResponse(BaseModel):
    query: str = Field(..., description="User query")
    answer: str = Field(..., description="Generated answer")
    retrieved_chunks: List[str] = Field(default=[], description="Retrieved chunks")
    sources: List[SourceDocument] = Field(default=[], description="Detailed sources")
    total_chunks: int = Field(default=0, description="Number of chunks used")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time in ms")

class SimpleQueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total: int

class Document(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = {}

class Chunk(BaseModel):
    text: str
    index: int = 0
    metadata: Optional[Dict[str, Any]] = {}

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int
    model: str

class VectorPoint(BaseModel):
    id: str
    vector: List[float]
    payload: Dict[str, Any]
    score: Optional[float] = None

class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]

class UploadResponse(BaseModel):
    filename: str
    chunks_count: int
    message: str
    success: bool = True

class DocumentInfo(BaseModel):
    name: str
    chunks: int
    pages: List[int] = []
    uploaded_at: Optional[str] = None

class DocumentsListResponse(BaseModel):
    total_documents: int
    documents: List[DocumentInfo]
    total_chunks: int

class DeleteResponse(BaseModel):
    deleted_count: int
    message: str

class ClearResponse(BaseModel):
    deleted_count: int
    message: str

class CollectionStats(BaseModel):
    name: str
    points_count: int
    vectors_count: int
    dimension: int
    hybrid_search_enabled: bool = False

class ServiceStats(BaseModel):
    collection: CollectionStats
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    documents_count: int

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    services: Dict[str, bool] = Field(default={})
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int = 500

class FeedbackRequest(BaseModel):
    query_id: str
    answer_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=500)

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: Optional[str] = None

class CompareResult(BaseModel):
    mode: str
    results: List[SearchResult]
    avg_score: float
    response_time_ms: float