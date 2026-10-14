"""
Schémas Pydantic unifiés de l'API.

Remplace 3 fichiers qui définissaient chacun leur propre version des mêmes
concepts (requests.py, schemas.py, db/documents.py) — dont un FeedbackRequest
dupliqué avec des champs différents. Source unique désormais.

Si app/api/routes/search.py ou app/main.py importaient depuis
`app.models.requests` ou `app.db.documents`, mettre à jour les imports vers
`app.models.schemas`.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid


class SearchType(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


# --- Modèles de données internes (représentation API d'un document/chunk) ---

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: Optional[Dict[str, Any]] = {}


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = None
    text: str
    chunk_index: int = 0
    metadata: Optional[Dict[str, Any]] = {}


class SourceDocument(BaseModel):
    content: str = Field(..., description="Chunk content")
    filename: str = Field(default="unknown")
    page: int = Field(default=0)
    score: float = Field(default=0.0)
    title_path: str = Field(default="")


# --- Requêtes / réponses RAG ("ask" : retrieval + génération LLM) ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=10, ge=1, le=50, description="Chunks récupérés avant reranking")
    top_n: int = Field(default=5, ge=1, le=20, description="Chunks utilisés pour la réponse")
    use_hybrid: bool = Field(default=True)
    use_reranker: bool = Field(default=True)
    return_sources: bool = Field(default=True, description="Inclure les sources dans la réponse")


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceDocument] = []
    total_chunks: int = 0
    processing_time_ms: Optional[float] = None


class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=20)
    top_k: int = Field(default=5, ge=1, le=20)


# --- Requêtes / réponses recherche pure (sans LLM) ---

class SimpleSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    search_type: SearchType = Field(default=SearchType.HYBRID)
    score_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    metadata: Dict[str, Any] = {}


class CompareSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    modes: List[SearchType] = [SearchType.VECTOR, SearchType.HYBRID, SearchType.KEYWORD]


class CompareResult(BaseModel):
    mode: str
    results: List[SearchResult]
    avg_score: float
    response_time_ms: float


# --- Ingestion ---

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    filename: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}


class IngestFileRequest(BaseModel):
    filename: str
    content: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = {}


class UploadResponse(BaseModel):
    filename: str
    chunks_count: int
    message: str
    success: bool = True


# --- Gestion des documents indexés ---

class DocumentInfo(BaseModel):
    name: str
    chunks: int
    pages: List[int] = []
    uploaded_at: Optional[str] = None


class DocumentsListResponse(BaseModel):
    total_documents: int
    documents: List[DocumentInfo]
    total_chunks: int


class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    filename: Optional[str] = None
    filter_conditions: Optional[Dict[str, Any]] = None


class DeleteResponse(BaseModel):
    deleted_count: int
    message: str


class ClearCollectionRequest(BaseModel):
    confirm: bool = Field(..., description="Doit être True pour confirmer la suppression")


class ClearResponse(BaseModel):
    deleted_count: int
    message: str


class UpdateMetadataRequest(BaseModel):
    ids: List[str]
    metadata: Dict[str, Any]


# --- Stats / santé ---

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
    services: Dict[str, bool] = {}
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int = 500


# --- Embeddings bruts (endpoint de debug/introspection éventuel) ---

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int
    model: str


class VectorPoint(BaseModel):
    id: str
    vector: List[float]
    payload: Dict[str, Any]
    score: Optional[float] = None


# --- Feedback utilisateur ---

class FeedbackRequest(BaseModel):
    query_id: str
    answer_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=500)
    relevant_sources: Optional[List[str]] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: Optional[str] = None