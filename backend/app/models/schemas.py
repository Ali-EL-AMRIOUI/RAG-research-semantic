"""
Schémas Pydantic unifiés de l'API.

Remplace 3 fichiers qui définissaient chacun leur propre version des mêmes
concepts (requests.py, schemas.py, db/documents.py) — dont un FeedbackRequest
dupliqué avec des champs différents. Source unique désormais.

Si app/api/routes/search.py ou app/main.py importaient depuis
`app.models.requests` ou `app.db.documents`, mettre à jour les imports vers
`app.models.schemas`.
"""
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchType(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


# --- Modèles de données internes (représentation API d'un document/chunk) ---

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: dict[str, Any] | None = {}


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str | None = None
    text: str
    chunk_index: int = 0
    metadata: dict[str, Any] | None = {}


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
    sources: list[SourceDocument] = []
    total_chunks: int = 0
    processing_time_ms: float | None = None


class BatchQueryRequest(BaseModel):
    queries: list[str] = Field(..., min_items=1, max_items=20)
    top_k: int = Field(default=5, ge=1, le=20)


# --- Requêtes / réponses recherche pure (sans LLM) ---

class SimpleSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    search_type: SearchType = Field(default=SearchType.HYBRID)
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    filters: dict[str, Any] | None = None


class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    metadata: dict[str, Any] = {}


class CompareSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    modes: list[SearchType] = [SearchType.VECTOR, SearchType.HYBRID, SearchType.KEYWORD]


class CompareResult(BaseModel):
    mode: str
    results: list[SearchResult]
    avg_score: float
    response_time_ms: float


# --- Ingestion ---

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    filename: str | None = None
    metadata: dict[str, Any] | None = {}


class IngestFileRequest(BaseModel):
    filename: str
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = {}


class UploadResponse(BaseModel):
    filename: str
    chunks_count: int
    message: str
    success: bool = True


# --- Gestion des documents indexés ---

class DocumentInfo(BaseModel):
    name: str
    chunks: int
    pages: list[int] = []
    uploaded_at: str | None = None


class DocumentsListResponse(BaseModel):
    total_documents: int
    documents: list[DocumentInfo]
    total_chunks: int


class DeleteRequest(BaseModel):
    ids: list[str] | None = None
    filename: str | None = None
    filter_conditions: dict[str, Any] | None = None


class DeleteResponse(BaseModel):
    deleted_count: int
    message: str


class ClearCollectionRequest(BaseModel):
    confirm: bool = Field(..., description="Doit être True pour confirmer la suppression")


class ClearResponse(BaseModel):
    deleted_count: int
    message: str


class UpdateMetadataRequest(BaseModel):
    ids: list[str]
    metadata: dict[str, Any]


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
    services: dict[str, bool] = {}
    message: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    status_code: int = 500


# --- Embeddings bruts (endpoint de debug/introspection éventuel) ---

class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimension: int
    model: str


class VectorPoint(BaseModel):
    id: str
    vector: list[float]
    payload: dict[str, Any]
    score: float | None = None


# --- Feedback utilisateur ---

class FeedbackRequest(BaseModel):
    query_id: str
    answer_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = Field(default=None, max_length=500)
    relevant_sources: list[str] | None = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: str | None = None