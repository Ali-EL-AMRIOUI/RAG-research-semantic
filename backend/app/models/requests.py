from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class SearchType(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=5000)
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1, le=50)
    top_n: int = Field(default=3, description="Number of chunks to use for response", ge=1, le=20)
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Search type")
    use_reranker: bool = Field(default=True, description="Use reranking")
    score_threshold: Optional[float] = Field(default=None, description="Minimum similarity threshold", ge=0, le=1)
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class SimpleSearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    limit: int = Field(default=10, description="Number of results", ge=1, le=100)
    search_type: SearchType = Field(default=SearchType.HYBRID)
    score_threshold: Optional[float] = Field(default=None, ge=0, le=1)


class IngestRequest(BaseModel):
    text: str = Field(..., description="Text to ingest", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Associated metadata")
    filename: Optional[str] = Field(default=None, description="Source filename")


class IngestFileRequest(BaseModel):
    filename: str = Field(..., description="File name")
    content: str = Field(..., description="File text content", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default={})


class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = Field(default=None, description="Specific IDs to delete")
    filename: Optional[str] = Field(default=None, description="Delete all chunks of a file")
    filter_conditions: Optional[Dict[str, Any]] = Field(default=None, description="Filters for deletion")


class ClearCollectionRequest(BaseModel):
    confirm: bool = Field(..., description="Deletion confirmation (must be True)")


class UpdateMetadataRequest(BaseModel):
    ids: List[str] = Field(..., description="Point IDs to modify")
    metadata: Dict[str, Any] = Field(..., description="New metadata (will be merged)")


class BatchSearchRequest(BaseModel):
    queries: List[str] = Field(..., description="List of queries", min_items=1, max_items=50)
    top_k: int = Field(default=5, ge=1, le=20)
    search_type: SearchType = Field(default=SearchType.HYBRID)


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="Query ID")
    answer_id: str = Field(..., description="Answer ID")
    rating: int = Field(..., description="Rating from 1 to 5", ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=500)
    relevant_sources: Optional[List[str]] = Field(default=None)


class CompareSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    modes: List[SearchType] = Field(default=[SearchType.VECTOR, SearchType.HYBRID, SearchType.KEYWORD])