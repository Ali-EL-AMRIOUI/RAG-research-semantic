from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uuid


class Document(BaseModel):
    id: str = None
    text: str
    metadata: Optional[Dict[str, Any]] = {}

    def __init__(self, **data):
        if "id" not in data or data["id"] is None:
            data["id"] = str(uuid.uuid4())
        super().__init__(**data)


class Chunk(BaseModel):
    id: str = None
    document_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = {}

    def __init__(self, **data):
        if "id" not in data or data["id"] is None:
            data["id"] = str(uuid.uuid4())
        super().__init__(**data)


class VectorRecord(BaseModel):
    id: str
    vector: List[float]
    payload: Dict[str, Any]


class DocumentStoreHelper:

    @staticmethod
    def to_payload(chunk: Chunk, embedding: List[float]) -> VectorRecord:
        return VectorRecord(
            id=chunk.id,
            vector=embedding,
            payload={
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
        )