"""
Point d'entrée FastAPI.

Changements vs la version précédente :
1. Auth : POST /upload, POST /ask, POST /clear et DELETE /documents exigent
   désormais un header X-API-Key valide (voir app.core.security). Ces
   endpoints écrivent dans Qdrant, coûtent un appel Groq, ou peuvent vider
   toute la collection — inacceptable sans protection maintenant que
   kubernetes/ingress.yaml expose cette API publiquement. GET /documents,
   GET /stats et /health restent ouverts (lecture seule, faible risque).
2. Décision search.py vs main.py : search.py supprimé (voir
   git rm backend/app/api/routes/search.py). Il définissait un deuxième
   chemin RAG concurrent (get_rag_service() avec son propre @lru_cache)
   jamais monté ici — le monter tel quel aurait créé une DEUXIÈME instance
   d'IndustrialRAGService (modèle d'embedding, reranker, connexions Qdrant
   dupliqués en mémoire). main.py fait déjà tout ce que search.py propose,
   avec une seule instance partagée.

Inchangé depuis la version précédente :
- /health vérifie une vraie connectivité Qdrant (vector_store.is_healthy()).
- setup_logging() appelé au démarrage.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import io
import logging
import traceback
from dotenv import load_dotenv

load_dotenv()
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from app.core.config import CORS_ORIGINS, API_HOST, API_PORT, API_RELOAD, LOG_LEVEL, LOG_FORMAT, LOG_FILE
from app.core.logging import setup_logging
from app.core.security import verify_api_key

setup_logging(level=LOG_LEVEL, format_type=LOG_FORMAT, log_file=LOG_FILE)
logger = logging.getLogger(__name__)

from app.services.rag_service import IndustrialRAGService


class AskRequest(BaseModel):
    question: str


class DeleteRequest(BaseModel):
    filename: str


app = FastAPI(title="Semantic RAG API", description="API for semantic search with Qdrant and Groq")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {
    '.pdf', '.txt', '.md', '.markdown',
    '.doc', '.docx', '.odt',
    '.xls', '.xlsx', '.csv',
    '.ppt', '.pptx',
    '.html', '.htm', '.xml',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
    '.json', '.yaml', '.yml',
    '.rtf', '.epub'
}

ACCEPTED_FORMATS_STRING = "PDF, TXT, MD, DOC, DOCX, ODT, XLS, XLSX, CSV, PPT, PPTX, HTML, XML, JPG, PNG, GIF, JSON, YAML, RTF, EPUB"

try:
    rag_service = IndustrialRAGService()
    logger.info("RAG service initialized successfully")
except Exception as e:
    logger.error(f"RAG service initialization error: {e}")
    traceback.print_exc()
    rag_service = None


@app.get("/health")
async def health_check():
    qdrant_healthy = False
    if rag_service is not None and hasattr(rag_service, 'vector_store') and rag_service.vector_store is not None:
        qdrant_healthy = rag_service.vector_store.is_healthy()

    return {
        "status": "ok" if (rag_service is not None and qdrant_healthy) else "degraded",
        "rag_service": rag_service is not None,
        "qdrant": qdrant_healthy
    }


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...)):
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    filename = file.filename
    if '.' not in filename:
        raise HTTPException(status_code=400, detail="Invalid file name: no extension")

    file_ext = '.' + filename.split('.')[-1].lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Accepted formats: {ACCEPTED_FORMATS_STRING}"
        )

    try:
        logger.info(f"Receiving file: {filename} (type: {file_ext})")
        content = await file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max size: 100 MB")

        file_stream = io.BytesIO(content)

        chunks_count = rag_service.ingest(file_stream, filename=filename)

        logger.info(f"Ingestion successful: {filename} ({chunks_count} chunks)")
        return {"message": "Success", "filename": filename, "chunks": chunks_count}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("UPLOAD CRITICAL ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def get_documents():
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        documents = rag_service.get_all_documents()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error get_documents: {e}")
        return {"documents": []}


@app.delete("/documents", dependencies=[Depends(verify_api_key)])
async def delete_document(request: DeleteRequest):
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        deleted = rag_service.delete_document(request.filename)
        return {"message": f"Document '{request.filename}' deleted", "deleted_chunks": deleted}
    except Exception as e:
        logger.error(f"Error delete_document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", dependencies=[Depends(verify_api_key)])
async def ask_question(request: AskRequest):
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag_service.run(request.question)
        return result
    except Exception as e:
        logger.error("ASK ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        return rag_service.get_stats()
    except Exception as e:
        logger.error(f"Error stats: {e}")
        return {"error": str(e)}


@app.post("/clear", dependencies=[Depends(verify_api_key)])
async def clear_collection():
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        deleted = rag_service.clear_collection()
        return {"message": "Collection cleared", "deleted_points": deleted}
    except Exception as e:
        logger.error(f"Error clear: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )