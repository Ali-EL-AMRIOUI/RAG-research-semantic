from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client.http import models
from pydantic import BaseModel
import uvicorn
import io
import os
import traceback
from dotenv import load_dotenv

load_dotenv()
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from app.services.rag_service import IndustrialRAGService
from app.core.config import CORS_ORIGINS, API_HOST, API_PORT, API_RELOAD

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
    print("RAG service initialized successfully")
except Exception as e:
    print(f"RAG service initialization error: {e}")
    traceback.print_exc()
    rag_service = None

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rag_service": rag_service is not None,
        "qdrant": rag_service is not None and hasattr(rag_service, 'vector_store') and rag_service.vector_store is not None
    }

@app.post("/upload")
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
        print(f"Receiving file: {filename} (type: {file_ext})")
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max size: 100 MB")
        
        file_stream = io.BytesIO(content)
        
        chunks_count = rag_service.ingest(file_stream, filename=filename)
        
        print(f"Ingestion successful: {filename} ({chunks_count} chunks)")
        return {"message": "Success", "filename": filename, "chunks": chunks_count}
        
    except HTTPException:
        raise
    except Exception as e:
        print("UPLOAD CRITICAL ERROR:")
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
        print(f"Error get_documents: {e}")
        return {"documents": []}

@app.delete("/documents")
async def delete_document(request: DeleteRequest):
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        deleted = rag_service.delete_document(request.filename)
        return {"message": f"Document '{request.filename}' deleted", "deleted_chunks": deleted}
    except Exception as e:
        print(f"Error delete_document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: AskRequest):
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag_service.run(request.question)
        return result
    except Exception as e:
        print("ASK ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        return rag_service.get_stats()
    except Exception as e:
        print(f"Error stats: {e}")
        return {"error": str(e)}

@app.post("/clear")
async def clear_collection():
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        deleted = rag_service.clear_collection()
        return {"message": "Collection cleared", "deleted_points": deleted}
    except Exception as e:
        print(f"Error clear: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )