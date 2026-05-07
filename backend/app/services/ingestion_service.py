import logging
from app.db.vector_store import IndustrialVectorStore
from app.services.embedding_service import IndustrialEmbeddingService
from app.services.rerank_service import IndustrialReranker
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
from app.utils.document_loader import DocumentChunk
from app.utils.chunking import Chunk
from app.utils.document_loader import IndustrialDocumentLoader
from app.utils.chunking import IndustrialChunker
from app.services.embedding_service import IndustrialEmbeddingService
from app.db.vector_store import IndustrialVectorStore
from app.services.rerank_service import IndustrialReranker

logger = logging.getLogger(__name__)


class IngestionService:
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_cache: bool = True,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "phd_collection",
        vector_dim: int = 384
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
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
        
        self.embedder = IndustrialEmbeddingService(
            model_name=embedding_model,
            use_cache=use_cache,
            batch_size=32
        )
        
        self.vector_store = IndustrialVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name,
            vector_dim=vector_dim,
            use_hybrid_search=True
        )
        
        logger.info(f"IngestionService initialized: chunk_size={chunk_size}, model={embedding_model}")
    
    def ingest_pdf_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        file_path = Path(file_path)
        metadata = metadata or {}
        
        logger.info(f"Ingesting PDF: {file_path.name}")
        
        chunks = self.doc_loader.load_and_chunk(file_path, metadata=metadata)
        
        if not chunks:
            logger.warning(f"No chunks generated for {file_path.name}")
            return 0
        
        return self._index_chunks(chunks, file_path.name)
    
    def ingest_pdf_stream(
        self,
        pdf_stream: BytesIO,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        metadata = metadata or {}
        metadata["filename"] = filename
        
        logger.info(f"Ingesting PDF stream: {filename}")
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_stream.getvalue())
            tmp_path = tmp.name
        
        try:
            chunks = self.doc_loader.load_and_chunk(Path(tmp_path), metadata=metadata)
            
            if not chunks:
                logger.warning(f"No chunks generated for {filename}")
                return 0
            
            return self._index_chunks(chunks, filename)
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "text_input"
    ) -> int:
        if not text or not text.strip():
            logger.warning("Empty text, ingestion skipped")
            return 0
        
        metadata = metadata or {}
        metadata["source"] = source
        metadata["filename"] = metadata.get("filename", f"text_{source}")
        
        logger.info(f"Ingesting text: {len(text)} characters")
        
        chunks = self.chunker.split_text(text, source=source)
        
        if not chunks:
            logger.warning("No chunks generated")
            return 0
        
        for chunk in chunks:
            chunk.metadata.update(metadata)
        
        return self._index_chunks(chunks, source)
    
    def ingest_markdown(
        self,
        markdown_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "markdown_input"
    ) -> int:
        if not markdown_text or not markdown_text.strip():
            logger.warning("Empty markdown, ingestion skipped")
            return 0
        
        metadata = metadata or {}
        metadata["source"] = source
        metadata["filename"] = metadata.get("filename", f"markdown_{source}")
        
        logger.info(f"Ingesting markdown: {len(markdown_text)} characters")
        
        chunks = self.chunker.split_markdown(markdown_text, metadata=metadata)
        
        if not chunks:
            logger.warning("No chunks generated")
            return 0
        
        return self._index_chunks(chunks, source)
    
    def _index_chunks(
        self,
        chunks: List[Union[DocumentChunk, Chunk]],
        source_name: str
    ) -> int:
        if not chunks:
            return 0
        
        chunk_texts = []
        embeddings = []
        metadata_list = []
        
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, 'text'):
                text = chunk.text
            elif hasattr(chunk, 'page_content'):
                text = chunk.page_content
            else:
                text = str(chunk)
            
            chunk_texts.append(text)
            
            chunk_metadata = {
                "chunk_index": i,
                "source": source_name,
            }
            
            if hasattr(chunk, 'metadata'):
                chunk_metadata.update(chunk.metadata)
            
            if hasattr(chunk, 'title_path') and chunk.title_path:
                chunk_metadata["title_path"] = " > ".join(chunk.title_path)
            
            metadata_list.append(chunk_metadata)
        
        logger.info(f"Generating {len(chunk_texts)} embeddings...")
        embeddings = self.embedder.embed_batch(chunk_texts, show_progress=True)
        
        inserted = self.vector_store.upsert_documents(
            chunks=chunk_texts,
            embeddings=embeddings,
            metadata=metadata_list
        )
        
        logger.info(f"{inserted} chunks indexed for {source_name}")
        return inserted
    
    def ingest_directory(
        self,
        directory_path: str,
        extensions: List[str] = None
    ) -> Dict[str, int]:
        extensions = extensions or ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.rtf', '.html', '.htm', '.txt', '.md', '.jpg', '.jpeg', '.png', '.tiff']
    
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} file(s) in {directory_path}")
        
        stats = {
            "total": len(files),
            "success": 0,
            "failed": 0,
            "total_chunks": 0
        }
        
        for file_path in files:
            try:
                chunks_count = self.ingest_pdf_file(file_path)
                if chunks_count > 0:
                    stats["success"] += 1
                    stats["total_chunks"] += chunks_count
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Ingestion error for {file_path.name}: {e}")
                stats["failed"] += 1
        
        logger.info(f"Ingestion completed: {stats['success']} success, {stats['failed']} failures, {stats['total_chunks']} chunks")
        return stats
    
    def delete_document(self, filename: str) -> int:
        return self.vector_store.delete_documents(filter_conditions={"filename": filename})
    
    def get_documents_list(self) -> List[str]:
        return self.vector_store.get_all_filenames()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedder.model_name,
            "vector_store": self.vector_store.get_collection_stats()
        }
    
    def clear_all(self) -> int:
        return self.vector_store.clear_collection()