import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI
app = FastAPI()

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.document_loader import IndustrialDocumentLoader
from app.utils.chunking import IndustrialChunker
from app.services.embedding_service import IndustrialEmbeddingService
from app.db.vector_store import IndustrialVectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "raw_data_dir": "./data/raw",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "all-MiniLM-L6-v2",
    "qdrant_host": "localhost",
    "qdrant_port": 6333,
    "collection_name": "phd_collection",
    "vector_dim": 384
}


def load_config():
    config = DEFAULT_CONFIG.copy()
    
    config["raw_data_dir"] = os.environ.get("RAW_DATA_DIR", config["raw_data_dir"])
    config["chunk_size"] = int(os.environ.get("CHUNK_SIZE", config["chunk_size"]))
    config["chunk_overlap"] = int(os.environ.get("CHUNK_OVERLAP", config["chunk_overlap"]))
    config["embedding_model"] = os.environ.get("EMBEDDING_MODEL", config["embedding_model"])
    config["qdrant_host"] = os.environ.get("QDRANT_HOST", config["qdrant_host"])
    config["qdrant_port"] = int(os.environ.get("QDRANT_PORT", config["qdrant_port"]))
    config["collection_name"] = os.environ.get("QDRANT_COLLECTION_NAME", config["collection_name"])
    config["vector_dim"] = int(os.environ.get("VECTOR_DIM", config["vector_dim"]))
    
    return config


def ingest_single_document(
    loader: IndustrialDocumentLoader,
    chunker: IndustrialChunker,
    embedding_service: IndustrialEmbeddingService,
    vector_store: IndustrialVectorStore,
    pdf_path: Path,
    metadata: dict = None
) -> bool:
    try:
        logger.info(f"Processing: {pdf_path.name}")
        
        markdown_text = loader.load_pdf(pdf_path)
        if not markdown_text:
            logger.warning(f"Empty document: {pdf_path.name}")
            return False
        
        logger.info(f"{len(markdown_text)} characters extracted")
        
        chunks = chunker.split_markdown(
            markdown_text, 
            metadata={"source": pdf_path.name}
        )
        
        if not chunks:
            logger.warning(f"No chunks generated for {pdf_path.name}")
            return False
        
        logger.info(f"{len(chunks)} chunks generated")
        
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedding_service.embed_batch(chunk_texts, show_progress=True)
        
        metadata_list = []
        for chunk in chunks:
            chunk_metadata = {
                "source": pdf_path.name,
                "filename": pdf_path.name,
                "chunk_index": chunk.chunk_index,
                "title_path": chunk.get_full_title(),
                **chunk.metadata
            }
            if metadata:
                chunk_metadata.update(metadata)
            metadata_list.append(chunk_metadata)
        
        inserted = vector_store.upsert_documents(
            chunks=chunk_texts,
            embeddings=embeddings,
            metadata=metadata_list
        )
        
        logger.info(f"{inserted} chunks indexed for {pdf_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error ingesting {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def ingest_directory(
    directory_path: str,
    recursive: bool = False,
    extensions: list = None
) -> dict:
    extensions = extensions or ['.pdf']
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    pdf_files = []
    for ext in extensions:
        if recursive:
            pdf_files.extend(directory.rglob(f"*{ext}"))
        else:
            pdf_files.extend(directory.glob(f"*{ext}"))
    
    logger.info(f"{len(pdf_files)} document(s) found in {directory_path}")
    
    if not pdf_files:
        return {"total": 0, "success": 0, "failed": 0, "files": []}
    
    config = load_config()
    
    loader = IndustrialDocumentLoader(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        use_cache=True
    )
    
    chunker = IndustrialChunker(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        preserve_tables=True,
        preserve_code_blocks=True
    )
    
    embedding_service = IndustrialEmbeddingService(
        model_name=config["embedding_model"],
        use_cache=True,
        batch_size=32
    )
    
    vector_store = IndustrialVectorStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
        collection_name=config["collection_name"],
        vector_dim=config["vector_dim"],
        use_hybrid_search=True
    )
    
    stats = {
        "total": len(pdf_files),
        "success": 0,
        "failed": 0,
        "files": []
    }
    
    for pdf_path in pdf_files:
        success = ingest_single_document(
            loader=loader,
            chunker=chunker,
            embedding_service=embedding_service,
            vector_store=vector_store,
            pdf_path=pdf_path
        )
        
        if success:
            stats["success"] += 1
            stats["files"].append({"name": pdf_path.name, "status": "success"})
        else:
            stats["failed"] += 1
            stats["files"].append({"name": pdf_path.name, "status": "failed"})
    
    logger.info("=" * 50)
    logger.info(f"INGESTION SUMMARY")
    logger.info(f"   Total: {stats['total']}")
    logger.info(f"   Success: {stats['success']}")
    logger.info(f"   Failed: {stats['failed']}")
    
    collection_stats = vector_store.get_collection_stats()
    logger.info(f"   Points in Qdrant: {collection_stats.get('points_count', 0)}")
    logger.info("=" * 50)
    
    return stats


def main():
    logger.info("Starting ingestion pipeline...")
    
    config = load_config()
    logger.info(f"Source directory: {config['raw_data_dir']}")
    logger.info(f"Chunk size: {config['chunk_size']}, Overlap: {config['chunk_overlap']}")
    logger.info(f"Embedding model: {config['embedding_model']}")
    logger.info(f"Qdrant: {config['qdrant_host']}:{config['qdrant_port']}")
    
    try:
        stats = ingest_directory(
            directory_path=config["raw_data_dir"],
            recursive=False,
            extensions=['.pdf']
        )
        
        if stats["failed"] > 0:
            logger.warning(f"{stats['failed']} document(s) failed")
            sys.exit(1)
        
        logger.info("Ingestion pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()