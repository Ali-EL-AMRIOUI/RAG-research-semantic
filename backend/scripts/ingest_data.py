"""
CLI d'ingestion en masse.

Ancienne version : réimplémentait tout son propre pipeline (loader, chunker,
embedder, vector_store) au lieu d'utiliser IngestionService — et appelait
`loader.load_pdf(pdf_path)`, une méthode qui n'a jamais existé sur
IndustrialDocumentLoader (elle existait seulement sur le loader Docling
abandonné, sous le nom `.export_to_markdown()`). AttributeError garanti dès
qu'un fichier était présent dans data/raw.

Contenait aussi un résidu `from fastapi import FastAPI; app = FastAPI()`
sans rapport avec un script CLI — supprimé.

Cette version est un simple wrapper autour de IngestionService, qui est
déjà la pipeline réelle et testée du projet (utilisée aussi par l'API).
Un seul chemin d'ingestion dans tout le repo, plus deux qui divergent.
"""
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.ingestion_service import IngestionService

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
    "collection_name": "documents",
}


def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config["raw_data_dir"] = os.environ.get("RAW_DATA_DIR", config["raw_data_dir"])
    config["chunk_size"] = int(os.environ.get("CHUNK_SIZE", config["chunk_size"]))
    config["chunk_overlap"] = int(os.environ.get("CHUNK_OVERLAP", config["chunk_overlap"]))
    config["embedding_model"] = os.environ.get("EMBEDDING_MODEL", config["embedding_model"])
    config["qdrant_host"] = os.environ.get("QDRANT_HOST", config["qdrant_host"])
    config["qdrant_port"] = int(os.environ.get("QDRANT_PORT", config["qdrant_port"]))
    config["collection_name"] = os.environ.get("QDRANT_COLLECTION_NAME", config["collection_name"])
    return config


def main():
    logger.info("Starting ingestion pipeline...")

    config = load_config()
    logger.info(f"Source directory: {config['raw_data_dir']}")
    logger.info(f"Chunk size: {config['chunk_size']}, Overlap: {config['chunk_overlap']}")
    logger.info(f"Embedding model: {config['embedding_model']}")
    logger.info(f"Qdrant: {config['qdrant_host']}:{config['qdrant_port']}")

    try:
        service = IngestionService(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            embedding_model=config["embedding_model"],
            qdrant_host=config["qdrant_host"],
            qdrant_port=config["qdrant_port"],
            collection_name=config["collection_name"],
        )

        stats = service.ingest_directory(config["raw_data_dir"])

        logger.info("=" * 50)
        logger.info("INGESTION SUMMARY")
        logger.info(f"   Total: {stats['total']}")
        logger.info(f"   Success: {stats['success']}")
        logger.info(f"   Failed: {stats['failed']}")
        logger.info(f"   Chunks indexed: {stats['total_chunks']}")
        logger.info("=" * 50)

        if stats["failed"] > 0:
            logger.warning(f"{stats['failed']} document(s) failed")
            sys.exit(1)

        logger.info("Ingestion pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()