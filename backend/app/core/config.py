import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in .env (required for LLM)")

os.environ["HF_TOKEN"] = HF_TOKEN or ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)

QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "phd_collection")
VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "384"))

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_USE_CACHE: bool = os.getenv("EMBEDDING_USE_CACHE", "True").lower() in ("true", "1", "yes")

LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

TOP_K: int = int(os.getenv("TOP_K", "10"))
TOP_N: int = int(os.getenv("TOP_N", "3"))
SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.4"))
USE_HYBRID_SEARCH: bool = os.getenv("USE_HYBRID_SEARCH", "True").lower() in ("true", "1", "yes")
USE_RERANKER: bool = os.getenv("USE_RERANKER", "True").lower() in ("true", "1", "yes")

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNK_SEPARATORS: list = ["\n\n", "\n", ". ", " ", ""]

RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "3"))

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
CACHE_DIR: Path = BASE_DIR / "cache"
EMBEDDING_CACHE_DIR: Path = CACHE_DIR / "embeddings"
DOCUMENT_CACHE_DIR: Path = CACHE_DIR / "documents"

for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, EMBEDDING_CACHE_DIR, DOCUMENT_CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "True").lower() in ("true", "1", "yes")
API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))

CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "standard")
LOG_FILE: Optional[str] = os.getenv("LOG_FILE", None)

USE_CACHE: bool = os.getenv("USE_CACHE", "True").lower() in ("true", "1", "yes")
CACHE_TTL: int = int(os.getenv("CACHE_TTL", "86400"))

DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
TESTING: bool = os.getenv("TESTING", "False").lower() in ("true", "1", "yes")

def get_config_dict() -> dict:
    return {
        "qdrant_host": QDRANT_HOST,
        "qdrant_port": QDRANT_PORT,
        "collection_name": QDRANT_COLLECTION_NAME,
        "vector_dim": VECTOR_DIM,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "top_n": TOP_N,
        "use_hybrid_search": USE_HYBRID_SEARCH,
        "use_reranker": USE_RERANKER,
        "debug": DEBUG,
    }