import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.core.config import EMBEDDING_CACHE_DIR
from app.core.logging import PerformanceLogger

load_dotenv()

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger("embedding_service")


class IndustrialEmbeddingService:
    """
    Service d'embedding texte. Supporte plusieurs modèles, cache disque,
    batch processing et fallback automatique.
    """

    SUPPORTED_MODELS: ClassVar[dict[str, int]] = {
        "all-MiniLM-L6-v2": 384,               # Rapide, léger, anglais principalement
        "all-mpnet-base-v2": 768,              # Plus précis, anglais
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-large-en-v1.5": 1024,
        "intfloat/e5-small-v2": 384,
        "sentence-transformers/gtr-t5-large": 768,
        "BAAI/bge-m3": 1024,                   # Multilingue FR/EN, recommandé pour ce projet
    }

    # Modèle recommandé pour un corpus FR/EN comme celui de ce projet.
    RECOMMENDED_MODEL = "BAAI/bge-m3"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_cache: bool = True,
        cache_dir: Path | None = None,
        batch_size: int = 32,
        device: str | None = None
    ):
        """
        Args:
            model_name: Nom du modèle SentenceTransformer. La valeur réelle
                utilisée en prod vient de EMBEDDING_MODEL dans .env — ce
                paramètre par défaut n'est qu'un filet de sécurité.
            use_cache: Activer le cache disque des embeddings
            cache_dir: Dossier de cache. Par défaut EMBEDDING_CACHE_DIR
                (défini dans app.core.config), pas un chemin en dur.
            batch_size: Taille des lots pour le traitement batch
            device: 'cpu', 'cuda', ou None (auto-détection)
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.cache_dir = cache_dir or EMBEDDING_CACHE_DIR

        if model_name in self.SUPPORTED_MODELS:
            self.dimension = self.SUPPORTED_MODELS[model_name]
        else:
            self.dimension = None  # déterminé après chargement

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)

            hf_token = os.environ.get("HF_TOKEN")
            if hf_token and "BAAI" in model_name:
                self.model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            if self.dimension is None:
                test_embedding = self.model.encode(["test"], show_progress_bar=False)
                self.dimension = len(test_embedding[0])

            logger.info(f"Model loaded: dim={self.dimension}, device={self.device}")

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            if model_name != "all-MiniLM-L6-v2":
                logger.info("Fallback to all-MiniLM-L6-v2")
                self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                self.dimension = 384
                self.model_name = "all-MiniLM-L6-v2"
            else:
                raise

    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_cache_path(self, text_hash: str) -> Path:
        return self.cache_dir / f"{text_hash}.pkl"

    def _save_to_cache(self, text_hash: str, embedding: list[float]) -> None:
        if not self.use_cache:
            return
        cache_path = self._get_cache_path(text_hash)
        with open(cache_path, "wb") as f:
            pickle.dump(embedding, f)

    def _load_from_cache(self, text_hash: str) -> list[float] | None:
        if not self.use_cache:
            return None
        cache_path = self._get_cache_path(text_hash)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None

    def embed_single(self, text: str, use_cache: bool = True) -> list[float]:
        if not text:
            return [0.0] * self.dimension

        text_hash = self._get_text_hash(text)

        if use_cache and self.use_cache:
            cached = self._load_from_cache(text_hash)
            if cached:
                return cached

        try:
            embedding = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
            embedding_list = embedding.tolist()

            if use_cache and self.use_cache:
                self._save_to_cache(text_hash, embedding_list)

            return embedding_list

        except Exception as e:
            logger.error(f"Single embedding error: {e}")
            return [0.0] * self.dimension

    def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
        use_cache: bool = True
    ) -> list[list[float]]:
        if not texts:
            return []

        import time
        batch_start = time.perf_counter()

        valid_texts = [t if t else " " for t in texts]

        embeddings = [None] * len(valid_texts)
        uncached_indices = []
        uncached_texts = []

        if use_cache and self.use_cache:
            for i, text in enumerate(valid_texts):
                text_hash = self._get_text_hash(text)
                cached = self._load_from_cache(text_hash)
                if cached:
                    embeddings[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(valid_texts)))
            uncached_texts = valid_texts

        if uncached_texts:
            try:
                batch_embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )

                for i, (idx, embedding) in enumerate(zip(uncached_indices, batch_embeddings)):
                    embedding_list = embedding.tolist()
                    embeddings[idx] = embedding_list
                    if use_cache and self.use_cache:
                        self._save_to_cache(self._get_text_hash(uncached_texts[i]), embedding_list)

            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                for i, idx in enumerate(uncached_indices):
                    embeddings[idx] = self.embed_single(uncached_texts[i], use_cache=False)

        batch_duration_ms = (time.perf_counter() - batch_start) * 1000
        perf_logger.log_embedding_time(count=len(valid_texts), duration_ms=batch_duration_ms)

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Alias pour embed_single (compatibilité avec le reste du code)."""
        return self.embed_single(query)

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "batch_size": self.batch_size,
            "use_cache": self.use_cache,
            "cache_size": len(list(self.cache_dir.glob("*.pkl"))) if self.use_cache else 0
        }

    def clear_cache(self) -> int:
        if not self.use_cache:
            return 0

        files = list(self.cache_dir.glob("*.pkl"))
        count = len(files)
        for f in files:
            f.unlink()
        logger.info(f"Cache cleared: {count} files deleted")
        return count

    @classmethod
    def get_available_models(cls) -> list[str]:
        return list(cls.SUPPORTED_MODELS.keys())

    @classmethod
    def get_model_dimension(cls, model_name: str) -> int | None:
        return cls.SUPPORTED_MODELS.get(model_name)


class EmbeddingServiceFactory:
    """Factory pour créer des services d'embedding avec des configurations prédéfinies."""

    @staticmethod
    def create_fast() -> IndustrialEmbeddingService:
        return IndustrialEmbeddingService(model_name="all-MiniLM-L6-v2", use_cache=True, batch_size=64)

    @staticmethod
    def create_accurate() -> IndustrialEmbeddingService:
        return IndustrialEmbeddingService(model_name="all-mpnet-base-v2", use_cache=True, batch_size=32)

    @staticmethod
    def create_high_quality() -> IndustrialEmbeddingService:
        return IndustrialEmbeddingService(model_name="BAAI/bge-large-en-v1.5", use_cache=True, batch_size=16)

    @staticmethod
    def create_multilingual() -> IndustrialEmbeddingService:
        """BGE-M3 : recommandé pour ce projet (corpus FR/EN)."""
        return IndustrialEmbeddingService(model_name="BAAI/bge-m3", use_cache=True, batch_size=16)

    @staticmethod
    def create_efficient() -> IndustrialEmbeddingService:
        return IndustrialEmbeddingService(model_name="intfloat/e5-small-v2", use_cache=True, batch_size=64)