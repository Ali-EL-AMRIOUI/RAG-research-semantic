import logging
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Union
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)


class IndustrialEmbeddingService:
    
    SUPPORTED_MODELS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-large-en-v1.5": 1024,
        "intfloat/e5-small-v2": 384,
        "sentence-transformers/gtr-t5-large": 768,
    }
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.cache_dir = cache_dir or Path("./cache_embeddings")
        
        if model_name in self.SUPPORTED_MODELS:
            self.dimension = self.SUPPORTED_MODELS[model_name]
        else:
            self.dimension = None
        
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
                raise e
    
    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    def _get_cache_path(self, text_hash: str) -> Path:
        return self.cache_dir / f"{text_hash}.pkl"
    
    def _save_to_cache(self, text_hash: str, embedding: List[float]) -> None:
        if not self.use_cache:
            return
        cache_path = self._get_cache_path(text_hash)
        with open(cache_path, "wb") as f:
            pickle.dump(embedding, f)
    
    def _load_from_cache(self, text_hash: str) -> Optional[List[float]]:
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
    
    def embed_single(self, text: str, use_cache: bool = True) -> List[float]:
        if not text:
            return [0.0] * self.dimension
        
        text_hash = self._get_text_hash(text)
        
        if use_cache and self.use_cache:
            cached = self._load_from_cache(text_hash)
            if cached:
                return cached
        
        try:
            embedding = self.model.encode(
                text, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embedding_list = embedding.tolist()
            
            if use_cache and self.use_cache:
                self._save_to_cache(text_hash, embedding_list)
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Single embedding error: {e}")
            return [0.0] * self.dimension
    
    def embed_batch(
        self, 
        texts: List[str], 
        show_progress: bool = False,
        use_cache: bool = True
    ) -> List[List[float]]:
        if not texts:
            return []
        
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
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
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
    def get_available_models(cls) -> List[str]:
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_model_dimension(cls, model_name: str) -> Optional[int]:
        return cls.SUPPORTED_MODELS.get(model_name)


class EmbeddingServiceFactory:
    
    @staticmethod
    def create_fast() -> IndustrialEmbeddingService:
        return IndustrialEmbeddingService(
            model_name="all-MiniLM-L6-v2",
            use_cache=True,
            batch_size=64
        )
    
    @staticmethod
    def create_accurate() -> IndustrialEmbeddingService:
        return IndustrialEmbeddingService(
            model_name="all-mpnet-base-v2",
            use_cache=True,
            batch_size=32
        )
    
    @staticmethod
    def create_high_quality() -> IndustrialEmbeddingService:
        return IndustrialEmbeddingService(
            model_name="BAAI/bge-large-en-v1.5",
            use_cache=True,
            batch_size=16
        )
    
    @staticmethod
    def create_efficient() -> IndustrialEmbeddingService:
        return IndustrialEmbeddingService(
            model_name="intfloat/e5-small-v2",
            use_cache=True,
            batch_size=64
        )