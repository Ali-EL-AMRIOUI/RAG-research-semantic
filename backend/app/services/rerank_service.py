import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    text: str
    score: float
    original_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "original_index": self.original_index,
            "metadata": self.metadata
        }


class IndustrialReranker:
    
    SUPPORTED_MODELS = {
        "cross-encoder/ms-marco-MiniLM-L-6-v2": {
            "dimension": 384,
            "speed": "fast",
            "accuracy": "good",
            "description": "Good speed/accuracy balance"
        },
        "cross-encoder/ms-marco-MiniLM-L-12-v2": {
            "dimension": 384,
            "speed": "medium",
            "accuracy": "better",
            "description": "More accurate but slower"
        },
        "cross-encoder/ms-marco-roberta-base-v2": {
            "dimension": 768,
            "speed": "slow",
            "accuracy": "best",
            "description": "Best accuracy, slower"
        },
        "BAAI/bge-reranker-base": {
            "dimension": 768,
            "speed": "medium",
            "accuracy": "excellent",
            "description": "Modern high-performance reranker"
        },
        "BAAI/bge-reranker-large": {
            "dimension": 1024,
            "speed": "slow",
            "accuracy": "state_of_art",
            "description": "State of the art, very accurate"
        }
    }
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        default_top_n: int = 3
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.default_top_n = default_top_n
        
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        try:
            logger.info(f"Loading Reranker: {model_name}")
            self.model = CrossEncoder(
                model_name,
                device=self.device
            )
            logger.info(f"Reranker loaded on {self.device}")
            
            if model_name in self.SUPPORTED_MODELS:
                self.model_info = self.SUPPORTED_MODELS[model_name]
            else:
                self.model_info = {
                    "dimension": "unknown",
                    "speed": "unknown",
                    "accuracy": "unknown",
                    "description": "Custom model"
                }
                
        except Exception as e:
            logger.error(f"Error loading reranker {model_name}: {e}")
            if model_name != "cross-encoder/ms-marco-MiniLM-L-6-v2":
                logger.info("Fallback to cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device)
                self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self.model_info = self.SUPPORTED_MODELS[self.model_name]
            else:
                raise e
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_scores: bool = False,
        threshold: Optional[float] = None
    ) -> List[str]:
        if not documents:
            return [] if not return_scores else []
        
        top_n = top_n or self.default_top_n
        top_n = min(top_n, len(documents))
        
        logger.info(f"Reranking: {len(documents)} documents for '{query[:50]}...'")
        
        pairs = [[query, doc] for doc in documents]
        
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            results = list(zip(documents, scores))
            results.sort(key=lambda x: x[1], reverse=True)
            
            if threshold is not None:
                results = [(doc, score) for doc, score in results if score >= threshold]
                top_n = min(top_n, len(results))
            
            top_results = results[:top_n]
            
            logger.info(f"Reranking completed: scores {[round(s, 3) for _, s in top_results]}")
            
            if return_scores:
                return [(doc, float(score)) for doc, score in top_results]
            else:
                return [doc for doc, _ in top_results]
                
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:top_n] if not return_scores else [(doc, 0.0) for doc in documents[:top_n]]
    
    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        top_n: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        texts = [doc.get(text_key, "") for doc in documents]
        
        reranked = self.rerank(query, texts, top_n=top_n, return_scores=True, threshold=threshold)
        
        results = []
        for text, score in reranked:
            original_doc = next((doc for doc in documents if doc.get(text_key) == text), {})
            result = original_doc.copy()
            result["rerank_score"] = score
            results.append(result)
        
        return results
    
    def rerank_search_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        text_key = "text" if search_results and "text" in search_results[0] else "content"
        
        return self.rerank_with_metadata(
            query=query,
            documents=search_results,
            text_key=text_key,
            top_n=top_n,
            threshold=threshold
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "default_top_n": self.default_top_n,
            **self.model_info
        }
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_model_recommendation(cls, requirement: str) -> str:
        recommendations = {
            "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "accurate": "cross-encoder/ms-marco-roberta-base-v2",
            "best": "BAAI/bge-reranker-large"
        }
        return recommendations.get(requirement, "cross-encoder/ms-marco-MiniLM-L-6-v2")