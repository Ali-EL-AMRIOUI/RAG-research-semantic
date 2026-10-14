"""
Service LLM — wrapper autour du client Groq (langchain_groq.ChatGroq).

Extrait de rag_service.py, qui gérait le chargement et l'appel LLM en dur,
mêlé à toute la logique de recherche. Isoler cette responsabilité permet de
la tester (avec un client mocké) et de changer de provider sans toucher au
reste du pipeline RAG.

rag_service.py utilise maintenant CETTE classe au lieu d'instancier son
propre ChatGroq — sinon ce fichier finit comme image_embedding_service.py
avant sa réécriture : correct, mais mort, dupliqué ailleurs.
"""
import logging
from typing import Optional, Dict, Any

from langchain_groq import ChatGroq

from app.core.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class LLMService:

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        self._client: Optional[ChatGroq] = None
        self._last_usage: Dict[str, Any] = {}

        resolved_key = api_key if api_key is not None else GROQ_API_KEY
        if not resolved_key:
            logger.warning("GROQ_API_KEY manquant — LLMService restera indisponible")
            return

        try:
            self._client = ChatGroq(
                model_name=model_name,
                temperature=temperature,
                groq_api_key=resolved_key,
            )
            logger.info(f"LLM Groq chargé: {model_name}")
        except Exception as e:
            logger.error(f"Erreur chargement LLM {model_name}: {e}")
            self._client = None

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str) -> str:
        """
        Génère une réponse à partir d'un prompt déjà construit. Ne construit
        PAS le prompt lui-même — cette responsabilité reste dans
        rag_service.py, qui connaît seul le contexte RAG (chunks, sources).
        """
        if not self.is_available:
            raise RuntimeError("LLM not initialized (vérifier GROQ_API_KEY)")

        response = self._client.invoke(prompt)
        self._last_usage = getattr(response, "usage_metadata", None) or {}
        return response.content

    @property
    def last_usage(self) -> Dict[str, Any]:
        """Tokens (input/output) du dernier appel generate(). Vide si non disponible."""
        return self._last_usage

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "available": self.is_available,
        }