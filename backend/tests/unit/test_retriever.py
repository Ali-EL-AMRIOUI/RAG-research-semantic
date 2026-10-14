"""
Unit tests for RetrievalService.

qdrant_client and sentence_transformers are stubbed via sys.modules — not
installed in this environment. These tests target OUR wiring logic
(dimension derivation, hybrid/vector dispatch), not Qdrant or
SentenceTransformer themselves.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("qdrant_client", MagicMock())
sys.modules.setdefault("qdrant_client.http", MagicMock())
sys.modules.setdefault("qdrant_client.models", MagicMock())

from app.services.retrieval_service import RetrievalService  # noqa: E402


class TestDimensionDerivation:
    """
    Regression: vector_dim used to be hardcoded to 384 in the vector_store
    constructor call, independent of the embedding model actually loaded.
    Any model with a different dimension (bge-large 1024, bge-m3 1024...)
    caused a silent mismatch with Qdrant.
    """

    @patch("app.services.retrieval_service.IndustrialVectorStore")
    @patch("app.services.retrieval_service.IndustrialEmbeddingService")
    def test_vector_store_receives_embedder_dimension(self, mock_embedder_cls, mock_store_cls):
        mock_embedder = MagicMock()
        mock_embedder.get_dimension.return_value = 1024
        mock_embedder_cls.return_value = mock_embedder

        RetrievalService(embedding_model="BAAI/bge-m3")

        _, kwargs = mock_store_cls.call_args
        assert kwargs["vector_dim"] == 1024

    @patch("app.services.retrieval_service.IndustrialVectorStore")
    @patch("app.services.retrieval_service.IndustrialEmbeddingService")
    def test_different_model_gives_different_dimension(self, mock_embedder_cls, mock_store_cls):
        mock_embedder = MagicMock()
        mock_embedder.get_dimension.return_value = 384
        mock_embedder_cls.return_value = mock_embedder

        RetrievalService(embedding_model="all-MiniLM-L6-v2")

        _, kwargs = mock_store_cls.call_args
        assert kwargs["vector_dim"] == 384


class TestSearchDispatch:

    @patch("app.services.retrieval_service.IndustrialVectorStore")
    @patch("app.services.retrieval_service.IndustrialEmbeddingService")
    def test_empty_query_returns_empty_without_calling_store(self, mock_embedder_cls, mock_store_cls):
        mock_embedder_cls.return_value = MagicMock(get_dimension=MagicMock(return_value=384))
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        service = RetrievalService()
        result = service.search("   ")

        assert result == []
        mock_store.search_hybrid.assert_not_called()
        mock_store.search_vector.assert_not_called()

    @patch("app.services.retrieval_service.IndustrialVectorStore")
    @patch("app.services.retrieval_service.IndustrialEmbeddingService")
    def test_hybrid_true_calls_search_hybrid(self, mock_embedder_cls, mock_store_cls):
        mock_embedder_cls.return_value = MagicMock(get_dimension=MagicMock(return_value=384))
        mock_store = MagicMock()
        mock_store.search_hybrid.return_value = []
        mock_store_cls.return_value = mock_store

        service = RetrievalService(use_hybrid=True)
        service.search("test query")

        mock_store.search_hybrid.assert_called_once()
        mock_store.search_vector.assert_not_called()

    @patch("app.services.retrieval_service.IndustrialVectorStore")
    @patch("app.services.retrieval_service.IndustrialEmbeddingService")
    def test_hybrid_false_calls_search_vector(self, mock_embedder_cls, mock_store_cls):
        mock_embedder_cls.return_value = MagicMock(get_dimension=MagicMock(return_value=384))
        mock_store = MagicMock()
        mock_store.search_vector.return_value = []
        mock_store_cls.return_value = mock_store

        service = RetrievalService(use_hybrid=False)
        service.search("test query")

        mock_store.search_vector.assert_called_once()
        mock_store.search_hybrid.assert_not_called()

    @patch("app.services.retrieval_service.IndustrialVectorStore")
    @patch("app.services.retrieval_service.IndustrialEmbeddingService")
    def test_per_call_override_takes_priority_over_default(self, mock_embedder_cls, mock_store_cls):
        mock_embedder_cls.return_value = MagicMock(get_dimension=MagicMock(return_value=384))
        mock_store = MagicMock()
        mock_store.search_vector.return_value = []
        mock_store_cls.return_value = mock_store

        service = RetrievalService(use_hybrid=True)
        service.search("test query", use_hybrid=False)  # override at call time

        mock_store.search_vector.assert_called_once()
        mock_store.search_hybrid.assert_not_called()

    @patch("app.services.retrieval_service.IndustrialVectorStore")
    @patch("app.services.retrieval_service.IndustrialEmbeddingService")
    def test_search_error_returns_empty_list_not_exception(self, mock_embedder_cls, mock_store_cls):
        mock_embedder_cls.return_value = MagicMock(get_dimension=MagicMock(return_value=384))
        mock_store = MagicMock()
        mock_store.search_hybrid.side_effect = Exception("connection refused")
        mock_store_cls.return_value = mock_store

        service = RetrievalService()
        result = service.search("test query")

        assert result == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))