"""
Unit tests for IndustrialEmbeddingService.

sentence_transformers/torch are stubbed via sys.modules — this environment
doesn't have them installed (multi-GB download), and these tests target
OUR wrapper logic (caching, fallback, dimension handling), not
SentenceTransformer itself.
"""
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("torch", MagicMock())

from app.services.embedding_service import (
    EmbeddingServiceFactory,
    IndustrialEmbeddingService,
)


@pytest.fixture
def temp_cache_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _make_mock_model(dimension=384):
    """Fake SentenceTransformer whose .encode() returns predictable vectors."""
    import numpy as np

    mock_model = MagicMock()

    def fake_encode(texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(dimension, dtype=float)
        return np.ones((len(texts), dimension), dtype=float)

    mock_model.encode.side_effect = fake_encode
    return mock_model


class TestInitialization:

    @patch("app.services.embedding_service.SentenceTransformer")
    def test_known_model_uses_declared_dimension(self, mock_st_cls, temp_cache_dir):
        mock_st_cls.return_value = _make_mock_model(384)
        service = IndustrialEmbeddingService(model_name="all-MiniLM-L6-v2", cache_dir=temp_cache_dir, device="cpu")
        assert service.dimension == 384

    @patch("app.services.embedding_service.SentenceTransformer")
    def test_default_cache_dir_comes_from_config(self, mock_st_cls, temp_cache_dir):
        """Regression: cache_dir used to default to a hardcoded
        './cache_embeddings' instead of EMBEDDING_CACHE_DIR from config."""
        mock_st_cls.return_value = _make_mock_model(384)
        service = IndustrialEmbeddingService(model_name="all-MiniLM-L6-v2", device="cpu")
        from app.core.config import EMBEDDING_CACHE_DIR
        assert service.cache_dir == EMBEDDING_CACHE_DIR

    @patch("app.services.embedding_service.SentenceTransformer")
    def test_load_failure_falls_back_to_minilm(self, mock_st_cls, temp_cache_dir):
        mock_st_cls.side_effect = [Exception("model not found"), _make_mock_model(384)]
        service = IndustrialEmbeddingService(model_name="some/broken-model", cache_dir=temp_cache_dir, device="cpu")
        assert service.model_name == "all-MiniLM-L6-v2"
        assert service.dimension == 384


class TestEmbedSingle:

    @patch("app.services.embedding_service.SentenceTransformer")
    def test_empty_text_returns_zero_vector(self, mock_st_cls, temp_cache_dir):
        mock_st_cls.return_value = _make_mock_model(384)
        service = IndustrialEmbeddingService(cache_dir=temp_cache_dir, device="cpu")
        assert service.embed_single("") == [0.0] * 384

    @patch("app.services.embedding_service.SentenceTransformer")
    def test_cache_hit_avoids_recomputation(self, mock_st_cls, temp_cache_dir):
        mock_st_cls.return_value = _make_mock_model(384)
        service = IndustrialEmbeddingService(cache_dir=temp_cache_dir, device="cpu", use_cache=True)

        service.embed_single("hello world")
        calls_after_first = service.model.encode.call_count

        service.embed_single("hello world")
        calls_after_second = service.model.encode.call_count

        assert calls_after_second == calls_after_first, "Second call should hit cache, not re-encode"


class TestEmbedBatch:

    @patch("app.services.embedding_service.SentenceTransformer")
    def test_empty_list_returns_empty_list(self, mock_st_cls, temp_cache_dir):
        mock_st_cls.return_value = _make_mock_model(384)
        service = IndustrialEmbeddingService(cache_dir=temp_cache_dir, device="cpu")
        assert service.embed_batch([]) == []

    @patch("app.services.embedding_service.SentenceTransformer")
    def test_batch_returns_one_vector_per_text(self, mock_st_cls, temp_cache_dir):
        mock_st_cls.return_value = _make_mock_model(384)
        service = IndustrialEmbeddingService(cache_dir=temp_cache_dir, device="cpu", use_cache=False)
        result = service.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert all(len(v) == 384 for v in result)


class TestCacheManagement:

    @patch("app.services.embedding_service.SentenceTransformer")
    def test_clear_cache_removes_files(self, mock_st_cls, temp_cache_dir):
        mock_st_cls.return_value = _make_mock_model(384)
        service = IndustrialEmbeddingService(cache_dir=temp_cache_dir, device="cpu", use_cache=True)
        service.embed_single("some text")
        assert len(list(temp_cache_dir.glob("*.pkl"))) > 0

        deleted = service.clear_cache()
        assert deleted > 0
        assert len(list(temp_cache_dir.glob("*.pkl"))) == 0


class TestFactory:

    @patch("app.services.embedding_service.SentenceTransformer")
    def test_create_multilingual_uses_bge_m3(self, mock_st_cls, temp_cache_dir):
        mock_st_cls.return_value = _make_mock_model(1024)
        with patch("app.services.embedding_service.EMBEDDING_CACHE_DIR", temp_cache_dir):
            service = EmbeddingServiceFactory.create_multilingual()
        assert service.model_name == "BAAI/bge-m3"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))