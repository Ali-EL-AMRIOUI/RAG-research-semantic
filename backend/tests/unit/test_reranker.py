"""
Unit tests for IndustrialReranker.

CrossEncoder (sentence_transformers) is stubbed via sys.modules — not
installed in this environment (heavy download). These tests target OUR
logic (sorting, top_n, threshold filtering, fallback), not CrossEncoder
itself.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("torch", MagicMock())

from app.services.rerank_service import IndustrialReranker  # noqa: E402


def _make_mock_cross_encoder(scores):
    """Fake CrossEncoder whose .predict() returns a fixed score sequence, in order."""
    mock = MagicMock()
    mock.predict.return_value = scores
    return mock


class TestRerankOrdering:

    @patch("app.services.rerank_service.CrossEncoder")
    def test_results_sorted_by_score_descending(self, mock_ce_cls):
        mock_ce_cls.return_value = _make_mock_cross_encoder([0.2, 0.9, 0.5])
        reranker = IndustrialReranker(device="cpu")

        docs = ["low relevance", "high relevance", "medium relevance"]
        result = reranker.rerank("query", docs, top_n=3)

        assert result == ["high relevance", "medium relevance", "low relevance"]

    @patch("app.services.rerank_service.CrossEncoder")
    def test_top_n_limits_results(self, mock_ce_cls):
        mock_ce_cls.return_value = _make_mock_cross_encoder([0.1, 0.9, 0.5, 0.3])
        reranker = IndustrialReranker(device="cpu")

        docs = ["a", "b", "c", "d"]
        result = reranker.rerank("query", docs, top_n=2)

        assert len(result) == 2
        assert result[0] == "b"  # score 0.9

    @patch("app.services.rerank_service.CrossEncoder")
    def test_empty_documents_returns_empty_list(self, mock_ce_cls):
        mock_ce_cls.return_value = _make_mock_cross_encoder([])
        reranker = IndustrialReranker(device="cpu")
        assert reranker.rerank("query", []) == []

    @patch("app.services.rerank_service.CrossEncoder")
    def test_threshold_filters_low_scores(self, mock_ce_cls):
        mock_ce_cls.return_value = _make_mock_cross_encoder([0.1, 0.9, 0.5])
        reranker = IndustrialReranker(device="cpu")

        docs = ["low", "high", "medium"]
        result = reranker.rerank("query", docs, top_n=3, threshold=0.4)

        assert result == ["high", "medium"]  # "low" (0.1) filtered out


class TestRerankWithMetadata:

    @patch("app.services.rerank_service.CrossEncoder")
    def test_preserves_original_metadata_and_adds_score(self, mock_ce_cls):
        mock_ce_cls.return_value = _make_mock_cross_encoder([0.9, 0.2])
        reranker = IndustrialReranker(device="cpu")

        documents = [
            {"text": "doc one", "filename": "a.pdf"},
            {"text": "doc two", "filename": "b.pdf"},
        ]
        result = reranker.rerank_with_metadata("query", documents, top_n=2)

        assert result[0]["filename"] == "a.pdf"
        assert "rerank_score" in result[0]


class TestRerankSearchResults:

    @patch("app.services.rerank_service.CrossEncoder")
    def test_detects_text_key_automatically(self, mock_ce_cls):
        mock_ce_cls.return_value = _make_mock_cross_encoder([0.5])
        reranker = IndustrialReranker(device="cpu")

        search_results = [{"text": "content here", "filename": "doc.pdf", "score": 0.8}]
        result = reranker.rerank_search_results("query", search_results, top_n=1)

        assert len(result) == 1
        assert result[0]["filename"] == "doc.pdf"


class TestModelFallback:

    @patch("app.services.rerank_service.CrossEncoder")
    def test_load_failure_falls_back_to_default_model(self, mock_ce_cls):
        mock_ce_cls.side_effect = [Exception("model not found"), _make_mock_cross_encoder([])]
        reranker = IndustrialReranker(model_name="some/broken-reranker", device="cpu")
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


class TestModelRecommendation:

    def test_multilingual_recommendation(self):
        assert IndustrialReranker.get_model_recommendation("multilingual") == "BAAI/bge-reranker-v2-m3"

    def test_unknown_requirement_falls_back_to_fast(self):
        assert IndustrialReranker.get_model_recommendation("nonexistent") == "cross-encoder/ms-marco-MiniLM-L-6-v2"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))