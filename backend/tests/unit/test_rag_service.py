"""
Unit tests for IndustrialRAGService.

All heavy dependencies (IngestionService, vector stores, LLM, reranker,
image embedder) are mocked. This file targets the ORCHESTRATION logic —
does ingest() route to the right service based on file type, does run()
accept the parameters callers actually pass, does the reranker actually
get invoked — because these are exactly the bugs found and fixed this
session, not generic sanity checks.
"""
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("qdrant_client", MagicMock())
sys.modules.setdefault("qdrant_client.http", MagicMock())
sys.modules.setdefault("qdrant_client.models", MagicMock())
sys.modules.setdefault("open_clip", MagicMock())
sys.modules.setdefault("langchain_groq", MagicMock())
sys.modules.setdefault("unstructured", MagicMock())
sys.modules.setdefault("unstructured.partition", MagicMock())
sys.modules.setdefault("unstructured.partition.auto", MagicMock())


def _build_service(use_reranker=True):
    """Builds an IndustrialRAGService with every heavy dependency mocked."""
    with patch("app.services.rag_service.IngestionService") as mock_ingestion_cls, \
         patch("app.services.rag_service.IndustrialVectorStore") as mock_vs_cls, \
         patch("app.services.rag_service.LLMService") as mock_llm_cls, \
         patch("app.services.rag_service.IndustrialReranker") as mock_reranker_cls:

        mock_ingestion = MagicMock()
        mock_ingestion.embedder = MagicMock(get_dimension=MagicMock(return_value=384))
        mock_ingestion.vector_store = MagicMock(use_hybrid_search=True)
        mock_ingestion.vector_store.search_hybrid.return_value = []
        mock_ingestion.vector_store.search_vector.return_value = []
        mock_ingestion_cls.return_value = mock_ingestion

        mock_vs_cls.return_value.search_vector.return_value = []  # image_vector_store default

        mock_llm = MagicMock()
        mock_llm.is_available = True
        mock_llm.last_usage = {}
        mock_llm.generate.return_value = "Generated answer"
        mock_llm_cls.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker_cls.return_value = mock_reranker

        from app.services.rag_service import IndustrialRAGService
        service = IndustrialRAGService(use_reranker=use_reranker)

        return service, mock_ingestion, mock_llm, mock_reranker


class TestIngestRouting:
    """
    Regression: ingest() used to be `print(...); return 1` — reported
    success without ever loading, chunking, or indexing anything.
    """

    def test_pdf_delegates_to_ingestion_service(self):
        service, mock_ingestion, _, _ = _build_service()
        mock_ingestion.ingest_pdf_stream.return_value = 5

        result = service.ingest(BytesIO(b"fake pdf bytes"), filename="report.pdf")

        assert result == 5
        mock_ingestion.ingest_pdf_stream.assert_called_once()

    def test_image_delegates_to_image_embedder_not_ingestion_service(self):
        service, mock_ingestion, _, _ = _build_service()
    
        mock_image_embedder = MagicMock()
        mock_image_embedder.embed_image.return_value = [0.1] * 512
        service._image_embedder = mock_image_embedder
    
        with (
            patch.object(service, "_caption_image", return_value="mocked caption"),
            patch.object(service, "_store_image_point", return_value=1)
        ):
            result = service.ingest(BytesIO(b"fake image bytes"), filename="photo.jpg")
    
        assert result == 1
        mock_ingestion.ingest_pdf_stream.assert_not_called()
        mock_image_embedder.embed_image.assert_called_once()

    def test_ingestion_error_returns_zero_not_exception(self):
        service, mock_ingestion, _, _ = _build_service()
        mock_ingestion.ingest_pdf_stream.side_effect = Exception("qdrant connection refused")

        result = service.ingest(BytesIO(b"fake pdf bytes"), filename="report.pdf")

        assert result == 0


class TestRunSignature:
    """
    Regression: run() originally only accepted (query, limit) — search.py
    called it with use_hybrid/use_reranker kwargs that didn't exist,
    causing a TypeError on every /search/rag request.
    """

    def test_run_accepts_top_k_top_n_use_hybrid_use_reranker(self):
        service, _mock_ingestion, _mock_llm, _ = _build_service()

        result = service.run(
            query="test question",
            top_k=10,
            top_n=3,
            use_hybrid=True,
            use_reranker=False,
        )

        assert "answer" in result

    def test_run_with_no_relevant_chunks_still_returns_answer(self):
        service, _mock_ingestion, mock_llm, _ = _build_service()
        mock_llm.generate.return_value = "I don't have information about that."

        result = service.run(query="unrelated question")

        assert result["source_documents"] == []
        assert result["answer"] == "I don't have information about that."


class TestRerankerActuallyInvoked:
    """
    Regression: the reranker was instantiated in __init__ (real model load
    cost paid) but never called anywhere in search()/run().
    """

    def test_reranker_called_when_enabled_and_results_present(self):
        service, mock_ingestion, _, mock_reranker = _build_service(use_reranker=True)

        fake_hit = MagicMock()
        fake_hit.content = "some content"
        fake_hit.score = 0.9
        fake_hit.metadata = {"filename": "doc.pdf", "page": 1}
        mock_ingestion.vector_store.search_hybrid.return_value = [fake_hit]

        mock_reranker.rerank_search_results.return_value = [
            {"text": "some content", "filename": "doc.pdf", "score": 0.9, "metadata": {}}
        ]

        service.search("test query", use_reranker=True)

        mock_reranker.rerank_search_results.assert_called_once()

    def test_reranker_not_called_when_disabled(self):
        service, mock_ingestion, _, mock_reranker = _build_service(use_reranker=True)

        fake_hit = MagicMock()
        fake_hit.content = "some content"
        fake_hit.score = 0.9
        fake_hit.metadata = {"filename": "doc.pdf", "page": 1}
        mock_ingestion.vector_store.search_hybrid.return_value = [fake_hit]

        service.search("test query", use_reranker=False)

        mock_reranker.rerank_search_results.assert_not_called()

    def test_reranker_not_called_when_no_results(self):
        service, _mock_ingestion, _, mock_reranker = _build_service(use_reranker=True)
        # search_hybrid already defaults to [] in _build_service

        service.search("test query", use_reranker=True)

        mock_reranker.rerank_search_results.assert_not_called()


class TestLLMUnavailable:

    def test_run_returns_error_message_when_llm_unavailable_not_exception(self):
        service, _mock_ingestion, mock_llm, _ = _build_service()
        mock_llm.generate.side_effect = RuntimeError("LLM not initialized (check GROQ_API_KEY)")

        result = service.run(query="test question")

        assert "answer" in result
        assert "sorry" in result["answer"].lower() or "error" in result["answer"].lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))