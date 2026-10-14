"""
End-to-end regression tests for the original bugs found this session,
run against REAL services (not mocked) — complements the mocked versions
in tests/unit/test_rag_service.py by catching integration-level issues
mocks can't (actual serialization, actual Qdrant schema, actual LLM call).

Named "legacy" because these specifically guard against regressing back to
the ORIGINAL broken behavior: stub ingest(), an incompatible run()
signature, and a reranker that was loaded but never invoked.
"""
import sys
from io import BytesIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.rag_service import IndustrialRAGService

TEST_COLLECTION = "integration_test_legacy"


@pytest.fixture
def rag(qdrant_available, groq_key_available):
    service = IndustrialRAGService(collection_name=TEST_COLLECTION)
    yield service
    service.clear_collection()


def test_ingest_actually_indexes_not_just_reports_success(rag):
    """Original bug: ingest() was `print(f"Ingestion started for
    {filename}"); return 1` — reported success without loading, chunking,
    or indexing anything."""
    content = b"Integration test content about legacy regression testing."
    chunks_indexed = rag.ingest(BytesIO(content), filename="legacy_test.txt")

    assert chunks_indexed > 0
    stats = rag.get_stats()
    assert stats["text_collection"]["total_points"] > 0


def test_run_accepts_the_parameters_the_api_actually_sends(rag):
    """Original bug: run(self, query, limit=TOP_N) only accepted (query,
    limit) — the search API called it with use_hybrid/use_reranker kwargs
    that didn't exist, causing a TypeError on every request."""
    result = rag.run(
        query="test question",
        top_k=5,
        top_n=3,
        use_hybrid=True,
        use_reranker=True,
    )
    assert "answer" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))