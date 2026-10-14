"""
Integration tests against a real Qdrant instance. Auto-skipped if Qdrant
isn't reachable (see conftest.py's qdrant_available fixture).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.db.vector_store import IndustrialVectorStore  # noqa: E402
from app.core.config import QDRANT_HOST, QDRANT_PORT  # noqa: E402

TEST_COLLECTION = "integration_test_collection"


@pytest.fixture
def test_store(qdrant_available):
    store = IndustrialVectorStore(
        host=QDRANT_HOST, port=QDRANT_PORT,
        collection_name=TEST_COLLECTION, vector_dim=4, use_hybrid_search=True,
    )
    yield store
    store.delete_collection()


def test_collection_is_created(test_store):
    stats = test_store.get_collection_stats()
    assert "error" not in stats


def test_upsert_and_search_roundtrip(test_store):
    chunks = ["hello world", "goodbye world"]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    metadata = [{"filename": "a.txt"}, {"filename": "b.txt"}]

    inserted = test_store.upsert_documents(chunks, embeddings, metadata)
    assert inserted == 2

    results = test_store.search_vector(query_vector=[1.0, 0.0, 0.0, 0.0], limit=1)
    assert len(results) == 1
    assert results[0].content == "hello world"


def test_hybrid_search_does_not_crash(test_store):
    """
    Regression: search_hybrid was defined twice in the original file — the
    second definition silently overrode the first, and neither did real
    hybrid search (one was a literal TODO stub). Confirms the real
    dense+sparse RRF implementation runs against actual Qdrant.
    """
    chunks = ["the quick brown fox", "a slow green turtle"]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    metadata = [{"filename": "a.txt"}, {"filename": "b.txt"}]
    test_store.upsert_documents(chunks, embeddings, metadata)

    results = test_store.search_hybrid(query_vector=[1.0, 0.0, 0.0, 0.0], query_text="fox", limit=2)
    assert isinstance(results, list)


def test_delete_by_filter_returns_accurate_count(test_store):
    """Regression: delete_documents used to always return 0 for
    filter-based deletion instead of the real count."""
    chunks = ["content one", "content two"]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    metadata = [{"filename": "target.txt"}, {"filename": "other.txt"}]
    test_store.upsert_documents(chunks, embeddings, metadata)

    deleted = test_store.delete_documents(filter_conditions={"filename": "target.txt"})
    assert deleted == 1


def test_is_healthy_reflects_real_connectivity(test_store):
    """Regression: /health used to report Qdrant as healthy just because
    the Python client object existed, even when Qdrant was unreachable."""
    assert test_store.is_healthy() is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))