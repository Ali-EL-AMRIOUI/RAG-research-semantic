"""
End-to-end pipeline test: ingest real text, then retrieve it. Auto-skipped
if Qdrant isn't reachable.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.ingestion_service import IngestionService  # noqa: E402
from app.core.config import QDRANT_HOST, QDRANT_PORT  # noqa: E402

TEST_COLLECTION = "integration_test_pipeline"


@pytest.fixture
def ingestion(qdrant_available):
    service = IngestionService(
        qdrant_host=QDRANT_HOST, qdrant_port=QDRANT_PORT, collection_name=TEST_COLLECTION,
    )
    yield service
    service.vector_store.delete_collection()


def test_ingest_text_then_search_finds_it(ingestion):
    inserted = ingestion.ingest_text(
        "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        source="test_fact.txt",
    )
    assert inserted > 0

    results = ingestion.vector_store.search_vector(
        query_vector=ingestion.embedder.embed_query("Where is the Eiffel Tower?"),
        limit=3,
    )
    assert len(results) > 0
    assert "Eiffel" in results[0].content


def test_ingest_empty_text_indexes_nothing(ingestion):
    inserted = ingestion.ingest_text("", source="empty.txt")
    assert inserted == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))