"""
Shared fixtures for integration tests. These need real infrastructure
(Qdrant reachable, valid GROQ_API_KEY) and skip automatically when it's
missing — rather than failing hard, which would break CI runs on a machine
without a live Qdrant instance or LLM key.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.config import QDRANT_HOST, QDRANT_PORT, GROQ_API_KEY  # noqa: E402


def _qdrant_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def qdrant_available():
    if not _qdrant_reachable():
        pytest.skip(f"Qdrant not reachable at {QDRANT_HOST}:{QDRANT_PORT} — run `docker start qdrant`")
    return True


@pytest.fixture(scope="session")
def groq_key_available():
    if not GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set in .env — required for LLM-dependent integration tests")
    return True


@pytest.fixture(scope="session")
def test_client(qdrant_available):
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)