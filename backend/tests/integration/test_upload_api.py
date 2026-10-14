"""
Integration tests for POST /upload via FastAPI TestClient. Auto-skipped if
Qdrant isn't reachable.
"""
import sys
from io import BytesIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_upload_rejects_unsupported_extension(test_client):
    response = test_client.post(
        "/upload",
        files={"file": ("test.exe", BytesIO(b"fake binary"), "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_rejects_empty_file(test_client):
    response = test_client.post(
        "/upload",
        files={"file": ("empty.txt", BytesIO(b""), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_txt_file_indexes_chunks(test_client):
    """
    Regression: ingest() used to be a stub that reported success without
    indexing anything (`print(...); return 1`). Confirms /upload produces
    chunks > 0 for a real small text file end-to-end.
    """
    content = b"This is a test document about integration testing in FastAPI applications."
    response = test_client.post(
        "/upload",
        files={"file": ("integration_test.txt", BytesIO(content), "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["chunks"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))