"""
Integration tests for POST /ask via FastAPI TestClient. Auto-skipped if
Qdrant isn't reachable (test_client fixture) or GROQ_API_KEY isn't set
(groq_key_available fixture).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_ask_empty_question_returns_400(test_client):
    response = test_client.post("/ask", json={"question": ""})
    assert response.status_code == 400


def test_ask_returns_answer_and_sources(test_client, groq_key_available):
    response = test_client.post("/ask", json={"question": "What is this system about?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data.get("source_documents", []), list)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))