import pytest


@pytest.fixture
def mock_groq_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")


# TODO: ajouter des mocks pour embedding_service / vector_store
# une fois leur interface confirmée (évite d'appeler les vraies API en test)
