"""
Unit tests for LLMService.

ChatGroq is mocked: we do not test Groq itself (outside of our control),
we test OUR logic — missing API key management, usage capture,
error propagation.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.llm_service import LLMService


class TestLLMServiceInitialization:

    def test_no_api_key_marks_unavailable(self):
        service = LLMService(api_key="")
        assert service.is_available is False

    @patch("app.services.llm_service.ChatGroq")
    def test_valid_api_key_marks_available(self, mock_chatgroq):
        mock_chatgroq.return_value = MagicMock()
        service = LLMService(api_key="fake-key-for-test")
        assert service.is_available is True
        mock_chatgroq.assert_called_once()

    @patch("app.services.llm_service.ChatGroq")
    def test_chatgroq_init_failure_marks_unavailable(self, mock_chatgroq):
        mock_chatgroq.side_effect = Exception("network error")
        service = LLMService(api_key="fake-key-for-test")
        assert service.is_available is False


class TestLLMServiceGenerate:

    def test_generate_raises_when_unavailable(self):
        service = LLMService(api_key="")
        with pytest.raises(RuntimeError):
            service.generate("test prompt")

    @patch("app.services.llm_service.ChatGroq")
    def test_generate_returns_response_content(self, mock_chatgroq):
        mock_response = MagicMock()
        mock_response.content = "Generated response"
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_chatgroq.return_value = mock_client

        service = LLMService(api_key="fake-key-for-test")
        answer = service.generate("What is the capital of France?")

        assert answer == "Generated response"
        mock_client.invoke.assert_called_once_with("What is the capital of France?")

    @patch("app.services.llm_service.ChatGroq")
    def test_last_usage_captured_after_generate(self, mock_chatgroq):
        """Regression: last_usage must reflect the REAL usage_metadata of the response,
        not remain {} after a successful call."""
        mock_response = MagicMock()
        mock_response.content = "OK"
        mock_response.usage_metadata = {"input_tokens": 42, "output_tokens": 7}

        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_chatgroq.return_value = mock_client

        service = LLMService(api_key="fake-key-for-test")
        assert service.last_usage == {}  # nothing before the first call

        service.generate("prompt")
        assert service.last_usage == {"input_tokens": 42, "output_tokens": 7}

    @patch("app.services.llm_service.ChatGroq")
    def test_missing_usage_metadata_defaults_to_empty_dict(self, mock_chatgroq):
        mock_response = MagicMock(spec=["content"])  # no usage_metadata attribute
        mock_response.content = "OK"

        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_chatgroq.return_value = mock_client

        service = LLMService(api_key="fake-key-for-test")
        service.generate("prompt")
        assert service.last_usage == {}


class TestGetModelInfo:

    def test_model_info_reflects_availability(self):
        service = LLMService(api_key="")
        info = service.get_model_info()
        assert info["available"] is False
        assert "model_name" in info


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))