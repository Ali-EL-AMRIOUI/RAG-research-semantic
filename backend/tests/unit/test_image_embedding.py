"""
Unit tests for ImageEmbeddingService.

open_clip and torch are stubbed via sys.modules — not installed in this
environment (multi-GB download). These tests target OUR wrapper logic
(dimension resolution, error handling, fallback), not OpenCLIP itself.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("open_clip", MagicMock())

from app.services.image_embedding_service import ImageEmbeddingService


def _make_mock_open_clip_return(embedding_dim=512):
    """create_model_and_transforms returns (model, _, preprocess) — mock all three."""
    mock_model = MagicMock()
    mock_preprocess = MagicMock()
    return (mock_model, None, mock_preprocess)


class TestInitialization:

    @patch("app.services.image_embedding_service.open_clip")
    def test_default_model_uses_512_dimension(self, mock_open_clip):
        mock_open_clip.create_model_and_transforms.return_value = _make_mock_open_clip_return()
        service = ImageEmbeddingService(device="cpu")
        assert service.dimension == 512

    @patch("app.services.image_embedding_service.open_clip")
    def test_larger_model_uses_768_dimension(self, mock_open_clip):
        mock_open_clip.create_model_and_transforms.return_value = _make_mock_open_clip_return()
        service = ImageEmbeddingService(model_name="ViT-L-14", pretrained="laion2b_s32b_b82k", device="cpu")
        assert service.dimension == 768

    @patch("app.services.image_embedding_service.open_clip")
    def test_load_failure_falls_back_to_default(self, mock_open_clip):
        mock_open_clip.create_model_and_transforms.side_effect = [
            Exception("checkpoint not found"),
            _make_mock_open_clip_return(),
        ]
        service = ImageEmbeddingService(model_name="broken-model", pretrained="broken-checkpoint", device="cpu")
        assert service.model_name == "ViT-B-32"
        assert service.dimension == 512


class TestEmbedImage:

    @patch("app.services.image_embedding_service.open_clip")
    @patch("app.services.image_embedding_service.Image")
    def test_invalid_image_returns_zero_vector_not_exception(self, mock_image_cls, mock_open_clip):
        mock_open_clip.create_model_and_transforms.return_value = _make_mock_open_clip_return()
        mock_image_cls.open.side_effect = Exception("cannot identify image file")

        service = ImageEmbeddingService(device="cpu")
        result = service.embed_image(b"not a real image")

        assert result == [0.0] * 512

    @patch("app.services.image_embedding_service.open_clip")
    def test_batch_calls_embed_image_per_item(self, mock_open_clip):
        mock_open_clip.create_model_and_transforms.return_value = _make_mock_open_clip_return()
        service = ImageEmbeddingService(device="cpu")

        with patch.object(service, "embed_image", return_value=[0.1] * 512) as mock_embed:
            result = service.embed_images_batch([b"img1", b"img2", b"img3"])

        assert mock_embed.call_count == 3
        assert len(result) == 3


class TestEmbedTextQuery:
    """
    Regression: rag_service.py used to query the image collection with the
    sentence-transformers text vector (384-dim) instead of CLIP's own text
    encoder (512-dim, same joint space as embed_image()). Caused a Qdrant
    dimension mismatch error, silently swallowed by a try/except.
    """

    @patch("app.services.image_embedding_service.open_clip")
    def test_text_query_returns_same_dimension_as_image_embeddings(self, mock_open_clip):
        mock_open_clip.create_model_and_transforms.return_value = _make_mock_open_clip_return()
        mock_open_clip.get_tokenizer.return_value = MagicMock(return_value=MagicMock(to=MagicMock(return_value="tokens")))

        service = ImageEmbeddingService(device="cpu")

        import numpy as np
        fake_features = MagicMock()
        fake_features.norm.return_value = MagicMock()
        fake_features.__truediv__ = MagicMock(return_value=MagicMock(
            cpu=MagicMock(return_value=MagicMock(
                numpy=MagicMock(return_value=np.ones(512))
            ))
        ))
        service.model.encode_text.return_value = fake_features

        result = service.embed_text_query("a photo of a cat")

        assert len(result) == service.dimension == 512

    @patch("app.services.image_embedding_service.open_clip")
    def test_text_query_error_returns_zero_vector_not_exception(self, mock_open_clip):
        mock_open_clip.create_model_and_transforms.return_value = _make_mock_open_clip_return()
        mock_open_clip.get_tokenizer.return_value = MagicMock(side_effect=Exception("tokenizer error"))

        service = ImageEmbeddingService(device="cpu")
        service.tokenizer = MagicMock(side_effect=Exception("tokenizer error"))

        result = service.embed_text_query("some query")

        assert result == [0.0] * 512


class TestGetDimension:

    @patch("app.services.image_embedding_service.open_clip")
    def test_get_dimension_matches_loaded_model(self, mock_open_clip):
        mock_open_clip.create_model_and_transforms.return_value = _make_mock_open_clip_return()
        service = ImageEmbeddingService(device="cpu")
        assert service.get_dimension() == service.dimension


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))