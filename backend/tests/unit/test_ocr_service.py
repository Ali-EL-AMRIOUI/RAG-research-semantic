"""
Unit tests for OCRService.

TestOCRServiceRealTesseract makes a real call to tesseract (installed in
this environment) on an image generated on the fly — not mocked, proof
that extraction actually works end-to-end. The rest covers error
handling (invalid file, missing tesseract) via mocks.
"""
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.ocr_service import OCRService


def _make_text_image(text: str) -> BytesIO:
    """Generates a PNG image containing text, to test OCR realistically."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), text, fill="black")

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestOCRServiceRealTesseract:
    """Real tests, without mocks — require the tesseract binary installed."""

    def test_is_available_true_when_tesseract_installed(self):
        service = OCRService()
        assert service.is_available() is True

    def test_extract_text_from_generated_image(self):
        service = OCRService(languages="eng")
        image_buf = _make_text_image("HELLO WORLD")
        result = service.extract_text(image_buf)
        assert "HELLO" in result.upper() or "WORLD" in result.upper()

    def test_extract_text_from_bytes(self):
        service = OCRService(languages="eng")
        image_buf = _make_text_image("TEST")
        raw_bytes = image_buf.getvalue()
        result = service.extract_text(raw_bytes)
        assert isinstance(result, str)


class TestOCRServiceErrorHandling:

    def test_invalid_image_returns_empty_string_not_exception(self):
        service = OCRService()
        garbage = BytesIO(b"this is not an image")
        result = service.extract_text(garbage)
        assert result == ""

    def test_nonexistent_path_returns_empty_string(self):
        service = OCRService()
        result = service.extract_text("/tmp/does_not_exist_12345.png")
        assert result == ""

    @patch("app.services.ocr_service.pytesseract")
    def test_is_available_false_when_tesseract_missing(self, mock_pytesseract):
        mock_pytesseract.get_tesseract_version.side_effect = Exception("tesseract not found")
        service = OCRService()
        assert service.is_available() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))