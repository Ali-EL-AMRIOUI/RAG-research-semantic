"""
Unit tests for IndustrialDocumentLoader.

unstructured is stubbed via sys.modules — not installed in this
environment. OCRService is mocked (already tested for real in
test_ocr_service.py). The cache round-trip is tested with REAL temp files
on disk, no mocking — that logic is pure Python and worth testing as-is.
"""
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

sys.modules.setdefault("unstructured", MagicMock())
sys.modules.setdefault("unstructured.partition", MagicMock())
sys.modules.setdefault("unstructured.partition.auto", MagicMock())

from app.utils.document_loader import DocumentChunk, IndustrialDocumentLoader


@pytest.fixture
def temp_cache_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_file():
    d = tempfile.mkdtemp()
    file_path = Path(d) / "sample.txt"
    file_path.write_text("some content", encoding="utf-8")
    yield file_path
    shutil.rmtree(d, ignore_errors=True)


class TestUnsupportedFormat:

    def test_unsupported_extension_raises_value_error(self, temp_cache_dir):
        loader = IndustrialDocumentLoader(cache_dir=temp_cache_dir)
        d = tempfile.mkdtemp()
        try:
            real_file = Path(d) / "file.xyz"
            real_file.write_text("some content")
            with pytest.raises(ValueError):
                loader.load_document(real_file)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestMissingFile:

    def test_nonexistent_file_raises_file_not_found(self, temp_cache_dir):
        loader = IndustrialDocumentLoader(cache_dir=temp_cache_dir)
        with pytest.raises(FileNotFoundError):
            loader.load_document(Path("/tmp/does_not_exist_12345.pdf"))


class TestCacheRoundTrip:
    """Real disk I/O, no mocking — this logic is pure Python and worth
    testing as-is rather than through a mock."""

    def test_save_then_load_returns_same_chunks(self, temp_cache_dir, temp_file):
        loader = IndustrialDocumentLoader(cache_dir=temp_cache_dir, use_cache=True)

        chunks = [
            DocumentChunk(text="first chunk", metadata={"source": "sample.txt"}, chunk_index=0),
            DocumentChunk(text="second chunk", metadata={"source": "sample.txt"}, chunk_index=1),
        ]
        loader._save_to_cache(temp_file, chunks)

        loaded = loader._load_from_cache(temp_file)

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].text == "first chunk"
        assert loaded[1].text == "second chunk"

    def test_cache_miss_returns_none(self, temp_cache_dir, temp_file):
        loader = IndustrialDocumentLoader(cache_dir=temp_cache_dir, use_cache=True)
        result = loader._load_from_cache(temp_file)
        assert result is None

    def test_disabled_cache_never_saves(self, temp_cache_dir, temp_file):
        loader = IndustrialDocumentLoader(cache_dir=temp_cache_dir, use_cache=False)
        chunks = [DocumentChunk(text="x", metadata={}, chunk_index=0)]
        loader._save_to_cache(temp_file, chunks)
        assert list(temp_cache_dir.glob("*.json")) == []

    def test_corrupted_cache_file_returns_none_not_exception(self, temp_cache_dir, temp_file):
        loader = IndustrialDocumentLoader(cache_dir=temp_cache_dir, use_cache=True)
        cache_path = loader._get_cache_path(temp_file)
        cache_path.write_text("{not valid json", encoding="utf-8")

        result = loader._load_from_cache(temp_file)
        assert result is None


class TestChunkTextOverlap:
    """
    _chunk_text() has its own manual overlap logic, separate from
    chunking.py's IndustrialChunker — regression-tested the same way.
    """

    def test_no_chunk_exceeds_chunk_size(self, temp_cache_dir):
        loader = IndustrialDocumentLoader(cache_dir=temp_cache_dir, chunk_size=100, chunk_overlap=20)
        text = "\n\n".join([f"Paragraph number {i} with some content in it." for i in range(20)])
        chunks = loader._chunk_text(text)
        for c in chunks:
            assert len(c.text) <= 100 + 5

    def test_consecutive_chunks_share_overlap_text(self, temp_cache_dir):
        loader = IndustrialDocumentLoader(cache_dir=temp_cache_dir, chunk_size=150, chunk_overlap=40)
        paragraphs = [f"Sentence number {i} giving this paragraph some real content." for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = loader._chunk_text(text)

        assert len(chunks) >= 2
        found_overlap = False
        for i in range(len(chunks) - 1):
            end_words = set(chunks[i].text[-30:].split())
            start_words = set(chunks[i + 1].text[:60].split())
            if end_words & start_words:
                found_overlap = True
                break
        assert found_overlap, "No shared words found between consecutive chunks — overlap not applied"

    def test_empty_text_returns_no_chunks(self, temp_cache_dir):
        loader = IndustrialDocumentLoader(cache_dir=temp_cache_dir)
        assert loader._chunk_text("") == []


class TestOCRDelegation:

    def test_extract_text_from_image_delegates_to_ocr_service(self, temp_cache_dir, temp_file):
        loader = IndustrialDocumentLoader(cache_dir=temp_cache_dir)
        loader.ocr_service = MagicMock()
        loader.ocr_service.extract_text.return_value = "extracted text"

        result = loader._extract_text_from_image(temp_file)

        assert result == "extracted text"
        loader.ocr_service.extract_text.assert_called_once_with(temp_file)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))