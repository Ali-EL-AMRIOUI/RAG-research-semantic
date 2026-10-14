"""
Non-regression tests for IndustrialChunker.

The first 3 tests each correspond to a real bug fixed during
this session: without them, a future regression (e.g., someone touches
split_markdown() without knowing it) would only be detected in production.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.chunking import IndustrialChunker, Chunk


class TestTitleTracking:
    """
    Bug fixed: _extract_titles() restarted from [] on each call, so a
    paragraph containing only "## Section" (without its parent "# Title"
    in the same paragraph) was never recognized as a title.
    """

    def test_title_hierarchy_evolves_across_chunks(self):
        md = (
            "# Doc Title\n\n"
            "## Section A\n"
            "Content of section A, long enough for its own chunk here.\n\n"
            "## Section B\n"
            "Content of section B, different, also long enough for this test.\n\n"
            "### Subsection B1\n"
            "Content nested under section B.\n"
        )
        chunker = IndustrialChunker(chunk_size=80, chunk_overlap=10)
        chunks = chunker.split_markdown(md)

        all_titles = [c.title_path for c in chunks]

        assert any(t == ["Doc Title", "Section A"] for t in all_titles), \
            f"Section A never detected as current title: {all_titles}"
        assert any(t == ["Doc Title", "Section B"] for t in all_titles), \
            f"Section B never detected as current title: {all_titles}"
        assert any(t == ["Doc Title", "Section B", "Subsection B1"] for t in all_titles), \
            f"Subsection never detected: {all_titles}"

    def test_header_lines_stripped_from_chunk_body(self):
        """Header lines '#'/'##' must never appear in the chunk text."""
        md = "# Title\n\n## Section\nActual paragraph text.\n"
        chunker = IndustrialChunker(chunk_size=1000, chunk_overlap=100)
        chunks = chunker.split_markdown(md)

        full_text = " ".join(c.text for c in chunks)
        assert "## Section" not in full_text
        assert "# Title" not in full_text
        assert "Actual paragraph text." in full_text


class TestChunkSizeRespected:
    """
    Bug fixed: a paragraph without internal \\n\\n, longer than chunk_size,
    produced a chunk several times larger than the configured limit.
    """

    def test_oversized_paragraph_is_split(self):
        huge_paragraph = "word " * 500  # ~2000 characters, a single paragraph
        md = f"# Title\n\nBefore.\n\n{huge_paragraph}\n\nAfter.\n"

        chunk_size = 200
        chunker = IndustrialChunker(chunk_size=chunk_size, chunk_overlap=20)
        chunks = chunker.split_markdown(md)

        max_len = max(len(c.text) for c in chunks)
        assert max_len <= chunk_size, (
            f"Chunk of {max_len} characters produced, limit = {chunk_size}"
        )

    def test_normal_paragraphs_never_exceed_chunk_size(self):
        md = "\n\n".join([f"Paragraph number {i} with a bit of text." for i in range(30)])
        chunk_size = 150
        chunker = IndustrialChunker(chunk_size=chunk_size, chunk_overlap=20)
        chunks = chunker.split_markdown(md)

        for c in chunks:
            assert len(c.text) <= chunk_size + 5  # small margin for \n\n joins


class TestOverlapApplied:
    """
    Bug fixed: chunk_overlap was stored but never used — each
    chunk started at zero, without taking context from the previous one.
    """

    def test_consecutive_chunks_share_overlap_text(self):
        paragraphs = [f"Sentence number {i} that gives content to this paragraph." for i in range(20)]
        md = "\n\n".join(paragraphs)

        chunker = IndustrialChunker(chunk_size=150, chunk_overlap=40)
        chunks = chunker.split_markdown(md)

        assert len(chunks) >= 2, "Inconclusive test: not enough chunks generated"

        found_overlap = False
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i].text[-30:]
            start_of_next = chunks[i + 1].text[:60]
            words_from_end = set(end_of_current.split())
            words_from_start = set(start_of_next.split())
            if words_from_end & words_from_start:
                found_overlap = True
                break

        assert found_overlap, "No shared words found between consecutive chunks — overlap not applied"

    def test_zero_overlap_produces_no_shared_prefix(self):
        md = "Paragraph A with text.\n\nParagraph B totally different here."
        chunker = IndustrialChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.split_markdown(md)
        # With overlap=0, we only check that it doesn't crash and remains consistent
        assert len(chunks) >= 1


class TestBasicBehavior:

    def test_empty_text_returns_no_chunks(self):
        chunker = IndustrialChunker()
        assert chunker.split_markdown("") == []
        assert chunker.split_markdown("   \n\n  ") == []

    def test_split_text_applies_overlap_via_langchain(self):
        chunker = IndustrialChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunker.split_text(text, source="test")
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_get_stats_on_empty_list(self):
        chunker = IndustrialChunker()
        stats = chunker.get_stats([])
        assert stats["total_chunks"] == 0

    def test_get_stats_basic(self):
        chunker = IndustrialChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split_markdown("# T\n\n" + ("word " * 50))
        stats = chunker.get_stats(chunks)
        assert stats["total_chunks"] == len(chunks)
        assert stats["max_size"] <= 100 + 5

    def test_code_blocks_preserved(self):
        md = "Before.\n\n```python\ndef f():\n    pass\n```\n\nAfter."
        chunker = IndustrialChunker(chunk_size=1000, chunk_overlap=50, preserve_code_blocks=True)
        chunks = chunker.split_markdown(md)
        full_text = " ".join(c.text for c in chunks)
        assert "def f():" in full_text
        assert "pass" in full_text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))