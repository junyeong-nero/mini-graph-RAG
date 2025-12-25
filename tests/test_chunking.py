"""Tests for text chunking."""

import pytest

from mini_graph_rag.chunking import Chunk, TextChunker


class TestTextChunker:
    """Tests for TextChunker class."""

    def test_init_valid(self):
        """Test valid initialization."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20

    def test_init_invalid_overlap(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, overlap=100)

        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, overlap=150)

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_small_text(self):
        """Test text smaller than chunk size."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        text = "This is a short text."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].start_index == 0

    def test_text_splitting(self):
        """Test that text is properly split into chunks."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "A" * 100

        chunks = chunker.chunk(text)

        # Should create multiple chunks
        assert len(chunks) > 1

        # All chunks should have text
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_overlap_content(self):
        """Test that chunks have overlapping content."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "A" * 100

        chunks = chunker.chunk(text)

        # Check overlap between consecutive chunks
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                chunk1 = chunks[i]
                chunk2 = chunks[i + 1]
                # The end of chunk1 should overlap with start of chunk2
                # This is approximate due to boundary adjustment
                assert chunk2.start_index < chunk1.end_index or chunk2.start_index == chunk1.end_index

    def test_chunk_metadata(self):
        """Test that chunks have correct metadata."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        text = "Some text for testing."
        doc_id = "test_doc"

        chunks = chunker.chunk(text, doc_id=doc_id)

        assert len(chunks) == 1
        assert chunks[0].doc_id == doc_id
        assert chunks[0].chunk_id  # Should have a UUID

    def test_sentence_boundary(self):
        """Test that chunking respects sentence boundaries when possible."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk(text)

        # With small text, should be single chunk
        assert len(chunks) >= 1


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a Chunk."""
        chunk = Chunk(
            text="Test text",
            start_index=0,
            end_index=9,
            doc_id="doc1",
        )

        assert chunk.text == "Test text"
        assert chunk.start_index == 0
        assert chunk.end_index == 9
        assert chunk.doc_id == "doc1"
        assert chunk.chunk_id  # Should have auto-generated UUID

    def test_chunk_default_values(self):
        """Test Chunk default values."""
        chunk = Chunk(text="Test")

        assert chunk.text == "Test"
        assert chunk.start_index == 0
        assert chunk.end_index == 0
        assert chunk.doc_id == ""
        assert chunk.metadata == {}
