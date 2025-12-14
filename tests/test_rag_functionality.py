"""
Basic tests for RAG functionality.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the backend src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.src.models.query import QueryRequest
from backend.src.services.rag_service import rag_service


@pytest.mark.asyncio
async def test_rag_service_basic_response():
    """
    Test that the RAG service can generate a basic response.
    """
    # Create a mock query request
    request = QueryRequest(
        query="What is robotics?",
        session_id="test-session-123",
        language="en"
    )

    # Mock the retrieval service to return some dummy chunks
    with patch.object(rag_service.retrieval_service, 'retrieve_chunks_by_request') as mock_retrieve:
        # Mock a source chunk
        from backend.src.models.query import SourceChunk
        mock_chunk = SourceChunk(
            chunk_id="test-chunk-123",
            chapter_number=1,
            section_title="Introduction",
            source_file_path="docs/intro.md",
            confidence_score=0.8
        )

        mock_retrieve.return_value = [mock_chunk]

        # Mock the retrieval service's get_chunk_content method
        with patch.object(rag_service.retrieval_service, 'get_chunk_content') as mock_get_content:
            mock_get_content.return_value = "Robotics is a field that deals with the design, construction, and operation of robots."

            # Call the get_response method
            response_text, source_chunks = await rag_service.get_response(request)

            # Assert that the response was generated
            assert isinstance(response_text, str)
            assert len(response_text) > 0
            assert len(source_chunks) == 1
            assert source_chunks[0].chunk_id == "test-chunk-123"


@pytest.mark.asyncio
async def test_rag_service_with_selected_text():
    """
    Test that the RAG service can handle selected text queries.
    """
    # Create a mock query request with selected text
    request = QueryRequest(
        query="Explain this concept",
        selected_text="The concept of artificial intelligence",
        session_id="test-session-123",
        language="en"
    )

    # Mock the retrieval service to return some dummy chunks
    with patch.object(rag_service.retrieval_service, 'retrieve_chunks_with_selected_text') as mock_retrieve:
        # Mock a source chunk
        from backend.src.models.query import SourceChunk
        mock_chunk = SourceChunk(
            chunk_id="test-chunk-123",
            chapter_number=2,
            section_title="AI Concepts",
            source_file_path="docs/ai.md",
            confidence_score=0.9,
            is_selected_text=True
        )

        mock_retrieve.return_value = [mock_chunk]

        # Mock the retrieval service's get_chunk_content method
        with patch.object(rag_service.retrieval_service, 'get_chunk_content') as mock_get_content:
            mock_get_content.return_value = "Artificial intelligence is intelligence demonstrated by machines."

            # Call the get_response_with_selected_text method
            response_text, source_chunks = await rag_service.get_response_with_selected_text(request)

            # Assert that the response was generated
            assert isinstance(response_text, str)
            assert len(response_text) > 0
            assert len(source_chunks) == 1
            assert source_chunks[0].chunk_id == "test-chunk-123"
            assert source_chunks[0].is_selected_text == True


def test_prepare_context():
    """
    Test that context is properly prepared from source chunks.
    """
    from backend.src.models.query import SourceChunk

    # Create mock source chunks
    chunk1 = SourceChunk(
        chunk_id="chunk1",
        chapter_number=1,
        section_title="Introduction",
        source_file_path="docs/intro.md",
        confidence_score=0.8
    )

    chunk2 = SourceChunk(
        chunk_id="chunk2",
        chapter_number=2,
        section_title="Advanced Topics",
        source_file_path="docs/advanced.md",
        confidence_score=0.7
    )

    # Mock the retrieval service's get_chunk_content method
    with patch.object(rag_service.retrieval_service, 'get_chunk_content') as mock_get_content:
        mock_get_content.side_effect = [
            "This is the introduction content.",
            "This is the advanced content."
        ]

        # Call the _prepare_context method
        context = rag_service._prepare_context([chunk1, chunk2])

        # Assert that the context contains the expected content
        assert "Chapter 1, Section 'Introduction'" in context
        assert "This is the introduction content." in context
        assert "Chapter 2, Section 'Advanced Topics'" in context
        assert "This is the advanced content." in context


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])