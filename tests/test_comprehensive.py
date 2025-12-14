"""
Comprehensive tests for the RAG system covering all functionality.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio

# Add the backend src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.src.models.query import QueryRequest, SourceChunk
from backend.src.services.rag_service import rag_service
from backend.src.services.agent_orchestrator import agent_orchestrator
from backend.src.services.skills_service import skills_service


class TestRAGService:
    """Tests for the RAG service functionality."""

    @pytest.mark.asyncio
    async def test_get_response_basic(self):
        """Test basic response generation."""
        request = QueryRequest(
            query="What is robotics?",
            session_id="test-session-123",
            language="en"
        )

        # Mock the retrieval service
        with patch.object(rag_service.retrieval_service, 'retrieve_chunks_by_request') as mock_retrieve:
            # Mock a source chunk
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
    async def test_get_response_with_selected_text(self):
        """Test response generation with selected text."""
        request = QueryRequest(
            query="Explain this concept",
            selected_text="The concept of artificial intelligence",
            session_id="test-session-123",
            language="en"
        )

        # Mock the retrieval service
        with patch.object(rag_service.retrieval_service, 'retrieve_chunks_with_selected_text') as mock_retrieve:
            # Mock a source chunk
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

    def test_prepare_context(self):
        """Test context preparation."""
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


class TestAgentOrchestrator:
    """Tests for the agent orchestrator functionality."""

    @pytest.mark.asyncio
    async def test_summarizer_agent(self):
        """Test the summarizer agent."""
        result = await agent_orchestrator.execute_agent_task(
            "summarizer",
            "Summarize this text",
            context="This is a long text that needs to be summarized. It contains important information that should be condensed into a shorter form."
        )

        assert result["status"] == "success"
        assert "result" in result
        assert isinstance(result["result"], str)

    @pytest.mark.asyncio
    async def test_translator_agent(self):
        """Test the translator agent."""
        result = await agent_orchestrator.execute_agent_task(
            "translator",
            "Translate this",
            context="Hello, how are you?",
            target_language="ur"
        )

        assert result["status"] == "success"
        assert "result" in result
        assert isinstance(result["result"], str)

    @pytest.mark.asyncio
    async def test_code_validation_agent(self):
        """Test the code validation agent."""
        sample_code = """
        #include <ros/ros.h>
        int main(int argc, char **argv) {
            ros::init(argc, argv, "hello_world");
            ros::NodeHandle nh;
            ROS_INFO("Hello, World!");
            return 0;
        }
        """

        result = await agent_orchestrator.execute_agent_task(
            "code_validator",
            "Validate ROS code",
            context=sample_code
        )

        assert result["status"] == "success"
        assert "result" in result
        assert isinstance(result["result"], str)

    @pytest.mark.asyncio
    async def test_multi_step_reasoning(self):
        """Test multi-step reasoning."""
        result = await agent_orchestrator.execute_multi_step_reasoning(
            "Translate and summarize this",
            context="Artificial intelligence is a wonderful field.",
            agents_sequence=["translator", "summarizer"]
        )

        assert result["status"] == "success"
        assert "results" in result
        assert len(result["results"]) == 2  # translator and summarizer


class TestSkillsService:
    """Tests for the skills service functionality."""

    @pytest.mark.asyncio
    async def test_execute_summarization(self):
        """Test the summarization skill."""
        with patch.object(skills_service.retrieval_service, 'retrieve_chunks_by_request') as mock_retrieve:
            mock_retrieve.return_value = []

            result = await skills_service.execute_summarization(
                "Summarize the content",
                context_text="This is a long text that needs to be summarized. It contains important information that should be condensed."
            )

            assert result["status"] == "success"
            assert "result" in result

    @pytest.mark.asyncio
    async def test_execute_translation(self):
        """Test the translation skill."""
        result = await skills_service.execute_translation(
            "Translate this text",
            target_language="ur",
            context_text="Hello, how are you?"
        )

        assert result["status"] == "success"
        assert "result" in result

    @pytest.mark.asyncio
    async def test_execute_code_validation(self):
        """Test the code validation skill."""
        sample_code = """
        def hello_world():
            print("Hello, World!")
        """

        result = await skills_service.execute_code_validation(sample_code)

        assert result["status"] == "success"
        assert "result" in result


class TestIntegration:
    """Integration tests for multiple components working together."""

    @pytest.mark.asyncio
    async def test_full_rag_flow(self):
        """Test a full RAG flow with multiple components."""
        request = QueryRequest(
            query="What are the main concepts in robotics?",
            session_id="integration-test-session",
            language="en"
        )

        # Mock the retrieval service to return some chunks
        with patch.object(rag_service.retrieval_service, 'retrieve_chunks_by_request') as mock_retrieve:
            # Mock multiple source chunks
            mock_chunks = [
                SourceChunk(
                    chunk_id="chunk1",
                    chapter_number=1,
                    section_title="Introduction",
                    source_file_path="docs/intro.md",
                    confidence_score=0.8
                ),
                SourceChunk(
                    chunk_id="chunk2",
                    chapter_number=2,
                    section_title="Components",
                    source_file_path="docs/components.md",
                    confidence_score=0.7
                )
            ]

            mock_retrieve.return_value = mock_chunks

            # Mock the retrieval service's get_chunk_content method
            with patch.object(rag_service.retrieval_service, 'get_chunk_content') as mock_get_content:
                mock_get_content.side_effect = [
                    "Robotics is a field that deals with the design, construction, and operation of robots.",
                    "A robot typically consists of sensors, actuators, and a control system."
                ]

                # Call the get_response method
                response_text, source_chunks = await rag_service.get_response(request)

                # Assert that the response was generated
                assert isinstance(response_text, str)
                assert len(response_text) > 0
                assert len(source_chunks) == 2
                assert source_chunks[0].chunk_id == "chunk1"
                assert source_chunks[1].chunk_id == "chunk2"

    @pytest.mark.asyncio
    async def test_multi_skill_task(self):
        """Test executing a multi-skill task."""
        with patch.object(skills_service.retrieval_service, 'retrieve_chunks_by_request') as mock_retrieve:
            mock_retrieve.return_value = []

            result = await skills_service.execute_multi_skill_task(
                "Explain AI concepts",
                skills_sequence=["summarizer", "translator"]
            )

            assert result["status"] == "success"
            assert "results" in result
            assert len(result["results"]) == 2  # summarizer and translator


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])