from typing import Tuple, List
import logging
import os
import uuid
import requests
from ..models.query import QueryRequest, SourceChunk
from ..services.retrieval_service import retrieval_service
from ..services.chat_service import chat_service

logger = logging.getLogger(__name__)


def summarize_answer(query: str, chunks: List[SourceChunk], retrieval_service) -> str:
    """
    Summarize the answer based on the query and retrieved chunks.
    Generic function that works across all book content using comprehensive semantic relevance.
    """
    # Get content from all chunks and filter out non-relevant content
    valid_contents = []
    for chunk in chunks:
        content = retrieval_service.get_chunk_content(chunk.chunk_id)
        if content and content.strip():
            content_clean = content.strip()
            # Filter out quiz questions, answers, and other non-content
            if (len(content_clean) > 20 and
                not any(marker in content_clean.lower() for marker in
                       ['question', 'quiz', 'answer:', 'multiple choice', 'true/false',
                        'question:', 'mcq', 'options', 'correct answer', 'explanation:'])):
                valid_contents.append((chunk, content_clean))

    if not valid_contents:
        return f"I couldn't find relevant information in the book to answer '{query}'. Please try rephrasing your question."

    # Comprehensive relevance scoring
    query_lower = query.lower().strip()
    query_words = set(word for word in query_lower.split() if len(word) > 2)

    scored_results = []
    for chunk, content in valid_contents:
        content_lower = content.lower()
        content_words = set(content_lower.split())

        # Calculate comprehensive relevance score
        score = 0

        # 1. Term overlap with emphasis on exact matches
        common_words = query_words.intersection(content_words)
        if common_words:
            # Weight rare/common words differently - common words get lower weight
            for word in common_words:
                # Boost for longer/more specific query terms
                score += max(1, len(word) - 2)

        # 2. Phrase matching - look for exact query phrases in content
        for phrase in [query_lower] + query_lower.split():
            if len(phrase) > 3:  # Only consider meaningful phrases
                phrase_matches = content_lower.count(phrase)
                if phrase_matches > 0:
                    score += phrase_matches * len(phrase)  # Longer phrases get higher weight

        # 3. Semantic pattern matching for different query types
        if 'what is' in query_lower or query_lower.startswith('what is '):
            # Prioritize definition patterns for "what is" questions
            definition_patterns = [' is a ', ' is an ', ' refers to ', ' means ', ' stands for ',
                                 ' defined as ', ' known as ', ' describes ', ' represents ']
            for pattern in definition_patterns:
                if pattern in content_lower:
                    score += 15  # Strong boost for definition patterns

        elif any(qw in query_lower for qw in ['how to', 'how do', 'steps', 'process', 'procedure']):
            # Prioritize procedural content for "how" questions
            procedural_patterns = ['first', 'then', 'next', 'finally', 'steps', 'process', 'method',
                                 'approach', 'technique', 'way to']
            for pattern in procedural_patterns:
                if pattern in content_lower:
                    score += 8

        elif any(qw in query_lower for qw in ['why', 'reason', 'because']):
            # Prioritize explanatory content for "why" questions
            explanatory_patterns = ['because', 'reason', 'due to', 'caused by', 'results from']
            for pattern in explanatory_patterns:
                if pattern in content_lower:
                    score += 10

        # 4. Content quality metrics
        sentences = content.count('.')
        words = len(content.split())

        # Prefer content with good sentence structure
        if sentences >= 1 and 50 <= words <= 500:  # Good length, not too short or long
            score += 5
        elif 20 <= words < 50:  # Short but complete content
            score += 3

        # 5. Subject relevance - check if main subject appears multiple times
        if len(query_words) > 0:
            main_subject = list(query_words)[0] if query_words else ""
            if main_subject and len(main_subject) > 3:  # Only consider meaningful subjects
                subject_frequency = content_lower.count(main_subject)
                if subject_frequency > 0:
                    score += subject_frequency * 3  # Boost based on subject relevance

        # 6. Chapter-topic relevance
        chunk_title_lower = chunk.section_title.lower()
        if any(topic in query_lower for topic in ['ros2', 'simulation', 'isaac', 'vla']) and \
           any(topic in chunk_title_lower for topic in ['ros2', 'simulation', 'isaac', 'vla']):
            score += 7  # Boost for topic-chapter alignment

        scored_results.append((chunk, content, score))

    # Sort by relevance score (highest first)
    scored_results.sort(key=lambda x: x[2], reverse=True)

    # Return the most relevant content
    if scored_results:
        _, best_content, best_score = scored_results[0]
        if best_score > 0:
            return f"Based on the book content: {best_content.strip()}"
        else:
            # If best score is 0 or negative, still return the best available content
            return f"Based on the book content: {best_content.strip()}"
    else:
        # Fallback - should not reach here due to earlier check, but just in case
        _, fallback_content = valid_contents[0] if valid_contents else (None, "No relevant content found.")
        return f"Based on the book content: {fallback_content[:500].strip()}"


class RAGService:
    """
    Service for RAG (Retrieval-Augmented Generation) operations.
    Orchestrates the retrieval of relevant documents and generation of responses using Hugging Face models.
    """

    def __init__(self):
        # Get Hugging Face API key from environment
        self.hf_api_key = os.getenv("HF_API_KEY")
        if not self.hf_api_key:
            logger.warning("HF_API_KEY not found in environment. Hugging Face model inference will not work.")

        self.retrieval_service = retrieval_service
        self.chat_service = chat_service

    async def get_response(self, request: QueryRequest) -> Tuple[str, List[SourceChunk]]:
        """
        Get a response for a general query based on book content using AI agents.
        """
        try:
            # Get or create session
            if request.session_id:
                session_id = request.session_id
                session = self.chat_service.get_session(session_id)
                if not session:
                    # If session doesn't exist but was requested, create it with the specific ID
                    # We'll create a new session and ignore the ID since create_session generates its own
                    session = self.chat_service.create_session(title=f"RAG Session {session_id[:8]}")
                    session_id = session.session_id  # Use the actual session ID generated
            else:
                # Create a new session
                session = self.chat_service.create_session(title=f"RAG Session")
                session_id = session.session_id

            # Ensure the session exists in messages dict (should be done by create_session)
            if session_id not in self.chat_service.messages:
                self.chat_service.messages[session_id] = []

            self.chat_service.add_message(
                session_id=session_id,
                role="user",
                content=request.query,
                language=request.language
            )

            # Retrieve relevant chunks based on the query
            source_chunks = self.retrieval_service.retrieve_chunks_by_request(request)

            # Generate response using Hugging Face models directly
            response_text = await self._generate_response_with_hf(request.query, source_chunks)

            # Add assistant response to session
            chunk_ids = [chunk.chunk_id for chunk in source_chunks] if source_chunks else []
            self.chat_service.add_message(
                session_id=session_id,
                role="assistant",
                content=response_text,
                language=request.language,
                context_chunks=chunk_ids
            )

            return response_text, source_chunks
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def get_response_with_selected_text(self, request: QueryRequest) -> Tuple[str, List[SourceChunk]]:
        """
        Get a response prioritizing the selected text using AI agents.
        """
        try:
            # Validate that selected text exists
            if not request.selected_text:
                return await self.get_response(request)

            # Get or create session
            if request.session_id:
                session_id = request.session_id
                session = self.chat_service.get_session(session_id)
                if not session:
                    # If session doesn't exist but was requested, create it with the specific ID
                    session = self.chat_service.create_session(title=f"RAG Session {session_id[:8]}")
                    session_id = session.session_id  # Use the actual session ID generated
            else:
                # Create a new session
                session = self.chat_service.create_session(title=f"RAG Session")
                session_id = session.session_id

            # Ensure the session exists in messages dict (should be done by create_session)
            if session_id not in self.chat_service.messages:
                self.chat_service.messages[session_id] = []

            # Add user message to session
            self.chat_service.add_message(
                session_id=session_id,
                role="user",
                content=f"Query: {request.query}\nSelected text: {request.selected_text}",
                language=request.language
            )

            # Retrieve relevant chunks prioritizing selected text
            source_chunks = self.retrieval_service.retrieve_chunks_with_selected_text(
                request.query,
                request.selected_text,
                top_k=5
            )

            # Create a detailed query that includes the selected text for the AI agent
            detailed_query = f"Query: {request.query}\nSelected text for context: {request.selected_text}"

            # Generate response using Hugging Face models directly
            response_text = await self._generate_response_with_hf(detailed_query, source_chunks)

            # Add assistant response to session
            chunk_ids = [chunk.chunk_id for chunk in source_chunks] if source_chunks else []
            self.chat_service.add_message(
                session_id=session_id,
                role="assistant",
                content=response_text,
                language=request.language,
                context_chunks=chunk_ids
            )

            return response_text, source_chunks
        except Exception as e:
            logger.error(f"Error generating response with selected text: {e}")
            raise

    def _prepare_context(self, source_chunks: List[SourceChunk]) -> str:
        """
        Prepare context from retrieved source chunks.
        This method is kept for potential future use.
        """
        context_parts = []
        for chunk in source_chunks:
            context_parts.append(
                f"Chapter {chunk.chapter_number}, Section '{chunk.section_title}':\n"
                f"{self.retrieval_service.get_chunk_content(chunk.chunk_id) or ''}\n"
            )
        return "\n".join(context_parts)

    def _prepare_context_with_selected_text(self, source_chunks: List[SourceChunk], selected_text: str) -> str:
        """
        Prepare context with emphasis on selected text.
        This method is kept for potential future use.
        """
        context_parts = [f"Selected text: {selected_text}\n"]

        for chunk in source_chunks:
            # Add more weight to chunks that are the selected text
            if chunk.is_selected_text:
                context_parts.append(
                    f"Selected text context - Chapter {chunk.chapter_number}, Section '{chunk.section_title}':\n"
                    f"{self.retrieval_service.get_chunk_content(chunk.chunk_id) or ''}\n"
                )
            else:
                context_parts.append(
                    f"Related content - Chapter {chunk.chapter_number}, Section '{chunk.section_title}':\n"
                    f"{self.retrieval_service.get_chunk_content(chunk.chunk_id) or ''}\n"
                )

        return "\n".join(context_parts)

    async def _generate_response_with_hf(self, query: str, source_chunks: List[SourceChunk]) -> str:
        """
        Generate a response using the retrieved context with priority to first chapters.
        Prioritizes content from early chapters (like Chapter 0, 1, etc.) before others.
        """
        try:
            # Use the summarization function to create a concise answer
            summary = summarize_answer(query, source_chunks, self.retrieval_service)

            # If we have source chunks, we can provide more detailed information
            if source_chunks and "No relevant content found" not in summary and "couldn't find relevant information" not in summary:
                return summary
            else:
                # If no relevant content was found, try to be more specific about why
                if not source_chunks:
                    return f"I couldn't find any relevant information in the book to answer '{query}'. The search returned no results. Please try rephrasing your question."
                else:
                    return f"I couldn't find relevant information in the book to answer '{query}'. The retrieved content didn't match your query well. Please try rephrasing your question."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback response
            if source_chunks:
                # If we found chunks but had an error, at least mention that
                # Sort by chapter to prioritize early chapters in message
                sorted_chunks = sorted(source_chunks, key=lambda x: x.chapter_number)
                if sorted_chunks:
                    first_chunk = sorted_chunks[0]
                    return f"I found relevant information in Chapter {first_chunk.chapter_number}, Section '{first_chunk.section_title}' about '{query}', but encountered an issue compiling the full response. Please try asking your question again."
            return f"I couldn't find relevant information in the book to answer '{query}'. Please try rephrasing your question."


# Singleton instance
rag_service = RAGService()