from typing import Tuple, List
import logging
import os
import uuid
import requests
from ..models.query import QueryRequest, SourceChunk
from ..services.retrieval_service import retrieval_service
from ..services.chat_service import chat_service
from ..services.hf_generation_service import hf_generation_service

logger = logging.getLogger(__name__)


def summarize_answer(query: str, chunks: List[SourceChunk], retrieval_service) -> str:
    """
    Summarize the answer based on the query and retrieved chunks.
    Generic function that works across all book content using comprehensive semantic relevance.
    Enhanced to better handle definition-type queries like "what is X".
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

    # Check if it's a definition query to apply special logic
    is_definition_query = 'what is' in query_lower or 'define' in query_lower or 'meaning of' in query_lower

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
        if is_definition_query:
            # Prioritize definition patterns for definition questions
            definition_patterns = [' is a ', ' is an ', ' refers to ', ' means ', ' stands for ',
                                 ' defined as ', ' known as ', ' describes ', ' represents ', ' called ', ' termed ']
            for pattern in definition_patterns:
                if pattern in content_lower:
                    score += 20  # Strong boost for definition patterns

            # Additional boost for short, clear definitions
            if 10 <= len(content.split()) <= 100 and (' is ' in content_lower or ' are ' in content_lower):
                score += 15

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
            # Extract main subject for "what is X" queries
            main_subject = ""
            if 'what is ' in query_lower:
                main_subject = query_lower.split('what is ')[1].strip().split()[0] if len(query_lower.split('what is ')) > 1 else ""
            elif query_words:
                main_subject = list(query_words)[0]

            if main_subject and len(main_subject) > 2:  # Only consider meaningful subjects
                subject_frequency = content_lower.count(main_subject.lower())
                if subject_frequency > 0:
                    score += subject_frequency * 5  # Higher boost for subject relevance in definition queries

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
        Generate a response using Hugging Face models with the retrieved context.
        Enhanced to better handle definition queries like "what is X".
        """
        try:
            # Check if it's a definition query to apply special logic
            query_lower = query.lower().strip()
            is_definition_query = any(phrase in query_lower for phrase in ['what is', 'define', 'meaning of', 'what does ... mean'])

            if source_chunks:
                # If it's a definition query, prioritize chunks that might contain definitions
                if is_definition_query:
                    # Re-rank chunks based on definition patterns before passing to generation
                    ranked_chunks = self._rank_chunks_for_definitions(query, source_chunks)

                    # Check if the top-ranked chunk contains a clear definition
                    if ranked_chunks:
                        top_chunk = ranked_chunks[0]
                        top_chunk_content = self.retrieval_service.get_chunk_content(top_chunk.chunk_id)

                        # If the top chunk contains a clear definition pattern, return it directly
                        if top_chunk_content and self._contains_clear_definition(query, top_chunk_content):
                            return f"Based on the book content: {top_chunk_content[:500]}..."

                    response = hf_generation_service.generate_response(query, ranked_chunks, self.retrieval_service)
                else:
                    response = hf_generation_service.generate_response(query, source_chunks, self.retrieval_service)
                return response
            else:
                return f"I couldn't find any relevant information in the book to answer '{query}'. The search returned no results. Please try rephrasing your question."
        except Exception as e:
            logger.error(f"Error generating response with Hugging Face: {e}")
            # Fallback to local summarization
            try:
                summary = summarize_answer(query, source_chunks, self.retrieval_service)
                return summary
            except:
                # Final fallback
                if source_chunks:
                    # Just return the content of the first chunk
                    first_chunk_content = self.retrieval_service.get_chunk_content(source_chunks[0].chunk_id)
                    if first_chunk_content:
                        return f"Based on the book content: {first_chunk_content[:500]}..."
                return f"I couldn't find relevant information in the book to answer '{query}'. Please try rephrasing your question."

    def _contains_clear_definition(self, query: str, content: str) -> bool:
        """
        Check if the content contains a clear definition for the query subject.
        """
        query_lower = query.lower().strip()

        # Extract the subject from "what is X" queries
        if 'what is ' in query_lower:
            subject = query_lower.split('what is ')[1].strip().split()[0] if len(query_lower.split('what is ')) > 1 else ""
            if subject and len(subject) > 2:  # Only consider meaningful subjects
                content_lower = content.lower()

                # Look for definition patterns with the subject
                definition_patterns = [
                    f"{subject} (is|was|are) ",
                    f"{subject} (is|was|are) a ",
                    f"{subject} (is|was|are) an ",
                    f"{subject} (stands|stands) for ",
                    f"{subject} (means|mean) ",
                    f"{subject} (refers|refer) to "
                ]

                import re
                for pattern in definition_patterns:
                    if re.search(pattern.replace('(', '(?:'), content_lower):
                        return True

                # Look for the subject followed by "is" pattern
                subject_is_pattern = rf"\b{re.escape(subject.lower())}\b.*\bis\b"
                if re.search(subject_is_pattern, content_lower):
                    # Check if what follows "is" is a reasonable definition
                    match = re.search(rf"\b{re.escape(subject.lower())}\b(.*)", content_lower)
                    if match:
                        after_subject = match.group(1)
                        # If it's followed by "is" and some descriptive text
                        if ' is ' in after_subject and len(after_subject.split()) < 50:  # Reasonable length for a definition
                            return True

        return False

    def _rank_chunks_for_definitions(self, query: str, source_chunks: List[SourceChunk]) -> List[SourceChunk]:
        """
        Re-rank chunks to prioritize those that might contain definitions for definition queries.
        Enhanced to look for specific definition patterns and prioritize exact definitions.
        """
        # Get content for each chunk to score it
        scored_chunks = []
        query_lower = query.lower().strip()

        for chunk in source_chunks:
            content = self.retrieval_service.get_chunk_content(chunk.chunk_id)
            if content:
                content_lower = content.lower()
                score = 0

                # Look for the exact definition pattern of ROS2 or similar
                # Check if this looks like a definition of the subject being asked about
                if 'what is ' in query_lower:
                    subject = query_lower.split('what is ')[1].strip().split()[0] if len(query_lower.split('what is ')) > 1 else ""
                    if subject:
                        # Look for exact definition patterns with the subject
                        # e.g., "ROS2 is a flexible framework" or "X is Y" patterns
                        subject_pattern = f"{subject.lower()} is "
                        if subject_pattern in content_lower:
                            score += 50  # High score for exact subject definition pattern

                        # Look for the specific pattern from the docs: "X (Full Name) is a ..."
                        import re
                        # Check for pattern like "ROS2 (Robot Operating System 2) is a ..."
                        if re.search(rf"{re.escape(subject.lower())} \([^)]+\) is (a|an) ", content_lower):
                            score += 60  # Very high score for formal definition pattern

                # Boost score if content contains definition patterns
                if any(pattern in content_lower for pattern in [' is a ', ' is an ', ' refers to ', ' means ', ' stands for ', ' defined as ', ' known as ', ' describes ', ' represents ', ' called ', ' termed ']):
                    score += 20
                elif ' is ' in content_lower and 10 <= len(content.split()) <= 100:  # Short, likely definition
                    score += 15
                elif 'definition' in content_lower:
                    score += 18

                # Boost if main subject appears in content
                if 'what is ' in query_lower:
                    subject = query_lower.split('what is ')[1].strip().split()[0] if len(query_lower.split('what is ')) > 1 else ""
                    if subject and len(subject) > 2:
                        if subject.lower() in content_lower:
                            score += 10

                # Penalize procedural content for definition queries
                if any(word in content_lower for word in ['first', 'then', 'next', 'finally', 'step', 'process', 'how to', 'method', 'exercise', 'goal:', 'steps:', 'instructions:']):
                    score -= 15

                # Additional boost for content that appears to be introductory or foundational
                if any(word in content_lower for word in ['introduction', 'what is', 'overview', 'fundamentals', 'basic']):
                    score += 12

                scored_chunks.append((chunk, score))
            else:
                scored_chunks.append((chunk, 0))  # Default score for chunks without content

        # Sort by score in descending order
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return just the chunks in the new order
        return [chunk for chunk, score in scored_chunks]


# Singleton instance
rag_service = RAGService()