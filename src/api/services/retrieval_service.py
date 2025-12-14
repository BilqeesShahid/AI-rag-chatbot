import logging
from typing import List, Optional
from ..models.embedding import DocumentChunk
from ..models.query import SourceChunk
from ..models.query import QueryRequest
from ..services.tfidf_embedding_service import tfidf_embedding_service
from ..utils.helpers import calculate_similarity_score

logger = logging.getLogger(__name__)


def reformulate_query(query: str) -> str:
    """
    Reformulate the user query to improve search results.
    This function can expand the query with synonyms or related terms.
    """
    # For now, we'll return the query as is, but this can be enhanced with
    # NLP techniques to expand the query with synonyms or related terms
    # In a more advanced implementation, we might use techniques like:
    # - Query expansion with synonyms
    # - Named entity recognition
    # - Query understanding and reformulation

    # For now, just return the original query
    # In the future, we could enhance this with more sophisticated techniques
    return query


class RetrievalService:
    """
    Service for retrieving relevant document chunks based on queries.
    Handles both general queries and selected text prioritization.
    """

    def __init__(self):
        self.embedding_service = tfidf_embedding_service

    def retrieve_chunks(self, query: str, top_k: int = 5, language: Optional[str] = None) -> List[SourceChunk]:
        """
        Retrieve the most relevant chunks for a given query.
        """
        try:
            # Reformulate the query before searching
            reformulated_query = reformulate_query(query)

            # Search for similar chunks in the vector database
            similar_chunks = self.embedding_service.search_similar(reformulated_query, top_k, language=language)

            # Convert to SourceChunk objects with confidence scores
            source_chunks = []
            for chunk in similar_chunks:
                # Calculate confidence score based on similarity
                # For now, we'll use a simple approach; in a real system,
                # this would be based on the distance returned by the vector search
                confidence_score = 0.8  # Placeholder - would be calculated from distance in real implementation

                source_chunk = SourceChunk(
                    chunk_id=chunk.id,
                    chapter_number=chunk.chapter_number,
                    section_title=chunk.section_title,
                    source_file_path=chunk.source_file_path,
                    confidence_score=confidence_score
                )
                source_chunks.append(source_chunk)

            return source_chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise

    def retrieve_chunks_with_selected_text(self, query: str, selected_text: str, top_k: int = 5) -> List[SourceChunk]:
        """
        Retrieve chunks prioritizing the selected text.
        """
        try:
            # Reformulate the query before searching
            reformulated_query = reformulate_query(query)

            # First, search for chunks relevant to the selected text
            selected_text_chunks = self.embedding_service.search_similar(selected_text, top_k // 2)

            # Then, search for chunks relevant to the query
            query_chunks = self.embedding_service.search_similar(reformulated_query, top_k // 2)

            # Combine and deduplicate chunks
            all_chunks = selected_text_chunks + query_chunks
            seen_ids = set()
            unique_chunks = []

            for chunk in all_chunks:
                if chunk.id not in seen_ids:
                    unique_chunks.append(chunk)
                    seen_ids.add(chunk.id)

            # Convert to SourceChunk objects
            source_chunks = []
            for i, chunk in enumerate(unique_chunks[:top_k]):
                # Higher confidence for chunks from selected text
                is_selected_text = i < len(selected_text_chunks)
                confidence_score = 0.9 if is_selected_text else 0.7

                source_chunk = SourceChunk(
                    chunk_id=chunk.id,
                    chapter_number=chunk.chapter_number,
                    section_title=chunk.section_title,
                    source_file_path=chunk.source_file_path,
                    confidence_score=confidence_score,
                    is_selected_text=is_selected_text
                )
                source_chunks.append(source_chunk)

            return source_chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks with selected text: {e}")
            raise

    def retrieve_chunks_by_request(self, request: QueryRequest) -> List[SourceChunk]:
        """
        Retrieve chunks based on a query request, handling both regular and selected text queries.
        Enhanced to prioritize definition-like content for "what is" queries.
        """
        if request.selected_text:
            # Use selected text prioritization
            return self.retrieve_chunks_with_selected_text(
                request.query,
                request.selected_text,
                top_k=5
            )
        else:
            # Use regular retrieval
            chunks = self.retrieve_chunks(
                request.query,
                top_k=10  # Get more chunks to have more options for definition matching
            )

            # For "what is" type queries, re-rank chunks based on definition content
            if self._is_definition_query(request.query):
                chunks = self._prioritize_definition_chunks(chunks, request.query)

            return chunks[:5]  # Return top 5 after re-ranking

    def _is_definition_query(self, query: str) -> bool:
        """
        Check if the query is asking for a definition (e.g., "what is X", "define X", etc.)
        """
        query_lower = query.lower().strip()
        definition_indicators = [
            'what is ',
            'define ',
            'definition of ',
            'what does ',
            'meaning of ',
            'explain '
        ]
        return any(indicator in query_lower for indicator in definition_indicators)

    def _prioritize_definition_chunks(self, chunks: List[SourceChunk], query: str) -> List[SourceChunk]:
        """
        Re-rank chunks to prioritize those containing definition-like content.
        Also filter out non-relevant content like quiz questions.
        """
        query_lower = query.lower().strip()
        # Extract the subject being defined (e.g., "what is ROS2" -> "ROS2")
        subject = ""
        if query_lower.startswith('what is '):
            subject = query_lower[8:].strip()
        elif query_lower.startswith('define '):
            subject = query_lower[7:].strip()
        elif 'definition of ' in query_lower:
            subject = query_lower.split('definition of ')[1].strip()
        elif 'what does ' in query_lower and ' mean' in query_lower:
            subject = query_lower.split('what does ')[1].split(' mean')[0].strip()

        # Score chunks based on definition content and filter out non-relevant content
        scored_chunks = []
        for chunk in chunks:
            content = self.get_chunk_content(chunk.chunk_id)
            if content:
                content_clean = content.strip()

                # Skip if content looks like a quiz question or contains obvious non-content
                if (len(content_clean) > 20 and  # Ensure it's substantial content
                    not content_clean.lower().startswith('question') and
                    not content_clean.lower().startswith('quiz') and
                    not content_clean.lower().startswith('answer') and
                    'multiple choice' not in content_clean.lower() and
                    'true/false' not in content_clean.lower() and
                    'question ' not in content_clean.lower() and  # catch "Question 4:" etc.
                    'answer:' not in content_clean.lower()):

                    score = 0
                    content_lower = content_clean.lower()

                    # Boost score if content contains definition patterns
                    if ' is a ' in content_lower or ' is an ' in content_lower:
                        score += 10  # Strong definition pattern
                    elif ' is ' in content_lower and len(content_lower.split()) < 100:  # Short definition
                        score += 5
                    elif 'definition' in content_lower:
                        score += 8
                    elif 'means' in content_lower or 'stands for' in content_lower:
                        score += 7

                    # Boost if the subject being defined appears in the content
                    if subject and subject in content_lower:
                        score += 3

                    # Boost if section title matches the query subject
                    if subject and subject in chunk.section_title.lower():
                        score += 2

                    scored_chunks.append((chunk, score))

        # Sort by score in descending order
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return chunks in new order (only the chunk objects, not scores)
        return [chunk for chunk, score in scored_chunks]

    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """
        Retrieve the content of a specific chunk by its ID.
        """
        try:
            chunk = self.embedding_service.get_embedding(chunk_id)
            return chunk.content if chunk else None
        except Exception as e:
            logger.error(f"Error retrieving chunk content: {e}")
            return None


# Singleton instance
retrieval_service = RetrievalService()