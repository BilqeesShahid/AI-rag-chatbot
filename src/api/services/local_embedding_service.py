import logging
from typing import List, Optional
import os
import pickle
import numpy as np

# Try to import sklearn, but fallback to simple approach if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer = None
    cosine_similarity = None

from ..models.embedding import DocumentChunk
from ..utils.constants import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """
    Service for generating and storing embeddings locally without external dependencies.
    Uses TF-IDF for keyword-based similarity and stores embeddings in memory/file.
    """

    def __init__(self):
        # Check if sklearn is available
        if SKLEARN_AVAILABLE:
            # Initialize TF-IDF vectorizer for keyword-based similarity
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2),  # Include unigrams and bigrams
                max_features=EMBEDDING_DIMENSION  # Use the configured embedding dimension
            )
            self.use_sklearn = True
        else:
            # Fallback to simple approach
            self.use_sklearn = False
            logger.info("Sklearn not available, using simple keyword matching")

        # Store all document contents with their IDs for TF-IDF processing
        self.chunk_store = {}  # {chunk_id: DocumentChunk}
        self.all_contents = []  # List of all document contents
        self.content_to_id = {}  # {content_index: chunk_id}

        # Try to load any previously stored data
        self.load_embeddings_from_file()

    def save_embeddings_to_file(self):
        """Save embeddings to a local file for persistence."""
        try:
            data = {
                'chunks': self.chunk_store,
                'all_contents': self.all_contents,
                'content_to_id': self.content_to_id
            }
            with open('local_embeddings.pkl', 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save embeddings to file: {e}")

    def load_embeddings_from_file(self):
        """Load embeddings from a local file."""
        try:
            if os.path.exists('local_embeddings.pkl'):
                with open('local_embeddings.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.chunk_store = data.get('chunks', {})
                    self.all_contents = data.get('all_contents', [])
                    self.content_to_id = data.get('content_to_id', {})
        except Exception as e:
            logger.warning(f"Could not load embeddings from file: {e}")

    def store_embedding(self, document_chunk: DocumentChunk) -> str:
        """
        Store a document chunk with its TF-IDF representation locally.
        """
        try:
            # Store the chunk
            self.chunk_store[document_chunk.id] = document_chunk

            # Add content to the list for TF-IDF processing
            content_index = len(self.all_contents)
            self.all_contents.append(document_chunk.content)
            self.content_to_id[content_index] = document_chunk.id

            # Re-fit the vectorizer with all contents to maintain vocabulary consistency (if sklearn is available)
            if self.use_sklearn and len(self.all_contents) > 0:
                try:
                    self.vectorizer.fit(self.all_contents)
                except:
                    # If there's an issue with fitting, continue anyway
                    pass

            # Save to file for persistence
            self.save_embeddings_to_file()

            return document_chunk.id
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            raise

    def search_similar(self, query: str, top_k: int = 5, language: Optional[str] = None) -> List[DocumentChunk]:
        """
        Search for similar document chunks to the query using TF-IDF similarity.
        Falls back to simple keyword matching if sklearn is not available.
        """
        try:
            if not self.all_contents:
                logger.warning("No contents in store, returning empty results")
                return []

            if self.use_sklearn:
                # Use TF-IDF approach
                try:
                    # Fit the vectorizer with all contents first
                    self.vectorizer.fit(self.all_contents)

                    # Transform all contents and the query
                    all_vectors = self.vectorizer.transform(self.all_contents)
                    query_vector = self.vectorizer.transform([query])

                    # Calculate similarities
                    similarities = cosine_similarity(query_vector, all_vectors)[0]

                    # Create results list
                    results = []
                    # Create a list of (index, similarity) pairs
                    indexed_similarities = list(enumerate(similarities))
                    # Sort by similarity (highest first)
                    indexed_similarities.sort(key=lambda x: x[1], reverse=True)

                    # Get top-k results
                    top_results = indexed_similarities[:top_k]

                    for content_index, similarity in top_results:
                        chunk_id = self.content_to_id.get(content_index)
                        if chunk_id and chunk_id in self.chunk_store:
                            chunk = self.chunk_store[chunk_id]
                            results.append(chunk)

                    return results
                except Exception as e:
                    logger.warning(f"TF-IDF approach failed: {e}, falling back to simple matching")
                    return self._simple_keyword_search(query, top_k)
            else:
                # Use simple keyword matching
                return self._simple_keyword_search(query, top_k)
        except Exception as e:
            logger.error(f"Error searching for similar documents: {e}")
            return []

    def _simple_keyword_search(self, query: str, top_k: int) -> List[DocumentChunk]:
        """
        Simple keyword-based search as a fallback when sklearn is not available.
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Calculate similarity scores based on keyword overlap
        scored_chunks = []
        for content_index, content in enumerate(self.all_contents):
            chunk_id = self.content_to_id.get(content_index)
            if chunk_id and chunk_id in self.chunk_store:
                content_lower = content.lower()
                content_words = set(content_lower.split())

                # Calculate overlap score
                overlap = len(query_words.intersection(content_words))
                total_words = len(query_words.union(content_words))

                if total_words > 0:
                    score = overlap / total_words
                else:
                    score = 0

                # Also add bonus for exact phrase matches
                if query_lower in content_lower:
                    score += 0.5  # Boost for exact phrase matches

                scored_chunks.append((self.chunk_store[chunk_id], score))

        # Sort by score in descending order
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return [chunk for chunk, score in scored_chunks[:top_k]]

    def get_embedding(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a document chunk by its ID.
        """
        try:
            return self.chunk_store.get(chunk_id)
        except Exception as e:
            logger.error(f"Error retrieving embedding: {e}")
            return None

    def delete_embedding(self, chunk_id: str):
        """
        Delete an embedding by ID.
        """
        try:
            if chunk_id in self.chunk_store:
                # Find and remove the content from all_contents
                chunk = self.chunk_store[chunk_id]
                content_to_remove = chunk.content
                if content_to_remove in self.all_contents:
                    index_to_remove = self.all_contents.index(content_to_remove)
                    self.all_contents.pop(index_to_remove)
                    # Update content_to_id mapping
                    updated_content_to_id = {}
                    for idx, original_idx in self.content_to_id.items():
                        if original_idx != chunk_id:
                            if idx > index_to_remove:
                                new_idx = idx - 1
                            else:
                                new_idx = idx
                            updated_content_to_id[new_idx] = original_idx
                    self.content_to_id = updated_content_to_id

                del self.chunk_store[chunk_id]

            # Save updated embeddings
            self.save_embeddings_to_file()
        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")


# Singleton instance
local_embedding_service = LocalEmbeddingService()