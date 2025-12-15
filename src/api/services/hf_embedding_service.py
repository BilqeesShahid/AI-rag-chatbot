import logging
import os
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
import numpy as np

from ..models.embedding import DocumentChunk
from ..utils.constants import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION
)

logger = logging.getLogger(__name__)


class HFEmbeddingService:
    """
    Service for generating and storing embeddings using Hugging Face models.
    Falls back to simple keyword-based approach if sentence transformers fail.
    """

    def __init__(self):
        # Try to initialize Hugging Face sentence transformer model
        self.model = None
        self.use_simple_embeddings = False

        try:
            # Check if sentence_transformers is available
            import sentence_transformers
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = 384  # Size of all-MiniLM-L6-v2 embeddings
            logger.info("Initialized Hugging Face sentence transformer model")
        except ImportError:
            logger.info("sentence_transformers not available, using simple embeddings")
            self.use_simple_embeddings = True
            self.embedding_dimension = EMBEDDING_DIMENSION
        except OSError:
            # This handles the DLL loading errors that can occur with torch/transformers
            logger.info("sentence_transformers not available (DLL error), using simple embeddings")
            self.use_simple_embeddings = True
            self.embedding_dimension = EMBEDDING_DIMENSION
        except Exception as e:
            logger.warning(f"Failed to initialize Hugging Face sentence transformer model: {e}")
            logger.info("Falling back to simple keyword-based embeddings")
            self.use_simple_embeddings = True
            self.embedding_dimension = EMBEDDING_DIMENSION

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False  # Using HTTP for better compatibility
        )

        # Ensure collection exists
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Ensure the Qdrant collection exists with the correct configuration.
        """
        try:
            # Try to get collection info to see if it exists
            self.qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
            logger.info(f"Collection {QDRANT_COLLECTION_NAME} already exists")
        except Exception as e:
            logger.warning(f"Could not access collection {QDRANT_COLLECTION_NAME}: {e}")
            # Attempt to create collection if it doesn't exist
            try:
                self.qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection {QDRANT_COLLECTION_NAME} with embedding size {self.embedding_dimension}")
            except Exception as create_error:
                logger.error(f"Could not create collection: {create_error}")
                raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text using Hugging Face sentence transformer or simple approach.
        """
        try:
            if self.use_simple_embeddings:
                # Use a simple keyword-based approach
                return self._generate_simple_embedding(text)
            else:
                # Use Hugging Face sentence transformer
                embedding = self.model.encode([text])[0]
                # Convert to list if it's a numpy array
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.embedding_dimension

    def _generate_simple_embedding(self, text: str) -> List[float]:
        """
        Generate a simple embedding using keyword frequency approach.
        """
        import hashlib
        # Create a deterministic vector based on the text content
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        # Convert hash to a vector of the required size
        vector = []
        for i in range(0, len(text_hash), 2):
            if len(vector) >= self.embedding_dimension:
                break
            hex_pair = text_hash[i:i+2]
            val = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            vector.append(val)
        # Pad or truncate to required size
        while len(vector) < self.embedding_dimension:
            vector.append(0.0)
        return vector[:self.embedding_dimension]

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using Hugging Face sentence transformer or simple approach.
        """
        try:
            if self.use_simple_embeddings:
                # Use simple approach for batch
                embeddings = []
                for text in texts:
                    embedding = self._generate_simple_embedding(text)
                    embeddings.append(embedding)
                return embeddings
            else:
                # Use Hugging Face sentence transformer
                embeddings = self.model.encode(texts)
                # Convert to list of lists if it's a numpy array
                if hasattr(embeddings, 'tolist'):
                    embeddings = embeddings.tolist()
                return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dimension for _ in texts]

    def store_embedding(self, document_chunk: DocumentChunk) -> str:
        """
        Store a document chunk with its embedding in Qdrant.
        """
        try:
            # Generate embedding
            embedding_vector = self.generate_embedding(document_chunk.content)

            # Prepare the point for Qdrant
            point = PointStruct(
                id=document_chunk.id,
                vector=embedding_vector,
                payload={
                    "content": document_chunk.content,
                    "chapter_number": document_chunk.chapter_number,
                    "section_title": document_chunk.section_title,
                    "source_file_path": document_chunk.source_file_path,
                    "chunk_index": document_chunk.chunk_index,
                    "metadata": document_chunk.metadata or {}
                }
            )

            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[point]
            )

            return document_chunk.id
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            raise

    def store_embeddings_batch(self, document_chunks: List[DocumentChunk]) -> List[str]:
        """
        Store multiple document chunks with their embeddings in Qdrant.
        """
        try:
            point_structs = []
            for chunk in document_chunks:
                # Generate embedding
                embedding_vector = self.generate_embedding(chunk.content)

                # Prepare the point for Qdrant
                point = PointStruct(
                    id=chunk.id,
                    vector=embedding_vector,
                    payload={
                        "content": chunk.content,
                        "chapter_number": chunk.chapter_number,
                        "section_title": chunk.section_title,
                        "source_file_path": chunk.source_file_path,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata or {}
                    }
                )
                point_structs.append(point)

            # Store all points in Qdrant
            self.qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=point_structs
            )

            return [chunk.id for chunk in document_chunks]
        except Exception as e:
            logger.error(f"Error storing embeddings batch: {e}")
            raise

    def search_similar(self, query: str, top_k: int = 5, language: Optional[str] = None) -> List[DocumentChunk]:
        """
        Search for similar document chunks to the query using Hugging Face embeddings.
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)

            # Prepare search filter for language if specified
            search_filter = None
            if language:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.language",
                            match=models.MatchValue(value=language)
                        )
                    ]
                )

            # Search in Qdrant
            search_response = self.qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=query_embedding,
                limit=top_k,
                query_filter=search_filter
            )

            # Extract results from the response object
            search_results = search_response.points

            # Convert results to DocumentChunk objects
            results = []
            for hit in search_results:
                payload = hit.payload
                chunk = DocumentChunk(
                    id=hit.id,
                    content=payload.get("content", ""),
                    chapter_number=payload.get("chapter_number", 0),
                    section_title=payload.get("section_title", ""),
                    source_file_path=payload.get("source_file_path", ""),
                    embedding_vector=hit.vector if hasattr(hit, 'vector') and hit.vector is not None else query_embedding,
                    chunk_index=payload.get("chunk_index", 0),
                    metadata=payload.get("metadata", {})
                )
                results.append(chunk)

            return results
        except Exception as e:
            logger.error(f"Error searching for similar documents: {e}")
            raise

    def delete_embedding(self, chunk_id: str):
        """
        Delete an embedding from Qdrant by ID.
        """
        try:
            self.qdrant_client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=models.PointIdsList(
                    points=[chunk_id]
                )
            )
        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")
            raise

    def get_embedding(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a document chunk by its ID.
        """
        try:
            records = self.qdrant_client.retrieve(
                collection_name=QDRANT_COLLECTION_NAME,
                ids=[chunk_id]
            )

            if records:
                record = records[0]
                payload = record.payload
                return DocumentChunk(
                    id=record.id,
                    content=payload.get("content", ""),
                    chapter_number=payload.get("chapter_number", 0),
                    section_title=payload.get("section_title", ""),
                    source_file_path=payload.get("source_file_path", ""),
                    embedding_vector=record.vector,
                    chunk_index=payload.get("chunk_index", 0),
                    metadata=payload.get("metadata", {})
                )
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding: {e}")
            raise


# Singleton instance
hf_embedding_service = HFEmbeddingService()