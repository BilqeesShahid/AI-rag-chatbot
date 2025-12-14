import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np

from ..models.embedding import DocumentChunk
from ..utils.constants import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION
)

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """
    Service for generating and storing embeddings using local sentence transformers.
    This avoids quota limits from external APIs.
    """

    def __init__(self):
        # Initialize local sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model

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
        except:
            # Create collection if it doesn't exist
            self.qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 produces 384-dimensional vectors
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection {QDRANT_COLLECTION_NAME}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text using local sentence transformer.
        """
        try:
            # Generate embedding using local model
            embedding = self.model.encode([text])[0].tolist()
            return embedding
        except Exception as e:
            logger.error(f"Error generating local embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        """
        try:
            embeddings = self.model.encode(texts).tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    def store_embedding(self, document_chunk: DocumentChunk) -> str:
        """
        Store a document chunk with its embedding in Qdrant.
        """
        try:
            # Generate embedding using local model
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
            # Extract content from all chunks
            texts = [chunk.content for chunk in document_chunks]

            # Generate embeddings in batch for efficiency
            embedding_vectors = self.generate_embeddings_batch(texts)

            point_structs = []
            for i, chunk in enumerate(document_chunks):
                point = PointStruct(
                    id=chunk.id,
                    vector=embedding_vectors[i],
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

    def search_similar(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Search for similar document chunks to the query.
        """
        try:
            # Generate embedding for the query using local model
            query_embedding = self.generate_embedding(query)

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k
            )

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
                    embedding_vector=hit.vector,
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
local_embedding_service = LocalEmbeddingService()