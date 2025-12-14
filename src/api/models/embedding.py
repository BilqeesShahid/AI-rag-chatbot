from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class DocumentChunk(BaseModel):
    """
    Represents a chunk of content from the book that has been processed for RAG.
    """
    id: str  # Unique identifier for the chunk
    content: str  # The actual text content of the chunk
    chapter_number: int  # Chapter number from the source document
    section_title: str  # Title of the section this chunk belongs to
    source_file_path: str  # Path to the original Markdown file
    embedding_vector: Optional[List[float]] = None  # The embedding vector (optional for retrieval)
    chunk_index: int | str  # Position of this chunk within the document (can be int like 2 or string like "2.1")
    metadata: Optional[dict] = None  # Additional metadata (word count, headings, etc.)


class DocumentChunkWithDistance(DocumentChunk):
    """
    Document chunk with similarity distance for retrieval results.
    """
    distance: float