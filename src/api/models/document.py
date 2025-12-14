from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class DocumentProcessingRequest(BaseModel):
    """
    Request for processing a document for RAG.
    """
    source_path: str  # Path to the source document or directory
    chunk_size: Optional[int] = 400  # Size of text chunks in words
    overlap: Optional[int] = 50  # Overlap between chunks in words


class DocumentChunkRequest(BaseModel):
    """
    Request for chunking a document.
    """
    content: str  # The content to be chunked
    chunk_size: Optional[int] = 400  # Size of text chunks in words
    overlap: Optional[int] = 50  # Overlap between chunks in words
    source_file_path: Optional[str] = None  # Path to the original file
    chapter_number: Optional[int] = None  # Chapter number from the source document
    section_title: Optional[str] = None  # Title of the section


class ProcessedDocumentChunk(BaseModel):
    """
    Represents a processed document chunk ready for embedding.
    """
    content: str  # The chunk content
    metadata: dict  # Metadata including source path, chapter, section, etc.
    chunk_index: int | str  # Position of this chunk within the document (can be int like 2 or string like "2.1")


class DocumentProcessingResult(BaseModel):
    """
    Result of document processing operation.
    """
    processed_chunks: List[ProcessedDocumentChunk]
    total_chunks: int
    source_path: str
    processing_time: float  # Time taken for processing in seconds
    status: str  # Processing status (success, error, etc.)