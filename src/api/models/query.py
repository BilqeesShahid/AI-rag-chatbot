from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """
    Represents a request made to the RAG system.
    """
    query: str  # The original query text
    session_id: Optional[str] = None  # Reference to the session (if applicable)
    selected_text: Optional[str] = None  # Text selected by the user (for /ask/selected-text)
    language: Optional[str] = "en"  # Preferred response language (default: "en")
    context_window: Optional[int] = 5  # Number of previous messages to include in context (default: 5)


class SourceChunk(BaseModel):
    """
    Represents a source chunk used in the response.
    """
    chunk_id: str
    chapter_number: int
    section_title: str
    source_file_path: str
    confidence_score: float
    is_selected_text: Optional[bool] = False  # Whether this chunk was the selected text


class QueryResponse(BaseModel):
    """
    Represents a response from the RAG system.
    """
    response_id: str  # Unique identifier for the response
    response: str  # The generated response text
    source_chunks: List[SourceChunk]  # IDs of chunks used to generate the response
    session_id: str  # Session identifier
    timestamp: datetime  # When the response was generated
    language: str  # Language of the response