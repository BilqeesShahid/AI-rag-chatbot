from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class ChatSession(BaseModel):
    """
    Represents a user's conversation session with the chatbot.
    """
    session_id: str  # Unique identifier for the session
    user_id: Optional[str] = None  # Identifier for the user (optional for anonymous)
    created_at: datetime  # Timestamp when the session was created
    updated_at: datetime  # Timestamp of last activity
    title: str  # Generated title for the conversation
    is_active: bool  # Whether the session is currently active


class ChatMessage(BaseModel):
    """
    Represents a single message in a conversation.
    """
    message_id: str  # Unique identifier for the message
    session_id: str  # Reference to the parent session
    role: str  # The role of the message sender (user, assistant, system)
    content: str  # The text content of the message
    timestamp: datetime  # When the message was created
    language: str  # Language of the message (e.g., "en", "ur")
    context_chunks: Optional[List[str]] = []  # IDs of document chunks used to generate this response