from typing import Dict, List, Optional
from datetime import datetime
import uuid
from ..models.chat import ChatSession, ChatMessage
from ..utils.constants import DEFAULT_CONTEXT_WINDOW


class ChatService:
    """
    Service for managing chat sessions and conversation history.
    """

    def __init__(self):
        # In-memory storage for sessions (in production, use a database)
        self.sessions: Dict[str, ChatSession] = {}
        self.messages: Dict[str, List[ChatMessage]] = {}

    def create_session(self, user_id: Optional[str] = None, title: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.
        """
        session_id = str(uuid.uuid4())
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            title=title or f"Session {session_id[:8]}",
            is_active=True
        )

        self.sessions[session_id] = session
        self.messages[session_id] = []

        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by ID.
        """
        return self.sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str, language: str = "en",
                   context_chunks: Optional[List[str]] = None) -> ChatMessage:
        """
        Add a message to a session.
        """
        if session_id not in self.messages:
            raise ValueError(f"Session {session_id} does not exist")

        message = ChatMessage(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            language=language,
            context_chunks=context_chunks or []
        )

        self.messages[session_id].append(message)

        # Update session's updated_at
        if session_id in self.sessions:
            self.sessions[session_id].updated_at = datetime.now()

        return message

    def get_session_history(self, session_id: str, limit: int = DEFAULT_CONTEXT_WINDOW) -> List[ChatMessage]:
        """
        Get the conversation history for a session.
        """
        if session_id not in self.messages:
            return []

        # Return the last 'limit' messages
        all_messages = self.messages[session_id]
        return all_messages[-limit:]

    def get_recent_context(self, session_id: str, limit: int = DEFAULT_CONTEXT_WINDOW) -> str:
        """
        Get the recent conversation context as a formatted string.
        """
        messages = self.get_session_history(session_id, limit)

        context_parts = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role}: {msg.content}")

        return "\n".join(context_parts)

    def clear_session(self, session_id: str):
        """
        Clear all messages from a session.
        """
        if session_id in self.messages:
            self.messages[session_id] = []

    def end_session(self, session_id: str):
        """
        Mark a session as inactive.
        """
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            self.sessions[session_id].updated_at = datetime.now()


# Singleton instance
chat_service = ChatService()