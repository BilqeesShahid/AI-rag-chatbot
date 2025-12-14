from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid

# Absolute imports
from src.api.models.query import QueryRequest, QueryResponse, SourceChunk
from src.api.services.rag_service import rag_service

router = APIRouter()



@router.post("/ask", response_model=QueryResponse)
async def ask_endpoint(request: QueryRequest):
    """
    Endpoint to ask questions about the book content.
    """
    try:
        # Validate request
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 1000:  # Max query length from constants
            raise HTTPException(status_code=400, detail="Query is too long")

        # Generate a response ID
        response_id = str(uuid.uuid4())

        # Get response from RAG service
        response_text, source_chunks = await rag_service.get_response(request)

        # Create response object
        response = QueryResponse(
            response_id=response_id,
            response=response_text,
            source_chunks=source_chunks,
            session_id=request.session_id or str(uuid.uuid4()),
            timestamp=datetime.now(),
            language=request.language
        )

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")