from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import re
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """
    Middleware for security validation of requests.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope)
        # Validate the request before passing it to the app
        try:
            await self.validate_request(request)
        except HTTPException as e:
            # Create an error response
            response = JSONResponse(
                status_code=e.status_code,
                content={"error": {"code": e.status_code, "message": e.detail}}
            )
            await response(scope, receive, send)
            return

        # If validation passes, continue with the request
        await self.app(scope, receive, send)

    async def validate_request(self, request: Request):
        """
        Validate the incoming request for security concerns.
        """
        # Check content type for POST requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                # Only allow JSON content for our API
                raise HTTPException(
                    status_code=400,
                    detail="Content-Type must be application/json"
                )

        # Read the request body
        try:
            body = await request.body()
            if body:
                # For JSON requests, validate the content
                if request.headers.get("content-type", "").startswith("application/json"):
                    import json
                    try:
                        json_body = json.loads(body)
                        # Validate all string fields in the JSON
                        self._validate_json_content(json_body)
                    except json.JSONDecodeError:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid JSON in request body"
                        )
        except Exception as e:
            logger.error(f"Error reading request body: {e}")
            raise HTTPException(
                status_code=400,
                detail="Error reading request body"
            )

    def _validate_json_content(self, obj):
        """
        Recursively validate JSON content for potential security issues.
        """
        if isinstance(obj, str):
            self._validate_string_content(obj)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                self._validate_json_content(key)  # Validate keys too
                self._validate_json_content(value)
        elif isinstance(obj, list):
            for item in obj:
                self._validate_json_content(item)

    def _validate_string_content(self, text: str):
        """
        Validate string content for potential security issues.
        """
        # Check for potential SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(\b(OR|AND)\s+.*\s*=\s*['\"][^'\"]*['\"]\s*(\s+OR\s+|\s+AND\s+))",
            r"(;\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC))",
        ]

        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {text[:100]}...")
                raise HTTPException(
                    status_code=400,
                    detail="Request contains potentially unsafe content"
                )

        # Check for potential XSS patterns
        xss_patterns = [
            r"<script",
            r"javascript:",
            r"vbscript:",
            r"on\w+\s*=",
            r"<iframe",
            r"<object",
            r"<embed",
        ]

        for pattern in xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potential XSS detected: {text[:100]}...")
                raise HTTPException(
                    status_code=400,
                    detail="Request contains potentially unsafe content"
                )

        # Check for potential command injection
        cmd_patterns = [
            r"[;&|]",
            r"\$\(.*\)",
            r"`.*`",
            r"exec\(",
        ]

        for pattern in cmd_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potential command injection detected: {text[:100]}...")
                raise HTTPException(
                    status_code=400,
                    detail="Request contains potentially unsafe content"
                )

        # Check for excessively long strings (potential DoS)
        if len(text) > 10000:  # Adjust threshold as needed
            raise HTTPException(
                status_code=413,
                detail="Request contains overly long content"
            )


# Function to add security middleware to FastAPI app
def add_security_middleware(app):
    """
    Add security middleware to the FastAPI application.
    """
    return SecurityMiddleware(app)