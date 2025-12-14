from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
import requests
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()

class AuthMiddleware:
    """
    Authentication middleware for validating Supabase user sessions.
    """

    @staticmethod
    async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """
        Verify the Supabase authentication token and return user information.
        """
        token = credentials.credentials

        try:
            # Get Supabase URL and anon key from environment variables
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

            if not supabase_url or not supabase_anon_key:
                logger.warning("Supabase environment variables not set, using mock authentication")
                # In development, if environment variables are not set, return a mock user
                # This allows the app to work in development without Supabase
                user_info = {
                    "id": "mock-user-id",
                    "email": "mock@example.com",
                    "authenticated": True
                }
                return user_info

            # Validate the token with Supabase
            headers = {
                "Authorization": f"Bearer {token}",
                "apikey": supabase_anon_key
            }

            response = requests.get(
                f"{supabase_url}/auth/v1/user",
                headers=headers
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired authentication token"
                )

            user_data = response.json()

            # Return user information
            user_info = {
                "id": user_data.get("id"),
                "email": user_data.get("email"),
                "authenticated": True,
                "user_data": user_data
            }

            return user_info

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Authentication failed"
            )

    @staticmethod
    async def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
        """
        Get current user information if authenticated, otherwise return None.
        This is used for endpoints that work differently based on auth status.
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix

        try:
            # Get Supabase URL and anon key from environment variables
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

            if not supabase_url or not supabase_anon_key:
                logger.warning("Supabase environment variables not set, using mock authentication in optional check")
                # In development, return a mock user if environment variables are not set
                user_info = {
                    "id": "mock-user-id",
                    "email": "mock@example.com",
                    "authenticated": True
                }
                return user_info

            # Validate the token with Supabase
            headers = {
                "Authorization": f"Bearer {token}",
                "apikey": supabase_anon_key
            }

            response = requests.get(
                f"{supabase_url}/auth/v1/user",
                headers=headers
            )

            if response.status_code != 200:
                return None

            user_data = response.json()

            # Return user information
            user_info = {
                "id": user_data.get("id"),
                "email": user_data.get("email"),
                "authenticated": True,
                "user_data": user_data
            }

            return user_info

        except Exception as e:
            logger.error(f"Optional authentication check error: {str(e)}")
            return None