import os

from fastapi import Request, status, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, RetryError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# 1. Strict CORS Lockdown
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],  # Default cap across all routes
    headers_enabled=True,
)

def setup_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # 2. Global Exception Handlers
    @app.exception_handler(ResourceExhausted)
    async def google_rate_limit_handler(req: Request, exc: ResourceExhausted):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "RateLimitExceeded",
                "message": "Google Gemini API rate limit reached. Please wait a moment and try again.",
                "details": str(exc),
            }
        )

    @app.exception_handler(RetryError)
    @app.exception_handler(TimeoutError)
    async def timeout_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": "UpstreamTimeout",
                "message": "The LLM request timed out. Please retry your generation request.",
                "details": str(exc),
            },
        )

    @app.exception_handler(GoogleAPIError)
    async def general_google_api_handler(request: Request, exc: GoogleAPIError):
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": "GoogleAPIError",
                "message": "An error occurred while communicating with the Google GenAI API.",
                "details": str(exc),
            },
        )