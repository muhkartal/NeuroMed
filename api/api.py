from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextvars import ContextVar
import uuid
import os
from typing import Callable, List, Optional

# Import routers
from api.routers import api_router
from api.core.config import settings
from api.core.logging import configure_logging
from api.core.exceptions import BaseAPIException
from api.db.session import SessionLocal, engine
from api.db import models

# Configure logging
logger = logging.getLogger("api")
configure_logging()

# Create request ID context
request_id_contextvar: ContextVar[str] = ContextVar("request_id", default="")

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="MedExplain AI Pro - Advanced personal health assistant API",
    version="1.0.0",
    docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
    openapi_url="/api/openapi.json" if settings.ENVIRONMENT != "production" else None,
)

# CORS middleware configuration
origins = []
if settings.BACKEND_CORS_ORIGINS:
    origins_raw = settings.BACKEND_CORS_ORIGINS.split(",")
    for origin in origins_raw:
        use_origin = origin.strip()
        origins.append(use_origin)
    logger.info(f"Configured CORS origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log requests and add request ID
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    # Generate a request ID
    request_id = str(uuid.uuid4())
    request_id_contextvar.set(request_id)

    # Add request ID to request state
    request.state.request_id = request_id

    # Start timer for request duration
    start_time = time.time()

    # Log the incoming request
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host,
        },
    )

    # Process the request
    try:
        response = await call_next(request)

        # Calculate request duration
        process_time = (time.time() - start_time) * 1000

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log the completed request
        logger.info(
            f"Request completed: {request.method} {request.url.path} - {response.status_code}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(process_time, 2),
            },
        )

        return response
    except Exception as e:
        # Log the exception
        logger.exception(
            f"Request failed: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
            },
        )
        raise

# Exception handlers
@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    """Handler for custom API exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "request_id": getattr(request.state, "request_id", None),
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler for unhandled exceptions"""
    logger.exception(
        f"Unhandled exception in {request.method} {request.url.path}",
        extra={"request_id": getattr(request.state, "request_id", None)},
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "details": str(exc) if settings.DEBUG else None,
            "request_id": getattr(request.state, "request_id", None),
        },
    )

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Include API routers
app.include_
