from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

from . import config
from .database import create_db_and_tables
from .routers import auth, health, rag, translate
from core.logging_utils import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage startup and shutdown events for the application.
    """
    # Startup
    logger.info("FastAPI application starting up...")
    create_db_and_tables()
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    logger.info("FastAPI application startup complete. Database tables and cache initialized.")
    yield
    # Shutdown
    logger.info("FastAPI application shutdown complete.")


app = FastAPI(
    title="Modular RAG API with Ollama",
    description="API endpoint for interacting with a RAG pipeline using local LLMs via Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)

# Security Warning for default key
if config.SECRET_KEY == "a_very_secret_key_for_dev_DO_NOT_USE_IN_PROD":
    logger.warning(
        "CRITICAL SECURITY WARNING: The API is running with the default insecure SECRET_KEY. "
        "This key MUST be changed for production environments."
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(auth.router)
app.include_router(rag.router)
app.include_router(health.router)
app.include_router(translate.router)


@app.get("/", tags=["Root"])
async def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"message": "Welcome to the Indic Law Assistant API"}


# --- Run the API Server ---
if __name__ == "__main__":
    # This block is for direct execution, e.g., for debugging.
    # Production deployments should use a proper ASGI server like Uvicorn directly.
    print("Starting FastAPI server for RAG API...")
    print("Ensure Ollama is running with required models:")
    print(f"  Generator: {config.GENERATOR_MODEL}")
    print(f"  Rewriter/Reranker: {config.REWRITER_RERANKER_MODEL}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
