import os
from dotenv import load_dotenv

load_dotenv()

# --- General Settings ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Ollama Settings ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gemma3:1b")
REWRITER_RERANKER_MODEL = os.getenv("REWRITER_RERANKER_MODEL", "gemma3:1b")

# Embedding model (used by SentenceTransformer in models.py)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# --- RAG Pipeline Settings ---
INITIAL_RETRIEVAL_K = int(os.getenv("INITIAL_RETRIEVAL_K", 10))
RERANKED_TOP_N = int(os.getenv("RERANKED_TOP_N", 5))
VECTOR_STORE_POOL_WORKERS = int(os.getenv("VECTOR_STORE_POOL_WORKERS", 2))

# --- Vector Store Settings ---
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "parser", "vector_db", "law_docs"))
DOC_COLLECTION = os.getenv("DOC_COLLECTION", "law_docs_v1")

import warnings

# --- Authentication Settings ---
SECRET_KEY = os.getenv("SECRET_KEY")
if SECRET_KEY is None:
    # In a real production scenario, you might raise an error or have a more robust check.
    # For now, use a clearly insecure default and issue a strong warning.
    SECRET_KEY = "a_very_secret_key_for_dev_DO_NOT_USE_IN_PROD"
    warnings.warn(
        "SECURITY WARNING: SECRET_KEY is not set in environment variables. "
        "Using a default, insecure key. THIS IS NOT SAFE FOR PRODUCTION. "
        "Please set a strong SECRET_KEY environment variable.",
        UserWarning
    )
elif SECRET_KEY == "a_very_secret_key_for_dev_DO_NOT_USE_IN_PROD":
    warnings.warn(
        "SECURITY WARNING: SECRET_KEY is set to the default insecure key. "
        "THIS IS NOT SAFE FOR PRODUCTION. Please set a strong, unique SECRET_KEY environment variable.",
        UserWarning
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# --- CORS Settings ---
# Expecting a comma-separated string for multiple origins, e.g., "http://localhost:8501,https://yourdomain.com"
ALLOWED_ORIGINS_STR = os.getenv("API_ALLOWED_ORIGINS", "http://localhost:8501")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",") if origin.strip()]
if not ALLOWED_ORIGINS:
    # Default to a restrictive set if empty after parsing, or if the env var was empty.
    ALLOWED_ORIGINS = ["http://localhost:8501"] # Or consider `[]` for maximum restriction by default
    warnings.warn(
        "INFO: API_ALLOWED_ORIGINS environment variable not set or empty. "
        "Defaulting to allow only 'http://localhost:8501'. "
        "Set API_ALLOWED_ORIGINS for production environments.",
        UserWarning
    )


# --- Translation API ---
DEFAULT_TRANSLATION_URL = "https://b348-35-192-106-64.ngrok-free.app/translate" # Dev only ngrok URL
TRANSLATION_API_URL = os.getenv("TRANSLATION_API_URL", DEFAULT_TRANSLATION_URL)
ENABLE_TRANSLATION_CHAIN = True  # Set to False to disable translation steps

if TRANSLATION_API_URL == DEFAULT_TRANSLATION_URL and ENABLE_TRANSLATION_CHAIN:
    warnings.warn(
        "INFO: TRANSLATION_API_URL is using the default ngrok URL. "
        "This is suitable for development/testing only. For production, set a stable TRANSLATION_API_URL.",
        UserWarning
    )

# --- Semantic Cache Settings ---
CACHE_REDIS_HOST = os.getenv("CACHE_REDIS_HOST", "localhost")
CACHE_REDIS_PORT = int(os.getenv("CACHE_REDIS_PORT", 6379))
CACHE_REDIS_DB = int(os.getenv("CACHE_REDIS_DB", 0))
CACHE_NAME = os.getenv("CACHE_NAME", "llm_semantic_cache")
# It's often good to use a fast, lightweight model for cache embeddings
CACHE_EMBEDDING_MODEL = os.getenv("CACHE_EMBEDDING_MODEL", "nomic-embed-text:latest")
# Threshold for cache hit, from 0 (exact match) to 1 (very dissimilar)
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", 0.1))