from redisvl.extensions.llmcache import SemanticCache
from langchain_ollama import OllamaEmbeddings
from redisvl.utils.vectorize import CustomTextVectorizer
from typing import List
import asyncio

from . import config
from core.logging_utils import get_logger

logger = get_logger(__name__)

# --- Global Cache Instance ---
_semantic_cache = None

def create_vectorizer_for_cache():
    """Creates a custom vectorizer for the semantic cache using Ollama."""
    try:
        # The embedding model for the cache should be fast and effective.
        ollama_embedder = OllamaEmbeddings(model=config.CACHE_EMBEDDING_MODEL)

        # RedisVL requires synchronous functions for its vectorizer.
        def sync_embed(text: str) -> List[float]:
            return ollama_embedder.embed_query(text)

        def sync_embed_many(texts: List[str]) -> List[List[float]]:
            return ollama_embedder.embed_documents(texts)

        # Async wrappers for compatibility if needed elsewhere, though RedisVL uses sync.
        async def async_embed(text: str) -> List[float]:
            return await asyncio.to_thread(sync_embed, text)

        async def async_embed_many(texts: List[str]) -> List[List[float]]:
            return await asyncio.to_thread(sync_embed_many, texts)

        return CustomTextVectorizer(
            embed=sync_embed,
            aembed=async_embed,
            embed_many=sync_embed_many,
            aembed_many=async_embed_many
        )
    except Exception as e:
        logger.error(f"Failed to create Ollama vectorizer for cache: {e}", exc_info=True)
        return None

def get_semantic_cache() -> SemanticCache:
    """
    Initializes and returns the SemanticCache instance using a lazy-loading pattern.
    """
    global _semantic_cache
    if _semantic_cache is None:
        logger.info("Initializing semantic cache for the first time...")
        vectorizer = create_vectorizer_for_cache()
        if not vectorizer:
            logger.error("Cannot initialize semantic cache without a vectorizer.")
            return None

        try:
            redis_url = f"redis://{config.CACHE_REDIS_HOST}:{config.CACHE_REDIS_PORT}/{config.CACHE_REDIS_DB}"
            _semantic_cache = SemanticCache(
                name=config.CACHE_NAME,
                redis_url=redis_url,
                distance_threshold=config.CACHE_THRESHOLD,
                vectorizer=vectorizer,
                connection_kwargs={
                    'decode_responses': True,
                    'socket_timeout': 5,
                    'retry_on_timeout': True
                }
            )
            logger.info(f"Semantic cache '{config.CACHE_NAME}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticCache: {e}", exc_info=True)
            # Set to None so the next call can try again
            _semantic_cache = None

    return _semantic_cache
