import logging
from typing import Optional, Any
from redisvl.extensions.llmcache import SemanticCache
from ..config import CACHE_TTL

logger = logging.getLogger(__name__)

async def get_cached_response(
    cache: SemanticCache,
    key: str,
    default: Any = None
) -> Optional[Any]:
    """Safely retrieve a cached response."""
    try:
        return await cache.get(key) or default
    except Exception as e:
        logger.error(f"Cache retrieval error: {e}")
        return default

async def set_cached_response(
    cache: SemanticCache,
    key: str,
    value: Any,
    ttl: int = CACHE_TTL
) -> bool:
    """Safely set a cached response."""
    try:
        await cache.set(key, value, ttl)
        return True
    except Exception as e:
        logger.error(f"Cache storage error: {e}")
        return False