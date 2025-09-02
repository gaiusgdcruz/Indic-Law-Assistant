from fastapi import APIRouter, Depends, JSONResponse
from fastapi_cache.decorator import cache

from ..auth import User, get_current_active_user
from ..models import get_llm_generator

router = APIRouter(
    prefix="/health",
    tags=["Health"],
    dependencies=[Depends(get_current_active_user)],
)


@router.get("")
@cache(expire=60)
async def health_check():
    """
    Health check endpoint that also verifies core components like the
    vector store and the LLM generator.
    """
    try:
        from ..vector_store import _vector_store

        if _vector_store is None:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "detail": "Vector store not initialized"},
            )

        llm = get_llm_generator()
        if llm is None:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "detail": "LLM not initialized"},
            )

        # A more thorough check could involve a simple query to the components
        # e.g., a test query to the LLM or checking the vector store count.

        return {"status": "ok", "components": {"vector_store": "ok", "llm": "ok"}}
    except Exception as e:
        return JSONResponse(
            status_code=503, content={"status": "error", "detail": str(e)}
        )
