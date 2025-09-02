import asyncio
import json
import time
from typing import AsyncGenerator, List

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from .. import config
from ..auth import User, get_current_active_user
from ..chain import get_rag_pipeline
from ..schemas import ChatMessage, QueryRequest, Language
from ..util.streaming import StreamManager
from ..utils import translate_text
from core.logging_utils import get_logger

router = APIRouter(
    tags=["RAG"],
    dependencies=[Depends(get_current_active_user)],
)

logger = get_logger(__name__)
stream_manager = StreamManager()


async def stream_rag_response(
    query: str, chat_history: List[ChatMessage], language: Language, stream_id: str
) -> AsyncGenerator[str, None]:
    """
    Streams the response from the RAG pipeline, checking for client disconnects.
    """
    try:
        rag_pipeline = get_rag_pipeline()

        if config.ENABLE_TRANSLATION_CHAIN and language == Language.MALAYALAM:
            query = await translate_text(query, Language.MALAYALAM, Language.ENGLISH)

        pipeline_input = {"query": query, "chat_history": chat_history}

        full_response = ""
        async for chunk in rag_pipeline.astream(pipeline_input):
            if not await stream_manager.is_stream_active(stream_id):
                logger.info(f"Stream {stream_id} was cancelled by client.")
                break

            answer_chunk = chunk.get("answer", "")
            if answer_chunk:
                full_response += answer_chunk

                if config.ENABLE_TRANSLATION_CHAIN and language == Language.MALAYALAM:
                    answer_chunk = await translate_text(answer_chunk, Language.ENGLISH, Language.MALAYALAM)

                yield f"data: {json.dumps({'content': answer_chunk})}\n\n"

        logger.info(f"Stream finished. Full response length: {len(full_response)}")
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Error in stream_rag_response: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("/stream")
async def stream_response(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
):
    """
    Handles the streaming request from the client, managing the stream lifecycle.
    """
    stream_id = f"{current_user.username}_{int(time.time())}"
    await stream_manager.add_stream(stream_id)
    background_tasks.add_task(stream_manager.remove_stream, stream_id)

    try:
        query = request.query.strip()
        if not query:
            raise ValueError("Empty query received")

        return StreamingResponse(
            stream_rag_response(
                query=query,
                chat_history=request.chat_history,
                language=request.language,
                stream_id=stream_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        # This cleanup is now redundant due to background_tasks, but good for safety
        await stream_manager.remove_stream(stream_id)
        logger.error(f"Error in stream_response endpoint: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.post("/rag_query")
async def query_rag_pipeline(request: QueryRequest):
    """
    Handles a non-streaming RAG query request using the new chain.
    """
    start_time = time.time()
    try:
        rag_pipeline = get_rag_pipeline()
        result = await rag_pipeline.ainvoke({"query": request.query, "chat_history": request.chat_history})

        processing_time = time.time() - start_time
        result["processing_time"] = processing_time

        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        logger.error("Document retrieval timed out")
        return JSONResponse(status_code=500, content={"detail": "Document retrieval timed out"})
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e)})
