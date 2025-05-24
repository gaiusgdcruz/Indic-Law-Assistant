import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator, List, Optional

import uvicorn
import httpx
from fastapi import (
    FastAPI,
    BackgroundTasks,
    Depends,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from jose import JWTError, jwt

from .auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    User,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    fake_users_db,
    get_current_active_user,
    get_user,
)
from .chain import is_legal_query, rag_pipeline
# Ensure config is imported to access its variables
from . import config
# from .config import GENERATOR_MODEL, REWRITER_RERANKER_MODEL, TRANSLATION_API_URL, ENABLE_TRANSLATION_CHAIN, SECRET_KEY, ALGORITHM, ALLOWED_ORIGINS
# Replace old logging setup with the new central one
from core.logging_utils import get_logger
from .models import get_llm_generator
from .prompts import answer_prompt
from .schemas import ChatMessage, QueryRequest, Token, Language, TranslationRequest, TranslationResponse
from .utils import format_chat_history, translate_text  # Import translate_text from utils

logger = get_logger(__name__) # Use the new central logger


# Initialize FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    FastAPICache.init(InMemoryBackend())
    logger.info("FastAPI application startup complete")
    yield
    # Shutdown
    try:
        # Clean up any remaining streams
        async with stream_manager.lock:
            stream_manager.active_streams.clear()
        # Force garbage collection
        import gc
        gc.collect()
    finally:
        logger.info("FastAPI application shutdown complete")


app = FastAPI(
    title="Modular RAG API with Ollama",
    description="API endpoint for interacting with a RAG pipeline using local LLMs via Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
# Initial startup warning for SECRET_KEY (already handled in config.py on import, but can be reiterated here if needed)
if config.SECRET_KEY == "a_very_secret_key_for_dev_DO_NOT_USE_IN_PROD":
    logger.warning(
        "CRITICAL SECURITY WARNING: The API is running with the default insecure SECRET_KEY. "
        "This key MUST be changed for production environments by setting a SECRET_KEY environment variable."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS, # Use the configured list
    allow_credentials=True, # Usually True for web apps with auth
    allow_methods=["*"],    # Or specify methods like ["GET", "POST"]
    allow_headers=["*"],    # Or specify allowed headers
)

# Constants
MIN_BUFFER_LENGTH = 50
MAX_BUFFER_LENGTH = 1000
SENTENCE_END_CHARS = {".", "!", "?"}
YIELD_CHARS = SENTENCE_END_CHARS.union({"\n", " "})


class StreamManager:
    def __init__(self):
        self.active_streams = set()
        self.lock = asyncio.Lock()

    async def add_stream(self, stream_id: str):
        async with self.lock:
            self.active_streams.add(stream_id)

    async def remove_stream(self, stream_id: str):
        async with self.lock:
            self.active_streams.discard(stream_id)

    async def is_stream_active(self, stream_id: str) -> bool:
        async with self.lock:
            return stream_id in self.active_streams


stream_manager = StreamManager()


async def process_chunk(content: str, response_buffer: str) -> tuple[str, bool]:
    """Process a chunk of content and determine if it should be yielded."""
    if not content:
        return response_buffer, False

    response_buffer += content
    trimmed_buffer = response_buffer.rstrip()

    if not trimmed_buffer:
        return response_buffer, False

    last_char = trimmed_buffer[-1]
    buffer_len = len(response_buffer)

    should_yield = (
        (buffer_len > 5 and (last_char in SENTENCE_END_CHARS or last_char == "\n"))
        or (buffer_len > MIN_BUFFER_LENGTH and last_char == " ")
        or (buffer_len > MAX_BUFFER_LENGTH)
    )

    return response_buffer, should_yield

async def generate_stream(
    query: str, stream_id: str, llm, is_legal: bool, chat_history: List[ChatMessage], language: Language
) -> AsyncGenerator[str, None]:
    try:
        # If translation is enabled and language is not English, translate to English
        # Use config.ENABLE_TRANSLATION_CHAIN and config.TRANSLATION_API_URL
        if config.ENABLE_TRANSLATION_CHAIN and language == Language.MALAYALAM:
            query = await translate_text(query, Language.MALAYALAM, Language.ENGLISH) # translate_text should use config.TRANSLATION_API_URL
        
        response_buffer = ""

        try:
            if not is_legal:
                # Format chat history for context
                history_text = format_chat_history(chat_history)
                prompt = f"""You are a helpful assistant. Consider the conversation so far:

{history_text}

Current question: {query}

Answer:"""
                async for chunk in llm.astream(prompt):
                    if not await stream_manager.is_stream_active(stream_id):
                        logger.info(f"Stream {stream_id} was cancelled")
                        break

                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    response_buffer, should_yield = await process_chunk(
                        content, response_buffer
                    )

                    if should_yield:
                        yield_content = response_buffer.strip()
                        if yield_content:
                            if language == Language.MALAYALAM:
                                translated_content = await translate_text(
                                    yield_content, Language.ENGLISH, Language.MALAYALAM
                                )
                                yield f"data: {json.dumps({'content': translated_content})}\n\n"
                            else:
                                yield f"data: {json.dumps({'content': yield_content})}\n\n"
                            response_buffer = ""
            else:
                logger.info(
                    f"Processing legal query with chat history: {len(chat_history)} messages"
                )
                try:
                    # Format chat history for RAG pipeline
                    formatted_chat_history = []
                    if chat_history:
                        for msg in chat_history:
                            # Handle ChatMessage objects directly
                            if isinstance(msg, ChatMessage):
                                formatted_chat_history.append(
                                    {"role": msg.role, "content": msg.content}
                                )
                            elif isinstance(msg, dict):  # Handle dictionaries
                                formatted_chat_history.append(ChatMessage(role=msg["role"], content=msg["content"]))

                    logger.info(f"Formatted chat history: {formatted_chat_history[:100]}")

                    # Log the input dictionary before invoking the RAG pipeline
                    input_data = {
                        "query": query,
                        "chat_history": formatted_chat_history
                        or [],  # Ensure we always pass a list
                    }
                    logger.info(f"Input data for RAG pipeline: {input_data}")

                    # Include formatted chat history in RAG pipeline
                    result = await rag_pipeline.ainvoke(input_data)

                    context = result.get("context", "")
                    formatted_history = formatted_chat_history

                    # Format the prompt
                    prompt = answer_prompt.format(
                        context=context, chat_history=formatted_history, question=query
                    )

                    async for chunk in llm.astream(prompt):
                        if not await stream_manager.is_stream_active(stream_id):
                            logger.info(f"Stream {stream_id} was cancelled")
                            break

                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        response_buffer, should_yield = await process_chunk(
                            content, response_buffer
                        )

                        if should_yield:
                            yield_content = response_buffer.strip()
                            if yield_content:
                                if language == Language.MALAYALAM:
                                    translated_content = await translate_text(
                                        yield_content, Language.ENGLISH, Language.MALAYALAM
                                    )
                                    yield f"data: {json.dumps({'content': translated_content})}\n\n"
                                else:
                                    yield f"data: {json.dumps({'content': yield_content})}\n\n"
                                response_buffer = ""

                except Exception as e:
                    logger.error(f"Error in legal query processing: {str(e)}", exc_info=True)
                    yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return

            # Yield any remaining content
            if response_buffer.strip():
                if language == Language.MALAYALAM:
                    translated_content = await translate_text(
                        response_buffer.strip(), Language.ENGLISH, Language.MALAYALAM
                    )
                    yield f"data: {json.dumps({'content': translated_content})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': response_buffer.strip()})}\n\n"

            # If translation is enabled and original language was Malayalam, translate back
            # Use config.ENABLE_TRANSLATION_CHAIN
            if config.ENABLE_TRANSLATION_CHAIN and language == Language.MALAYALAM:
                response_buffer = await translate_text(response_buffer.strip(), Language.ENGLISH, Language.MALAYALAM)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Error in stream generation: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Error in generate_stream: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"


# --- Authentication Endpoints ---
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: dict):
    user = authenticate_user(
        fake_users_db, form_data.get("username"), form_data.get("password")
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/token/refresh", response_model=dict)
async def refresh_access_token(payload: dict):
    """
    Refresh the access token using a valid refresh token.
    Expects JSON payload: {"refresh_token": "your_refresh_token_here"}
    """
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=400, detail="Missing refresh token")
    try:
        token_payload = jwt.decode(refresh_token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username: str = token_payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid refresh token (no subject)")
        user = get_user(fake_users_db, username)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_access_token(data={"sub": username}, expires_delta=access_token_expires)
    # Optionally, issue a new refresh token as well
    new_refresh_token = create_refresh_token(data={"sub": username}) # Assuming create_refresh_token also uses config.SECRET_KEY and config.ALGORITHM implicitly or via args
    return {"access_token": new_access_token, "refresh_token": new_refresh_token, "token_type": "bearer"}


# --- Protected API Endpoints ---
@app.post("/stream")
async def stream_response(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
):
    stream_id = f"{current_user.username}_{int(time.time())}"
    try:
        query = request.query.strip()
        if not query:
            raise ValueError("Empty query received")

        # Translate query to English if needed
        if request.language == Language.MALAYALAM:
            query = await translate_text(query, Language.MALAYALAM, Language.ENGLISH)

        # Use ChatMessage objects directly
        chat_history = request.chat_history
        logger.info(f"Received chat_history: {chat_history}")

        llm = get_llm_generator()
        logger.info(f"Processing query: {query}")

        # Generate unique stream ID
        await stream_manager.add_stream(stream_id)

        # Add cleanup task
        background_tasks.add_task(stream_manager.remove_stream, stream_id)

        # Check if it's a legal query
        is_legal = await asyncio.get_event_loop().run_in_executor(
            None, is_legal_query, query
        )
        logger.info(f"Query type: {'legal' if is_legal else 'general'}")

        return StreamingResponse(
            generate_stream(
                query=query,
                stream_id=stream_id,
                llm=llm,
                is_legal=is_legal,
                chat_history=chat_history,  # Pass as ChatMessage objects
                language=request.language,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
            background=background_tasks
        )
    except Exception as e:
        # Ensure stream is removed even if an error occurs
        await stream_manager.remove_stream(stream_id)
        logger.error(f"Error in stream_response: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.post("/rag_query")
async def query_rag_pipeline(
    request: QueryRequest, current_user: User = Depends(get_current_active_user)
):
    """Handles the RAG query request."""
    try:
        # Validate input type
        if not isinstance(request.query, str):
            raise ValueError(f"Invalid input type {type(request.query)}. Must be a string.")

        # Start timing
        start_time = time.time()

        # Process the query
        try:
            result = await asyncio.wait_for(
                rag_pipeline.ainvoke(request.query), timeout=30
            )  # Add timeout
        except asyncio.TimeoutError:
            logger.error("Document retrieval timed out")
            return JSONResponse(
                status_code=500, content={"detail": "Document retrieval timed out"}
            )
        except Exception as chain_error:
            logger.error(f"Chain error: {chain_error}")
            return JSONResponse(status_code=500, content={"detail": str(chain_error)})

        # Calculate processing time
        processing_time = time.time() - start_time

        # Ensure we have valid data
        if not isinstance(result, dict):
            result = {
                "answer": str(result),
                "sources": [],
                "rewritten_query": None,
                "confidence": 0.5,
            }

        # Format response
        response_data = {
            "answer": result.get(
                "answer", "I apologize, but I couldn't process your query properly."
            ),
            "sources": result.get("sources", []),
            "rewritten_query": result.get("rewritten_query"),
            "processing_time": processing_time,
            "confidence_score": result.get("confidence", 0.0),
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


# Cache the health check endpoint
@app.get("/health")
@cache(expire=60)
async def health_check(current_user: User = Depends(get_current_active_user)):
    """Health check endpoint that also verifies core components"""
    try:
        from .vector_store import _vector_store

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

        return {"status": "ok", "components": {"vector_store": "ok", "llm": "ok"}}
    except Exception as e:
        return JSONResponse(
            status_code=503, content={"status": "error", "detail": str(e)}
        )


# --- Translation API Endpoint ---
@app.post("/translate", response_model=TranslationResponse)
async def translate_text_endpoint(request: TranslationRequest):
    """Translate text between specified languages"""
    try:
        translated_text = await translate_text(request.text, request.source_lang, request.target_lang)
        return TranslationResponse(translated_text=translated_text)
    except Exception as e:
        error_detail = str(e)
        if error_detail == "'str' object is not callable":
            error_detail = (
                "Translation service error: "
                "A configuration error occurred where an expected function is a string. "
                "Please check your translation API configuration."
            )
        raise HTTPException(status_code=500, detail=error_detail)


# --- Run the API Server ---
if __name__ == "__main__":
    # Ensure Ollama server is running and models are pulled before starting
    print("Starting FastAPI server for RAG API...")
    print("Ensure Ollama is running with required models:")
    print(f"  Generator: {config.GENERATOR_MODEL}")
    print(f"  Rewriter/Reranker: {config.REWRITER_RERANKER_MODEL}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
