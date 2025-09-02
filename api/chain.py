import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)

from . import config
from .cache import get_semantic_cache
from .models import get_llm_generator, get_llm_rewriter_reranker
from .prompts import LEGAL_QUERY_CLASSIFIER_PROMPT, answer_prompt, rewrite_prompt
from .reranker import get_reranker
from .utils import format_docs
from .vector_store import _initialize_vector_store
from core.logging_utils import get_logger

logger = get_logger(__name__)

# --- Globals for lazy loading ---
_rag_pipeline = None
_pipeline_lock = Lock()
_vector_store_executor = None


def get_vector_store_executor():
    """Lazy initializer for the ThreadPoolExecutor."""
    global _vector_store_executor
    if _vector_store_executor is None:
        _vector_store_executor = ThreadPoolExecutor(max_workers=config.VECTOR_STORE_POOL_WORKERS)
    return _vector_store_executor


async def get_relevant_documents(query: str, vectorstore) -> list:
    """Retrieves and reranks documents asynchronously."""
    reranker_instance = get_reranker()
    executor = get_vector_store_executor()
    loop = asyncio.get_running_loop()

    initial_docs = await loop.run_in_executor(
        executor, vectorstore.similarity_search, query, config.INITIAL_RETRIEVAL_K
    )
    if not initial_docs:
        return []

    return reranker_instance.rerank_documents(
        query, initial_docs, config.RERANKED_TOP_N
    )


def build_rag_chain() -> RunnableSerializable:
    """Builds the core RAG pipeline."""
    vectorstore = _initialize_vector_store()
    if not vectorstore:
        raise ValueError("Failed to initialize vectorstore")

    llm_generator = get_llm_generator()
    llm_rewriter_reranker = get_llm_rewriter_reranker()

    # --- Chains ---
    classification_chain = (
        LEGAL_QUERY_CLASSIFIER_PROMPT | llm_rewriter_reranker | StrOutputParser()
    )

    legal_chain = (
        RunnablePassthrough.assign(
            rewritten_query=rewrite_prompt | llm_rewriter_reranker | StrOutputParser()
        )
        .assign(docs=lambda x: get_relevant_documents(x["rewritten_query"], vectorstore))
        .assign(context=lambda x: format_docs(x.get("docs", [])))
        .assign(answer=answer_prompt | llm_generator | StrOutputParser())
        | (lambda x: {"answer": x.get("answer", ""), "sources": x.get("docs", [])})
    )

    general_chain = (
        (lambda x: x["query"])
        | llm_generator
        | StrOutputParser()
        | (lambda x: {"answer": x, "sources": []})
    )

    branch = RunnableBranch(
        (lambda x: "LEGAL" in x["classification"].upper(), legal_chain),
        general_chain,
    )

    return RunnablePassthrough.assign(classification=classification_chain) | branch


class CachingRunnable(Runnable):
    """A wrapper that adds semantic caching to the RAG pipeline."""
    
    def __init__(self, runnable: Runnable):
        self.runnable = runnable
        self.cache = get_semantic_cache()

    def invoke(self, input_data: dict, config=None, **kwargs) -> dict:
        """Synchronous invocation with caching."""
        query = input_data.get("query", "")
        if not query or not self.cache:
            return self.runnable.invoke(input_data, config, **kwargs)

        cached_response = self.cache.check(prompt=query)
        if cached_response:
            logger.info(f"Cache hit for query: '{query[:50]}...'")
            return {"answer": cached_response[0]['response'], "sources": "From Cache"}

        logger.info(f"Cache miss for query: '{query[:50]}...'.")
        result = self.runnable.invoke(input_data, config, **kwargs)

        if isinstance(result, dict) and result.get("answer"):
            self.cache.store(prompt=query, response=result["answer"])
            logger.info(f"Stored new response in cache for query: '{query[:50]}...'")

        return result

    async def ainvoke(self, input_data: dict, config=None, **kwargs) -> dict:
        """Asynchronous invocation with caching."""
        query = input_data.get("query", "")
        if not query or not self.cache:
            return await self.runnable.ainvoke(input_data, config, **kwargs)

        loop = asyncio.get_running_loop()
        cached_response = await loop.run_in_executor(None, self.cache.check, query)

        if cached_response:
            logger.info(f"Semantic cache hit for query: '{query[:50]}...'")
            return {"answer": cached_response[0]['response'], "sources": "From Cache"}

        logger.info(f"Semantic cache miss for query: '{query[:50]}...'.")
        result = await self.runnable.ainvoke(input_data, config, **kwargs)

        if isinstance(result, dict) and result.get("answer"):
            await loop.run_in_executor(None, self.cache.store, query, result["answer"])
            logger.info(f"Stored new response in semantic cache for query: '{query[:50]}...'")

        return result


def get_rag_pipeline() -> Runnable:
    """
    Initializes and returns the full RAG pipeline (with caching) using a
    thread-safe, lazy-loading pattern.
    """
    global _rag_pipeline
    with _pipeline_lock:
        if _rag_pipeline is None:
            logger.info("Initializing RAG pipeline for the first time...")
            try:
                core_chain = build_rag_chain()
                _rag_pipeline = CachingRunnable(runnable=core_chain)
                logger.info("RAG pipeline with caching initialized successfully.")
            except Exception as e:
                logger.error(f"Fatal error during pipeline initialization: {e}", exc_info=True)
            # Bind the exception to the lambda's default arguments to capture its value
            _rag_pipeline = RunnableLambda(lambda x, e=e: {"answer": f"Error: Pipeline failed to initialize. {e}", "sources": []})
    return _rag_pipeline
