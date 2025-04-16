import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)

from . import config
from .models import (
    get_embeddings,
    get_llm_generator,
    get_llm_rewriter_reranker,
)
from .prompts import answer_prompt, rewrite_prompt
from .reranker import get_reranker  # Use the singleton getter function
from .utils import format_chat_history, format_docs
from .vector_store import _initialize_vector_store
from .schemas import Language
from .config import ENABLE_TRANSLATION_CHAIN
from .utils import translate_text

logger = logging.getLogger(__name__)

# Create a global executor for vector store operations
_vector_store_executor = ThreadPoolExecutor(max_workers=2)
logger.info("Created global ThreadPoolExecutor for vector store operations")


def is_legal_query(query: str) -> bool:
    """Determine if the query needs legal document retrieval."""
    legal_indicators = [
        "section", "act", "law", "ipc", "crpc", "constitution", "court",
        "legal", "rights", "case", "judgment", "supreme court", "high court",
        "petition", "article", "statute", "legislation", "indian", "india"
    ]
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in legal_indicators)


try:
    # Initialize vectorstore
    vectorestore = _initialize_vector_store()
    # Get reranker using the singleton pattern
    reranker = get_reranker()
    logger.info("Reranker and vectorstore initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise


async def get_relevant_documents(query: str, initial_k: int = config.INITIAL_RETRIEVAL_K, final_k: int = config.RERANKED_TOP_N, vectorstore=None) -> list:
    """Retrieves and reranks documents using the enhanced reranking method."""
    if vectorstore is None:
        logger.error("No vectorstore provided")
        return []

    try:
        # Add debug logging
        logger.info(f"Attempting similarity search with k={initial_k}")
        logger.info(f"Query: {query[:100]}...")

        # Get initial results with error handling
        try:
            loop = asyncio.get_running_loop()
            # Use the global executor to avoid creating/destroying executors
            initial_docs = await loop.run_in_executor(_vector_store_executor, 
                                                     vectorstore.similarity_search, 
                                                     query, 
                                                     initial_k)
            logger.info(f"Initial retrieval found {len(initial_docs)} documents")
        except Exception as search_error:
            logger.error(f"Similarity search failed: {search_error}")
            return []

        if not initial_docs:
            logger.warning("No documents found in initial retrieval")
            return []

        # Log retrieved content for debugging
        for i, doc in enumerate(initial_docs[:3]):
            logger.info(f"Doc {i+1} Preview: {doc.page_content[:200]}...")
            logger.info(f"Doc {i+1} Metadata: {doc.metadata}")

        # Rerank with enhanced method - use synchronous call
        try:
            # Get the reranker instance (singleton)
            reranker_instance = get_reranker()
            # Call directly without using executor to avoid shutdown issues
            reranked_docs = reranker_instance.rerank_documents(query, initial_docs, final_k, (0.5, 0.3, 0.2))
            logger.info(f"Reranking completed. Got {len(reranked_docs)} documents")
        except Exception as rerank_error:
            logger.error(f"Reranking failed: {rerank_error}")
            return initial_docs[:final_k]  # Fall back to initial docs if reranking fails

        return reranked_docs

    except Exception as e:
        logger.error(f"Error in get_relevant_documents: {e}", exc_info=True)
        return []


async def embed_and_retrieve(query, vectorstore):
    """Retrieves and reranks documents using enhanced reranking."""
    try:
        logger.info(f"Attempting to retrieve documents for query: {query[:50]}...")

        if vectorstore is None:
            logger.error("Vectorstore is None in embed_and_retrieve")
            return []

        # Pass the vectorstore explicitly
        docs = await get_relevant_documents(
            query=query,
            initial_k=config.INITIAL_RETRIEVAL_K,
            final_k=config.RERANKED_TOP_N,
            vectorstore=vectorstore  # Pass it explicitly
        )

        logger.info(f"Retrieved and reranked. Found {len(docs)} documents.")

        if docs:
            doc_sources = []
            for i, doc in enumerate(docs[:5], 1):
                score = doc.metadata.get('combined_score', 0.0)  # Ensure default is 0.0
                source = doc.metadata.get('source', 'Unknown')
                doc_sources.append(f"{i}. {source} (score: {score:.3f})")
            logger.info(f"Top documents:\n" + "\n".join(doc_sources))
        else:
            logger.warning("No documents retrieved")

        return docs
    except Exception as e:
        logger.error(f"Error during document retrieval: {e}", exc_info=True)
        return []


async def translation_chain(text: str, source_lang: str, target_lang: str) -> str:
    if not ENABLE_TRANSLATION_CHAIN or source_lang == target_lang:
        return text
    try:
        logger.info(f"Translating from {source_lang} to {target_lang}")
        return await translate_text(text, source_lang, target_lang)
    except Exception as e:
        logger.error(f"Translation failed ({source_lang}->{target_lang}): {e}", exc_info=True)
        return f"[Translation Error: {e}] {text}"


def build_rag_chain() -> RunnableSerializable:
    """Builds the complete RAG LCEL chain."""
    logger.info("Building RAG chain...")

    # Get component instances
    try:
        llm_generator = get_llm_generator()
        llm_rewriter_reranker = get_llm_rewriter_reranker()
        embeddings = get_embeddings()

        # Initialize vectorstore first and check it
        vectorstore = _initialize_vector_store()
        if vectorstore is None:
            raise ValueError("Failed to initialize vectorstore")

        logger.info("All components initialized successfully")
        # Log collection info for debugging
        try:
            collection_count = vectorstore._collection.count()
            logger.info(f"Vectorstore initialized with {collection_count} documents")
        except Exception as e:
            logger.warning(f"Could not get collection count: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        raise

    async def retrieve_docs(x):
        return await embed_and_retrieve(x['rewritten_query'], vectorstore)

    # 1. Query Rewriting Chain
    try:
        query_rewriter_chain = (
            rewrite_prompt
            | llm_rewriter_reranker
            | StrOutputParser()
        )
        logger.info("Query rewriter chain built successfully")
    except Exception as e:
        logger.error(f"Failed to build query rewriter chain: {e}", exc_info=True)
        raise
    
    def simple_chat_chain(input_data) -> dict:
        try:
            query = input_data if isinstance(input_data, str) else input_data.get('query', '')
            language = input_data.get('language', Language.ENGLISH) if isinstance(input_data, dict) else Language.ENGLISH
            
            logger.info(f"Processing simple chat query: {query[:50]}... in {language}")
            response = llm_generator.invoke(query)

            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)

            logger.info("Simple chat response generated successfully")
            return {
                "answer": answer,
                "sources": [],
                "rewritten_query": None,
                "context": None,
                "confidence": 1.0,
                "language": language
            }
        except Exception as e:
            logger.error(f"Error in simple chat chain: {e}", exc_info=True)
            return {
                "answer": "I apologize, but I encountered an error processing your query.",
                "sources": [],
                "rewritten_query": None,
                "context": None,
                "confidence": 0.0,
                "language": Language.ENGLISH
            }

    try:
        rag_chain = (
            RunnableLambda(lambda x: {"query": x} if isinstance(x, str) else x)
            | RunnablePassthrough.assign(original_query=lambda x: x["query"])
            | RunnableLambda(lambda x: logger.info(f"Step 1: Original Query - {x['original_query'][:50]}...") or x)
            | RunnablePassthrough.assign(
                rewritten_query=lambda x: query_rewriter_chain.invoke(x["original_query"])
            )
            | RunnableLambda(lambda x: logger.info(f"Step 2: Rewritten Query - {x['rewritten_query'][:50]}...") or x)
            | RunnablePassthrough.assign(
                retrieved_docs=RunnableLambda(retrieve_docs)
            )
            | RunnableLambda(lambda x: logger.info(f"Step 3: Retrieved {len(x['retrieved_docs'])} docs") or x)
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(x['retrieved_docs']) if x['retrieved_docs']
                else "No relevant legal documents found. Please try rephrasing your question or provide more specific legal terms."
            )
            | RunnableLambda(lambda x: logger.info(f"Step 4: Context length: {len(x['context'])}") or x)
            | RunnablePassthrough.assign(
                final_answer=(
                    RunnablePassthrough.assign(
                        prompt=lambda x: answer_prompt.format(
                            context=x['context'],
                            chat_history=format_chat_history(x.get('chat_history', [])) if x.get('chat_history') else "No previous conversation.",
                            question=x['original_query']
                        )
                    )
                    | RunnableLambda(lambda x: logger.info("Step 5: Generating final answer...") or x)
                    | (lambda x: llm_generator.invoke(x['prompt']))
                    | StrOutputParser()
                )
            )
            | RunnableLambda(lambda x: logger.info(f"Final chain output keys: {x.keys()}") or x)
            | (lambda x: {
                "answer": str(x["final_answer"]),
                "sources": [],
                "context": x["context"],
                "rewritten_query": x["rewritten_query"],
                "confidence": min(1.0, len(x["retrieved_docs"]) * 0.2) if x["retrieved_docs"] else 0.0
            })
        )
        logger.info("RAG chain assembled successfully")
    except Exception as e:
        logger.error(f"Failed to assemble RAG chain: {e}", exc_info=True)
        raise

    async def rag_chain_with_translation(input_data):
        try:
            query = input_data.get("query", "")
            chat_history = input_data.get("chat_history", [])
            source_lang = input_data.get("source_language", "en")
            target_lang = input_data.get("target_language", source_lang)

            if ENABLE_TRANSLATION_CHAIN and source_lang != "en":
                query = await translation_chain(query, source_lang, "en")

            rag_input = {
                "query": query,
                "chat_history": chat_history,
            }
            result = await rag_chain.ainvoke(rag_input)

            answer = result.get("answer", "")
            if ENABLE_TRANSLATION_CHAIN and source_lang != "en":
                answer = await translation_chain(answer, "en", source_lang)
                result["answer"] = answer

            return result
        except Exception as e:
            logger.error(f"Error in rag_chain_with_translation: {e}", exc_info=True)
            return {
                "answer": "Sorry, an error occurred while processing your legal query.",
                "sources": [],
                "rewritten_query": None,
                "context": None,
                "confidence": 0.0,
                "language": input_data.get("source_language", "en")
            }

    simple_chat_chain_runnable = RunnableLambda(simple_chat_chain)
    combined_chain = RunnableBranch(
        (lambda x: is_legal_query(x) if isinstance(x, str) else is_legal_query(x.get("query", "")),
         RunnableLambda(rag_chain_with_translation)),
        simple_chat_chain_runnable
    )

    logger.info("RAG chain built successfully")
    return combined_chain


rag_pipeline = build_rag_chain()