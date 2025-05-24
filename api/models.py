# Replace old logging setup with the new central one
from core.logging_utils import get_logger
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from . import config

logger = get_logger(__name__) # Use the new central logger

# Cache instances to avoid re-initialization on every request
_llm_generator = None
_llm_rewriter_reranker = None
_embeddings = None


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper class for SentenceTransformer to conform to Langchain Embeddings interface."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        """Initializes the SentenceTransformer model."""
        logger.info(f"Initializing SentenceTransformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            test_embedding = self.model.encode("test")
            logger.info(f"SentenceTransformer initialized successfully. Embedding dimension: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of documents."""
        logger.info(f"Embedding {len(texts)} documents...")
        embeddings = self.model.encode(texts)
        logger.info("Document embedding complete.")
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query."""
        logger.info(f"Embedding query: {text[:50]}...")
        embedding = self.model.encode(text)
        logger.info("Query embedding complete.")
        return embedding.tolist()


def get_llm_generator() -> ChatOllama:
    """Initializes and returns the generator LLM instance."""
    global _llm_generator
    if _llm_generator is None:
        logger.info(f"Initializing Generator LLM: {config.GENERATOR_MODEL}")
        _llm_generator = ChatOllama(
            model=config.GENERATOR_MODEL,
            temperature=0.6,
            base_url=config.OLLAMA_BASE_URL
        )
        try:
            response = _llm_generator.invoke("test")
            logger.info("LLM generator initialized successfully")
        except Exception as e:
            logger.error(f"Error testing LLM generator: {e}", exc_info=True)
            raise
    return _llm_generator


def get_llm_rewriter_reranker() -> ChatOllama:
    """Initializes and returns the rewriter/reranker LLM instance."""
    global _llm_rewriter_reranker
    if _llm_rewriter_reranker is None:
        logger.info(f"Initializing Rewriter/Reranker LLM: {config.REWRITER_RERANKER_MODEL}")
        _llm_rewriter_reranker = ChatOllama(
            model=config.REWRITER_RERANKER_MODEL,
            temperature=0,
            base_url=config.OLLAMA_BASE_URL
        )
        try:
            response = _llm_rewriter_reranker.invoke("test")
            logger.info("LLM rewriter/reranker initialized successfully")
        except Exception as e:
            logger.error(f"Error testing LLM rewriter/reranker: {e}", exc_info=True)
            raise
    return _llm_rewriter_reranker


def get_embeddings():
    """Get the embeddings model (now using SentenceTransformer)."""
    global _embeddings
    if _embeddings is None:
        try:
            _embeddings = SentenceTransformerEmbeddings()
        except Exception as e:
            raise
    return _embeddings