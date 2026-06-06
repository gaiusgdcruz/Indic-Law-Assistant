import logging
import os

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from . import config
from .models import get_embeddings

logger = logging.getLogger(__name__)

_vector_store = None


def _initialize_vector_store():
    """Creates or loads the Chroma vector store."""
    global _vector_store
    if _vector_store is None:
        try:
            logger.info("Initializing Chroma vector store...")
            embeddings = get_embeddings()
            logger.info("Embeddings model loaded successfully")
            
            _vector_store = Chroma(
                collection_name=config.DOC_COLLECTION,
                embedding_function=embeddings,
                persist_directory=config.CHROMA_DB_PATH,
                client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
            )
            
            # Add these debug logs
            try:
                count = _vector_store._collection.count()
                logger.info(f"Chroma collection '{config.DOC_COLLECTION}' initialized with {count} documents")
                if count == 0:
                    logger.error("Collection exists but contains no documents!")
            except Exception as e:
                logger.error(f"Failed to get collection count: {e}")
            
            return _vector_store
                
        except Exception as e:
            logger.error(f"Failed to initialize vectorstore: {e}", exc_info=True)
            return None
    return _vector_store


def get_retriever() -> VectorStoreRetriever:
    """Gets the vector store retriever."""
    try:
        logger.info("Getting vector store retriever...")
        vector_store = _initialize_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={'k': config.RETRIEVER_K}
        )
        logger.info(f"Retriever initialized with k={config.RETRIEVER_K}")
        return retriever
    except Exception as e:
        logger.error(f"Error getting retriever: {e}", exc_info=True)
        raise


# Initialize on import (or change to lazy loading)
# try:
#     _initialize_vector_store()
#     logger.info("Vector store initialized successfully on import")
# except Exception as e:
#     logger.error(f"Failed to initialize vector store on import: {e}", exc_info=True)
