from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        logger.info(f"Initializing SentenceTransformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            # Test the model
            test_embedding = self.model.encode("test")
            logger.info(f"SentenceTransformer initialized successfully. Embedding dimension: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Embedding {len(texts)} documents...")
        embeddings = self.model.encode(texts)
        logger.info("Document embedding complete.")
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        logger.info(f"Embedding query: {text[:50]}...")
        embedding = self.model.encode(text)
        logger.info("Query embedding complete.")
        return embedding.tolist()