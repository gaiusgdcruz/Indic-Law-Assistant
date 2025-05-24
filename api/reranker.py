# Replace old logging setup with the new central one
from core.logging_utils import get_logger
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from .config import REWRITER_RERANKER_MODEL
from .prompts import rerank_prompt

logger = get_logger(__name__) # Use the new central logger


# Create a global executor that will be shared across all rerankers
# This prevents the executor from being created and destroyed constantly
_global_executor = ThreadPoolExecutor(max_workers=4)
logger.info("Created global ThreadPoolExecutor for reranking")


class EnhancedReranker:
    """Enhanced reranking using multiple methods."""

    def __init__(self, model_name: str):
        """Initialize reranker with model name."""
        try:
            self.llm = ChatOllama(
                model=model_name,
                temperature=0.1,
                format="json",
            )
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            # Use the global executor instead of creating a new one
            self.executor = _global_executor
            logger.info(f"EnhancedReranker initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedReranker: {e}")
            raise

    # Remove __del__ method completely - we don't want to shut down the executor
    # when an instance of EnhancedReranker is garbage collected

    def semantic_similarity_score(self, query: str, doc: Document) -> float:
        """Get semantic similarity score using cross-encoder."""
        try:
            score = self.cross_encoder.predict([(query, doc.page_content)])
            score = float(score)
            logger.debug(f"Semantic score: {score}")
            return score
        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            return 0.0

    def llm_relevance_score(self, query: str, doc: Document) -> float:
        """Get relevance score from LLM."""
        try:
            formatted_prompt = rerank_prompt.format(
                query=query, document=doc.page_content
            )
            response = self.llm.invoke(formatted_prompt)

            try:
                if hasattr(response, "content"):
                    score_text = response.content
                else:
                    score_text = str(response)

                score = float("".join(c for c in score_text if c.isdigit() or c == ".") or "0")
                score = max(0, min(10, score))
                logger.debug(f"LLM score: {score}")
                return score
            except ValueError:
                logger.warning(f"Could not parse score from response: {response}")
                return 0.0

        except Exception as e:
            logger.error(f"LLM scoring failed: {e}", exc_info=True)
            return 0.0

    def metadata_score(self, doc: Document) -> float:
        """Score based on metadata quality and relevance."""
        score = 0.0
        metadata = doc.metadata

        if metadata.get("title"):
            score += 0.5
        if metadata.get("page_number"):
            score += 0.3

        page_num = metadata.get("page_number", 1)
        score += max(0, 1 - (page_num * 0.01))

        logger.debug(f"Metadata score: {score}")
        return min(score, 1.0)

    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_n: int,
        weights: Optional[Tuple[float, float, float]] = (0.5, 0.3, 0.2),
    ) -> List[Document]:
        """Rerank documents using multiple scoring methods."""
        if not documents:
            logger.warning("No documents to rerank")
            return []

        scored_docs = []
        BATCH_SIZE = 5  # Process in smaller batches

        try:
            for i in range(0, len(documents), BATCH_SIZE):
                batch = documents[i:i + BATCH_SIZE]
                futures = []
                
                try:
                    logger.debug(f"Processing batch {i//BATCH_SIZE + 1}, size {len(batch)}")
                    
                    # Score each document synchronously to avoid executor issues completely
                    # This is slower but more robust
                    for doc in batch:
                        try:
                            result = self.score_doc(query, doc, weights)
                            scored_docs.append(result)
                        except Exception as e:
                            logger.error(f"Error scoring document: {e}")
                            
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    continue

            # Sort all scored documents
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Successfully reranked {len(scored_docs)}/{len(documents)} documents")
            return [doc for doc, _ in scored_docs[:top_n]] if scored_docs else documents[:top_n]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            return documents[:top_n]  # Return original docs if reranking fails

    def score_doc(self, query: str, doc: Document, weights: Tuple[float, float, float]) -> Tuple[Document, float]:
        """Score a single document."""
        try:
            semantic_score = self.semantic_similarity_score(query, doc)
            llm_score = self.llm_relevance_score(query, doc)
            metadata_score = self.metadata_score(doc)
            
            combined_score = (
                weights[0] * semantic_score +
                weights[1] * llm_score +
                weights[2] * metadata_score
            )
            
            doc.metadata.update({
                "semantic_score": semantic_score,
                "llm_score": llm_score,
                "metadata_score": metadata_score,
                "combined_score": combined_score
            })
            
            return (doc, combined_score)
        except Exception as e:
            logger.error(f"Error scoring document: {e}", exc_info=True)
            return (doc, 0.0)


def parse_rerank_scores(results: list[str]) -> list[float]:
    """Parses scores from LLM responses."""
    scores = []
    for result in results:
        try:
            cleaned = "".join(c for c in result if c.isdigit() or c == ".")
            if not cleaned:
                logger.warning(f"No numeric value found in '{result}'. Using 0.")
                scores.append(0)
                continue

            score = float(cleaned)
            score = max(0, min(10, score))
            scores.append(score)
            logger.debug(f"Successfully parsed score: {score}")
        except Exception as e:
            logger.warning(f"Failed to parse score from '{result}': {e}")
            scores.append(0)
    return scores


# Singleton instance
_reranker_instance = None

def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = EnhancedReranker(model_name=REWRITER_RERANKER_MODEL)
    return _reranker_instance