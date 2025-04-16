from langchain_ollama import OllamaEmbeddings
from redisvl.utils.vectorize import CustomTextVectorizer
import asyncio
import config
from typing import List

def create_vectorizer():
    ollama_embedder = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    def sync_embed(text: str) -> List[float]:
        return ollama_embedder.embed_query(text)

    def sync_embed_many(texts: List[str]) -> List[List[float]]:
        return ollama_embedder.embed_documents(texts)

    # Define a wrapper for async single-text embedding
    async def async_embed(text: str) -> List[float]:
        # Run the synchronous method in a separate thread
        return await asyncio.to_thread(sync_embed, text)

    # Define a wrapper for async batch embedding
    async def async_embed_many(texts: List[str]) -> List[List[float]]:
        # Run the synchronous method in a separate thread
        return await asyncio.to_thread(sync_embed_many, texts)

    return CustomTextVectorizer(
        embed=sync_embed,
        aembed=async_embed,
        embed_many=sync_embed_many,
        aembed_many=async_embed_many
    )

vectorizer = create_vectorizer()