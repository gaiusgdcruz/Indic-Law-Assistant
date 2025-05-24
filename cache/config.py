import os
from dotenv import load_dotenv

# Load .env file from the project root, which is one level up from the cache directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Embedding Model Configuration - Use a distinct name if it's for the cache specifically
CACHE_EMBEDDING_MODEL = os.getenv("CACHE_EMBEDDING_MODEL", "nomic-embed-text")
# Align with main RAG embedding model if the cache is for RAG results (see item #6 of Top 10)
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


CACHE_NAME = os.getenv("CACHE_NAME", "OllamaLLMCache")
CACHE_REDIS_HOST = os.getenv("CACHE_REDIS_HOST", "localhost")
CACHE_REDIS_PORT = int(os.getenv("CACHE_REDIS_PORT", 6379))
CACHE_REDIS_DB = int(os.getenv("CACHE_REDIS_DB", 0))

# Ollama model used by cache's sample.py, distinct from main API's models
CACHE_OLLAMA_MODEL = os.getenv("CACHE_OLLAMA_MODEL", "llama3.2:1b")

# For consistency, if this module's vectorizer is used with the main embedding model:
# EMBEDDING_MODEL_FOR_CACHE_VECTORIZER = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

print(f"CACHE_EMBEDDING_MODEL: {CACHE_EMBEDDING_MODEL}") # For debugging .env loading
