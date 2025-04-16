import os
from dotenv import load_dotenv

load_dotenv()

# --- General Settings ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Ollama Settings ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gemma3:1b")
REWRITER_RERANKER_MODEL = os.getenv("REWRITER_RERANKER_MODEL", "gemma3:1b")

# Embedding model (used by SentenceTransformer in models.py)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# --- RAG Pipeline Settings ---
INITIAL_RETRIEVAL_K = int(os.getenv("INITIAL_RETRIEVAL_K", 10))
RERANKED_TOP_N = int(os.getenv("RERANKED_TOP_N", 5))

# --- Vector Store Settings ---
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "parser", "vector_db", "law_docs"))
DOC_COLLECTION = os.getenv("DOC_COLLECTION", "law_docs_v1")

# --- Authentication Settings ---
SECRET_KEY = os.getenv("SECRET_KEY", "a_very_secret_key_for_dev")  # CHANGE IN PRODUCTION!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# --- Translation API ---
TRANSLATION_API_URL = "https://b348-35-192-106-64.ngrok-free.app/translate"
ENABLE_TRANSLATION_CHAIN = True  # Set to False to disable translation steps