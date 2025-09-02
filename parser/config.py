import os
import multiprocessing
from dotenv import load_dotenv

# Load .env file from the project root, which is one level up from the parser directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Base Directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Folder Configurations ---
# It's better to define paths relative to the project root or use absolute paths from env vars.
# These defaults assume the parser module is part of a larger project structure.
UPLOAD_FOLDER = os.getenv("PARSER_UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
PROCESSED_FOLDER = os.getenv("PARSER_PROCESSED_FOLDER", os.path.join(BASE_DIR, "processed_pdfs"))
METADATA_FOLDER = os.getenv("PARSER_METADATA_FOLDER", os.path.join(BASE_DIR, "metadata"))
LOG_FOLDER = os.getenv("PARSER_LOG_FOLDER", os.path.join(BASE_DIR, "logs"))

# --- Vector Store Configuration (align with API) ---
# Use the same environment variable as the API module for the ChromaDB path to ensure consistency.
VECTOR_STORE_FOLDER = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "vector_db", "law_docs"))
COLLECTION_NAME = os.getenv("DOC_COLLECTION", "law_docs_v1")

# --- File Configurations ---
TRACKING_FILE = os.getenv("PARSER_TRACKING_FILE", os.path.join(BASE_DIR, "tracking.json"))

# --- Text Processing Configurations ---
CHUNK_SIZE = int(os.getenv("PARSER_CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("PARSER_CHUNK_OVERLAP", 100))

# --- Performance Configurations ---
# Number of worker processes to use for parsing PDFs. Defaults to half the CPU cores.
DEFAULT_WORKER_COUNT = max(1, multiprocessing.cpu_count() // 2)
PARSER_WORKER_COUNT = int(os.getenv("PARSER_WORKER_COUNT", DEFAULT_WORKER_COUNT))

# Number of documents to batch together when writing to ChromaDB.
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", 100))

# --- Embedding Model Configuration (align with API) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu") # Use EMBEDDING_DEVICE to align with API

# --- Ensure directories exist ---
# This is convenient for local development but might be better handled
# by a separate setup script in a production environment.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(METADATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# --- Debugging ---
# Use a logger instead of print for production code
# import logging
# logging.info(f"PARSER CHUNK_SIZE: {CHUNK_SIZE}")
# logging.info(f"PARSER_WORKER_COUNT: {PARSER_WORKER_COUNT}")
# logging.info(f"CHROMA_BATCH_SIZE: {CHROMA_BATCH_SIZE}")
