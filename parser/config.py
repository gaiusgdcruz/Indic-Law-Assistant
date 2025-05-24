import os
from dotenv import load_dotenv

# Load .env file from the project root, which is one level up from the parser directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Folder Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Parser module's directory

# It's often better to define paths relative to a project root determined at runtime,
# or make them absolute paths via env vars. For now, defaults are relative to this file's dir.
UPLOAD_FOLDER = os.getenv("PARSER_UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
STAGING_FOLDER = os.getenv("PARSER_STAGING_FOLDER", os.path.join(BASE_DIR, "staging_pdfs"))
PROCESSED_FOLDER = os.getenv("PARSER_PROCESSED_FOLDER", os.path.join(BASE_DIR, "processed_pdfs"))
METADATA_FOLDER = os.getenv("PARSER_METADATA_FOLDER", os.path.join(BASE_DIR, "metadata"))
# VECTOR_STORE_FOLDER is crucial and should align with api/config.py's CHROMA_DB_PATH if they are the same DB.
# For now, let's use the same env var name as in api/config.py for ChromaDB path to encourage consistency.
CHROMA_DB_PATH_PARSER_DEFAULT = os.path.join(BASE_DIR, "vector_db", "law_docs") # Default location within parser
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", CHROMA_DB_PATH_PARSER_DEFAULT)
VECTOR_STORE_FOLDER = CHROMA_DB_PATH # Use the same variable for clarity within this file

LOG_FOLDER = os.getenv("PARSER_LOG_FOLDER", os.path.join(BASE_DIR, "logs"))

# File Configurations
TRACKING_FILE = os.getenv("PARSER_TRACKING_FILE", os.path.join(BASE_DIR, "tracking.json"))

# Processing Configurations
CHUNK_SIZE = int(os.getenv("PARSER_CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("PARSER_CHUNK_OVERLAP", 100))
PDF_BATCH_SIZE = int(os.getenv("PARSER_PDF_BATCH_SIZE", 10)) # Number of PDFs to process by the main script loop (not worker)

# Embedding Model Configuration - Align with api/config.py by using same env var names
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DEVICE = os.getenv("PARSER_EMBEDDING_DEVICE", "cpu") # Parser might have different device needs

# ChromaDB Collection Name - Align with api/config.py
DOC_COLLECTION = os.getenv("DOC_COLLECTION", "law_docs_v1")
COLLECTION_NAME = DOC_COLLECTION # Use the same variable for clarity

# Ensure directories exist (consider if this should be done here or at script startup)
# For paths like VECTOR_STORE_FOLDER which might be outside this module's dir,
# the creation responsibility might be better handled by the script using it.
# However, for local defaults, this is fine.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STAGING_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(METADATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True) # This will create the CHROMA_DB_PATH if it's default
os.makedirs(LOG_FOLDER, exist_ok=True)

print(f"PARSER CHUNK_SIZE: {CHUNK_SIZE}") # For debugging .env loading
