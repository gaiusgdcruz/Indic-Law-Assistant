import os

# Folder Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get base dir of config file
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads") # Path relative to config file
STAGING_FOLDER = os.path.join(BASE_DIR, "staging_pdfs") # New staging folder
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed_pdfs")
METADATA_FOLDER = os.path.join(BASE_DIR, "metadata")
VECTOR_STORE_FOLDER = os.path.join(BASE_DIR, "vector_store") 
LOG_FOLDER = os.path.join(BASE_DIR, "logs")

# File Configurations
TRACKING_FILE = os.path.join(BASE_DIR, "tracking.json")

# Processing Configurations
CHUNK_SIZE = 500 
CHUNK_OVERLAP = 100
PDF_BATCH_SIZE = 10 # Number of PDFs to process in each batch

# Embedding Model Configuration
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = 'cpu'

# ChromaDB Collection Name
COLLECTION_NAME = "law_docs_v1"

# Ensure directories exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, VECTOR_STORE_FOLDER, METADATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)
