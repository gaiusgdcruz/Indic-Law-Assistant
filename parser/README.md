# Parser Module

This module is responsible for processing PDF documents, extracting their text content, generating embeddings, and indexing them into a ChromaDB vector store. This forms the core of the knowledge base for the RAG (Retrieval Augmented Generation) system.

## Functionality

-   **PDF Processing**: Scans a designated `uploads` folder for new or modified PDF files.
-   **Text Extraction**: Uses PyMuPDF (`fitz`) to extract text content from each page of the PDF.
-   **Text Cleaning**: Applies basic cleaning to the extracted text, such as removing excessive whitespace and normalizing line breaks.
-   **Text Chunking**: Splits the cleaned text into smaller, manageable chunks using `RecursiveCharacterTextSplitter` from LangChain, based on configured chunk size and overlap.
-   **Embedding Generation**: Generates vector embeddings for each text chunk using a HuggingFace sentence transformer model (`paraphrase-multilingual-MiniLM-L12-v2` by default).
-   **Vector Storage**: Stores the text chunks and their corresponding embeddings in a ChromaDB collection. Metadata associated with each chunk (e.g., source filename, page number, title) is also stored.
-   **Metadata Management**: Saves detailed metadata for each processed PDF (including PDF's own metadata and processing details) into a JSON file in the `metadata` folder.
-   **Processing Tracking**: Maintains a `tracking.json` file to keep a record of processed PDFs and their last modification times. This prevents reprocessing of already indexed and unchanged files.
-   **Multiprocessing**: Leverages multiprocessing to process PDFs, with each PDF handled by a separate worker process to improve efficiency (currently configured for a pool size of 1, meaning sequential processing by a worker that restarts for each PDF).
-   **Folder Management**: Organizes files into `uploads`, `processed_pdfs`, `metadata`, `vector_store`, and `logs` directories.

## Key Files

-   **`process_pdfs.py`**: The main script that orchestrates the entire parsing and embedding pipeline. It handles:
    -   Scanning for PDF files in the `uploads` directory.
    -   Checking `tracking.json` to identify new or modified files.
    -   Managing a multiprocessing pool for PDF processing.
    -   Calling worker functions to process individual PDFs.
    -   Updating `tracking.json` after processing.
    -   Logging the progress and any errors.
-   **`config.py`**: Contains all configuration parameters for the parser module, such as:
    -   Folder paths (uploads, processed, metadata, vector store, logs).
    -   `TRACKING_FILE` path.
    -   Chunking parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
    -   Embedding model name (`EMBEDDING_MODEL`) and device (`EMBEDDING_DEVICE`).
    -   ChromaDB collection name (`COLLECTION_NAME`).
-   **`tracking.json`**: A JSON file that stores information about the PDFs that have been processed. For each file, it typically stores the path to the processed PDF, the original path, the timestamp of the source file when it was last processed, and the time of processing. This helps in skipping files that haven't changed since their last processing.

## Setup

1.  **Dependencies**: The necessary Python packages are listed in the main `requirements.txt` file in the project root. Key dependencies include `PyMuPDF`, `langchain`, `langchain-community`, `sentence-transformers`, `chromadb`, `huggingface_hub`.
2.  **Configuration**:
    -   Ensure that the folder paths in `parser/config.py` are correctly set up relative to the project structure or use environment variables if the script is adapted to use them.
    -   The default embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) will be downloaded by HuggingFace's `sentence-transformers` library on first use if not already cached.
3.  **Input Files**: Place PDF files that need to be processed into the `parser/uploads/` directory.

## Execution

To run the PDF processing pipeline, execute the `process_pdfs.py` script from the project's root directory or directly if Python path considerations are handled:

```bash
python parser/process_pdfs.py
```

The script will:
1.  Log its activities to a date-stamped log file in the `parser/logs/` directory and to the console.
2.  Check the `parser/uploads/` folder for PDF files.
3.  Compare these files against `parser/tracking.json` to find new or modified PDFs.
4.  For each new/modified PDF:
    -   Extract text and metadata.
    -   Chunk the text.
    -   Generate embeddings for chunks.
    -   Store chunks and embeddings in the ChromaDB instance located at `parser/vector_store/`.
    -   Save a metadata JSON file for the PDF in `parser/metadata/`.
    -   Move the processed PDF from `parser/uploads/` to `parser/processed_pdfs/`.
    -   Update `parser/tracking.json`.

## Output

-   **Vector Store**: The `parser/vector_store/` directory will contain the ChromaDB database with indexed document chunks.
-   **Processed PDFs**: Successfully processed PDFs are moved to `parser/processed_pdfs/`.
-   **Metadata**: JSON files detailing each processed PDF are stored in `parser/metadata/`.
-   **Logs**: Log files are created in `parser/logs/`.
-   **Tracking File**: `parser/tracking.json` is updated with the status of processed files.

## Important Notes

-   The script is designed to be run repeatedly. It will only process new or updated PDFs based on the `tracking.json` file and file modification times.
-   The embedding model specified in `parser/config.py` should be consistent with the one used by the `api` module for querying, to ensure meaningful similarity searches.
-   The ChromaDB vector store created by this module is directly used by the `api` module for document retrieval.
```
