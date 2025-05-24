import os
import json
import fitz  # PyMuPDF
import logging
import re
import sys
from datetime import datetime
# DO NOT load models globally anymore
import glob
import shutil # Import shutil for file operations
import time # Import time for delays
import gc # Import garbage collector
import multiprocessing # Import multiprocessing
from typing import List, Dict, Any, Tuple

# Add parent directory to sys.path to allow relative import of config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Assuming config.py is at the same level as the parser directory or accessible
# If config.py is inside parser/, use: from . import config
from parser import config # Use relative import if config.py is in the same dir
from core.logging_utils import get_logger # Import the new central logger


# =========================
# Logger Setup
# =========================
# Remove old logger setup. The new logger will handle file logging if APP_LOG_FILE_PATH is set.
# Specific parser logs (like the old parse_embed_pipeline_{today}.log) are not directly supported
# by the new central logger by default, but all logs will go to APP_LOG_FILE_PATH and console.
# If a separate log file for parser is still desired, it would need custom handling
# outside the scope of get_logger, or get_logger could be extended.

logger = get_logger(__name__) # Use the new central logger


# =========================
# Utility Functions (can remain global)
# =========================
def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    base_name = os.path.basename(filename)
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', base_name)
    sanitized = sanitized.strip('. ')
    if not sanitized:
        return "invalid_filename"
    return sanitized

def load_tracking_info() -> Dict[str, Any]:
    """Loads the tracking information from the JSON file, ensuring structure."""
    default_structure = {"processed_files": {}}
    tracking_file = config.TRACKING_FILE
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r', encoding='utf-8') as f:
                # Handle empty file case explicitly before JSON decoding
                if os.fstat(f.fileno()).st_size == 0:
                    logger.warning(f"Tracking file {tracking_file} is empty. Initializing fresh.")
                    return default_structure

                data = json.load(f)
                # Validate the structure
                if isinstance(data, dict) and "processed_files" in data and isinstance(data["processed_files"], dict):
                    return data
                else:
                    logger.warning(f"Tracking file {tracking_file} has invalid structure. Initializing fresh.")
                    return default_structure

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {tracking_file}. Initializing fresh.")
            return default_structure
        except Exception as e:
            logger.error(f"Error loading tracking file {tracking_file}: {e}", exc_info=True)
            return default_structure

    # File does not exist
    return default_structure

def save_tracking_info(data: Dict[str, Any]):
    """Saves the tracking information to the JSON file."""
    tracking_file = config.TRACKING_FILE
    try:
        # Use a temporary file and rename for atomic write
        temp_file = tracking_file + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        os.replace(temp_file, tracking_file) # Atomic rename
    except Exception as e:
        logger.error(f"Error saving tracking file {tracking_file}: {e}", exc_info=True)
        # Clean up temp file if it exists on error
        if os.path.exists(temp_file):
             try: os.remove(temp_file)
             except Exception: pass

def clean_text(text: str) -> str:
    """Cleans the extracted text."""
    if not text: return ""
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple whitespace with single space
    text = re.sub(r'-\n', '', text) # Remove hyphens at line breaks
    text = re.sub(r'\n', ' ', text) # Replace newline characters with spaces
    # Add more specific cleaning rules if needed
    return text


# =========================
# Worker Process Function (replaces process_pdf)
# =========================
def worker_process_single_pdf(pdf_path_tuple: Tuple[str, str]) -> Tuple[str, bool]:
    """
    Function executed by each worker process for a single PDF.
    Loads models, processes PDF, adds to Chroma, moves file, saves metadata.
    Returns a tuple: (processed_pdf_path, success_boolean).
    """
    pdf_path, processed_pdf_path = pdf_path_tuple # Unpack input tuple

    # --- Need to import everything used within this function ---
    from langchain_community.embeddings import HuggingFaceEmbeddings # Use community if installed
    # from langchain_huggingface import HuggingFaceEmbeddings # Or this if newer package used
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter # <-- IMPORT THIS
    import fitz
    import gc
    import json
    from datetime import datetime
    import os
    import shutil
    # logging is already imported via core.logging_utils
    # from core.logging_utils import get_logger # Already imported at top level

    # Get logger instance for the worker using the new utility
    # The level will be determined by LOG_LEVEL env var or default in get_logger
    worker_logger = get_logger(multiprocessing.current_process().name)
    process_name = multiprocessing.current_process().name # Still useful for log messages if desired
    pdf_filename = os.path.basename(pdf_path)
    sanitized_base_name = sanitize_filename(pdf_filename.replace('.pdf', ''))
    metadata_path = os.path.join(config.METADATA_FOLDER, f"{sanitized_base_name}_metadata.json")
    chroma_batch_size = 1 # Keep batch size 1 for minimum memory per add

    worker_logger.info(f"Worker started for PDF: {pdf_filename}")

    # --- Load Models INSIDE Worker ---
    embedding_model_worker = None
    collection_worker = None
    try:
        worker_logger.info(f"Loading embedding model ({config.EMBEDDING_MODEL})...")
        embedding_model_worker = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.EMBEDDING_DEVICE},
        )
        _ = embedding_model_worker.embed_query("test")
        worker_logger.info(f"Embedding model loaded.")

        worker_logger.info(f"Connecting to ChromaDB at {config.VECTOR_STORE_FOLDER}...")
        collection_worker = Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.VECTOR_STORE_FOLDER,
            embedding_function=embedding_model_worker
        )
        # Optionally verify connection/collection count
        # count = collection_worker._collection.count()
        # worker_logger.info(f"Connected to ChromaDB. Collection '{config.COLLECTION_NAME}' count: {count}")
        worker_logger.info(f"Connected to ChromaDB.")

    except Exception as model_load_err:
        worker_logger.error(f"Failed to load models or connect to DB: {model_load_err}", exc_info=True)
        return (processed_pdf_path, False) # Indicate failure for this PDF

    # --- PDF Processing Logic ---
    worker_logger.info(f"Processing '{pdf_filename}' page-by-page...")
    documents_batch = []
    total_chunks = 0
    added_to_chroma_flag = False
    pdf_doc_metadata = {}
    total_pages = 0
    doc_fitz = None

    try:
        doc_fitz = fitz.open(pdf_path)
        pdf_doc_metadata = doc_fitz.metadata
        total_pages = len(doc_fitz)

        # --- Instantiate Text Splitter ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP, # Use char overlap
            length_function=len,
            is_separator_regex=False,
             # Common separators for legal text (adjust if needed)
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        # -------------------------------

        for page_num in range(total_pages):
            page_idx = page_num + 1
            # worker_logger.debug(f"Processing page {page_idx}/{total_pages}...")
            page = doc_fitz.load_page(page_num)
            page_text = page.get_text("text", sort=True) # Try sorting text blocks

            if not page_text or not page_text.strip():
                # worker_logger.debug(f"Skipping empty page {page_idx}")
                page = None; gc.collect(); continue

            cleaned_text = clean_text(page_text); page_text = None
            if not cleaned_text:
                # worker_logger.debug(f"Skipping page {page_idx} after cleaning yielded empty text")
                cleaned_text = None; page = None; gc.collect(); continue

            # --- Use RecursiveCharacterTextSplitter ---
            page_chunks_texts = text_splitter.split_text(cleaned_text)
            # ------------------------------------------

            cleaned_text = None # Release reference
            page_chunk_count = len(page_chunks_texts); total_chunks += page_chunk_count
            # worker_logger.debug(f"Page {page_idx}: generated {page_chunk_count} chunks.")

            for i, chunk_text in enumerate(page_chunks_texts): # Iterate through text chunks
                chunk_index_overall = total_chunks - page_chunk_count + i
                metadata = {"source": pdf_filename, "page_number": page_idx, "chunk_index_on_page": i}
                # Add title safely
                if pdf_doc_metadata.get('title'):
                    title_val = pdf_doc_metadata.get('title', '')
                    try: metadata['title'] = (title_val.decode('utf-8', errors='ignore') if isinstance(title_val, bytes) else str(title_val))[:200]
                    except Exception: pass # Ignore title errors

                # Create Document object directly from text chunk
                doc_obj = Document(page_content=chunk_text, metadata=metadata)
                documents_batch.append(doc_obj); chunk_text = None # Release reference

                if len(documents_batch) >= chroma_batch_size:
                    try:
                        # worker_logger.debug(f"Adding batch size {len(documents_batch)} (Chunk ~{chunk_index_overall})...")
                        collection_worker.add_documents(documents=documents_batch)
                        added_to_chroma_flag = True
                        # worker_logger.debug(f"Added batch (Chunk ~{chunk_index_overall}).")
                        documents_batch = []; gc.collect()
                    except Exception as e:
                        worker_logger.error(f"Failed Chroma batch add page {page_idx} chunk {i}: {e}", exc_info=True)
                        documents_batch = []; gc.collect() # Clear potentially problematic batch
                        # Decide if we should stop processing this PDF entirely on batch failure
                        # return (processed_pdf_path, False)

            page_chunks_texts = None; page = None; gc.collect() # Clean up after page loop

        doc_fitz.close(); doc_fitz = None; gc.collect()

        # Add final batch if any remain
        if documents_batch:
            try:
                # worker_logger.debug(f"Adding final batch {len(documents_batch)}...")
                collection_worker.add_documents(documents=documents_batch)
                added_to_chroma_flag = True
                # worker_logger.debug(f"Added final batch.")
                documents_batch = []; gc.collect()
            except Exception as e:
                worker_logger.error(f"Failed final Chroma batch add: {e}", exc_info=True)
                documents_batch = []; gc.collect() # Clear anyway

        # --- Save metadata (worker does this) ---
        os.makedirs(config.METADATA_FOLDER, exist_ok=True)
        serializable_pdf_metadata = {}
        for k, v in pdf_doc_metadata.items():
            try: json.dumps({k: v}); serializable_pdf_metadata[k] = v
            except TypeError: serializable_pdf_metadata[k] = v.isoformat() if isinstance(v, datetime) else v.decode('utf-8', errors='ignore') if isinstance(v, bytes) else str(v)

        file_metadata = {"source_filename": pdf_filename, "processed_pdf_path": processed_pdf_path, "extracted_on": datetime.now().isoformat(), "total_pages": total_pages, "pdf_metadata": serializable_pdf_metadata, "total_chunk_count": total_chunks, "added_to_chroma": added_to_chroma_flag}
        with open(metadata_path, 'w', encoding='utf-8') as f: json.dump(file_metadata, f, ensure_ascii=False, indent=4)
        worker_logger.info(f"Metadata saved to '{metadata_path}'")

        # --- Move processed PDF (worker does this) ---
        os.makedirs(config.PROCESSED_FOLDER, exist_ok=True)
        try:
            if os.path.exists(pdf_path):
                shutil.move(pdf_path, processed_pdf_path)
                worker_logger.info(f"Moved processed PDF to '{processed_pdf_path}'")
            else: worker_logger.warning(f"Source PDF '{pdf_path}' not found for moving.")
        except Exception as e: worker_logger.error(f"Failed to move PDF from '{pdf_path}': {e}")

        # Clean up before returning success
        pdf_doc_metadata = None; serializable_pdf_metadata = None; file_metadata = None; embedding_model_worker = None; collection_worker = None; text_splitter = None; gc.collect()
        worker_logger.info(f"Worker finished successfully for {pdf_filename}.")
        return (processed_pdf_path, True) # Return processed path and success

    except Exception as e:
        worker_logger.error(f"Unhandled error processing '{pdf_filename}': {e}", exc_info=True)
        if doc_fitz and not doc_fitz.is_closed:
            try: doc_fitz.close()
            except Exception: pass
        # Clean up models on error too
        embedding_model_worker = None; collection_worker = None; text_splitter = None; gc.collect()
        return (processed_pdf_path, False) # Return processed path and failure


# =========================
# Entry Point / Main Process
# =========================
if __name__ == "__main__":
    # Logger is already set up at the module level using get_logger(__name__)
    # No need to call setup_logger() anymore.
    logger.info("Starting PDF processing script using multiprocessing (Pool Size 1).")

    # Ensure required folders exist (run by main process)
    # Note: LOG_FOLDER from parser.config is for the old parser-specific log.
    # The new central logger uses APP_LOG_FILE_PATH from .env for its file output.
    # We still create parser-specific folders like UPLOAD_FOLDER, etc.
    required_folders = [config.UPLOAD_FOLDER, config.PROCESSED_FOLDER, config.METADATA_FOLDER, config.VECTOR_STORE_FOLDER, config.LOG_FOLDER]
    for folder in required_folders: os.makedirs(folder, exist_ok=True)

    # --- Find all PDFs in Upload Folder ---
    try:
        all_upload_files = glob.glob(os.path.join(config.UPLOAD_FOLDER, "*.pdf"))
        all_upload_files.extend(glob.glob(os.path.join(config.UPLOAD_FOLDER, "*.PDF")))
        all_upload_files = [f for f in all_upload_files if os.path.isfile(f)]
        logger.info(f"Found {len(all_upload_files)} PDF files in upload folder.")

        if not all_upload_files:
            logger.info("No PDFs found to process.")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Error finding files in upload folder: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Tracking Info & Filter Files (Main Process) ---
    tracking_info = load_tracking_info()
    files_to_process_tuples = [] # List of tuples (source_path, destination_path)
    skipped_count = 0

    for pdf_path in all_upload_files:
         processed_pdf_path = os.path.join(config.PROCESSED_FOLDER, os.path.basename(pdf_path))
         try:
             # Check modification time of the source file
             file_last_modified = os.path.getmtime(pdf_path)
         except OSError as e:
             logger.warning(f"Could not get modification time for source file: {pdf_path}. Skipping. Error: {e}")
             skipped_count += 1
             continue

         # Check tracking info using the *destination* path as the key
         if processed_pdf_path in tracking_info["processed_files"] and \
            tracking_info["processed_files"][processed_pdf_path].get("timestamp", 0) >= file_last_modified:
             # logger.debug(f"Skipping '{os.path.basename(pdf_path)}', already processed and unchanged (based on tracking file).")
             skipped_count += 1
         else:
             files_to_process_tuples.append((pdf_path, processed_pdf_path))

    logger.info(f"Skipped {skipped_count} files (already processed or inaccessible).")
    logger.info(f"Found {len(files_to_process_tuples)} new or modified files to process.")
    if not files_to_process_tuples:
         logger.info("No new files require processing.")
         sys.exit(0)

    # --- Process Files using Pool ---
    processed_count = 0
    failed_count = 0

    # Create a pool of 1 worker process
    # The worker will load models, process one PDF, and exit.
    logger.info("Starting processing pool (Size 1)...")
    # Use try-finally to ensure pool cleanup
    pool = None
    try:
        # Set maxtasksperchild=1 to force process restart after each task
        pool = multiprocessing.Pool(processes=1, maxtasksperchild=1)
        # map applies the function to each item in the iterable sequentially (due to pool size 1)
        # It returns results in the order of the input iterable
        results = pool.map(worker_process_single_pdf, files_to_process_tuples)
    except Exception as pool_err:
         logger.error(f"Error occurred during pool processing: {pool_err}", exc_info=True)
         # Results might be incomplete or empty here
         results = [] # Assume failure for all if pool crashes
    finally:
        if pool:
            pool.close() # Prevent new tasks
            pool.join() # Wait for worker to finish
            logger.info("Processing pool closed.")


    # --- Update Tracking Info and Summarize (Main Process) ---
    # Reload tracking info as it might have changed if multiple runs occurred (though not with Pool 1)
    # Also ensures we save the latest state even if some workers failed midway
    tracking_info = load_tracking_info()
    successful_processed_paths = set() # Keep track of successful paths for this run

    for i, result_tuple in enumerate(results):
         source_path, _ = files_to_process_tuples[i] # Get original source path
         processed_path, success = result_tuple

         if success:
             processed_count += 1
             successful_processed_paths.add(processed_path)
             # Update tracking info for successfully processed file
             try:
                 # We need modification time of the *source* for comparison,
                 # but we store the processed time stamp.
                 file_last_modified_source = os.path.getmtime(source_path)
             except OSError:
                 # If source is gone (moved by worker), use current time approx.
                 file_last_modified_source = time.time()
                 logger.warning(f"Could not get modification time for original source: {source_path}")

             tracking_info["processed_files"][processed_path] = {
                 "original_path": source_path,
                 "timestamp": file_last_modified_source, # Store source modification time
                 "processed_on": datetime.now().isoformat()
             }
         else:
             failed_count += 1
             logger.error(f"Processing failed for source: {os.path.basename(source_path)}")
             # Optionally move the source file from UPLOAD to a FAILED folder here
             failed_dir = os.path.join(config.BASE_DIR, "failed_upload_pdfs")
             os.makedirs(failed_dir, exist_ok=True)
             try:
                fail_dst = os.path.join(failed_dir, os.path.basename(source_path))
                if os.path.exists(source_path) and not os.path.exists(fail_dst):
                    shutil.move(source_path, fail_dst)
                    logger.warning(f"Moved failed source PDF '{os.path.basename(source_path)}' to {failed_dir}")
             except Exception as move_err:
                 logger.error(f"Could not move failed source PDF '{os.path.basename(source_path)}': {move_err}")

    # Save final tracking info
    save_tracking_info(tracking_info)
    logger.info("Final tracking information saved.")

    logger.info("============================================")
    logger.info("Overall PDF Processing Finished.")
    logger.info(f"Total Attempted in this run: {len(files_to_process_tuples)}")
    logger.info(f"Total Successfully Processed: {processed_count}")
    logger.info(f"Total Failed to Process: {failed_count}")
    logger.info("============================================")

