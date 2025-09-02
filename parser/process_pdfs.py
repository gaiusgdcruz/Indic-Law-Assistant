import os
import json
import fitz  # PyMuPDF
import logging
import re
import sys
from datetime import datetime
import glob
import shutil
import time
import gc
import multiprocessing
from typing import List, Dict, Any, Tuple

# Add parent directory to sys.path to allow relative import of config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser import config
from core.logging_utils import get_logger

logger = get_logger(__name__)

# Global variable to hold the model for a worker process
worker_embedding_model = None

def init_worker(model_name: str, device: str):
    """
    Initializer for each worker process. Loads the embedding model.
    """
    global worker_embedding_model
    from langchain_community.embeddings import HuggingFaceEmbeddings

    process_name = multiprocessing.current_process().name
    logger.info(f"[{process_name}] Initializing worker and loading embedding model '{model_name}'...")
    try:
        worker_embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
        )
        # Perform a test embed to ensure the model is loaded correctly
        _ = worker_embedding_model.embed_query("test")
        logger.info(f"[{process_name}] Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"[{process_name}] Failed to load embedding model: {e}", exc_info=True)
        worker_embedding_model = None


def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    if not filename:
        return "invalid_filename"

    base_name = os.path.basename(filename).strip()
    if not base_name or base_name.strip('. ') == "":
        return "invalid_filename"

    sanitized = re.sub(r'[\\/*?:"<>|]', '_', base_name)

    name, ext = os.path.splitext(sanitized)
    name = name.strip('. ')
    if not name:
        return "invalid_filename"

    return name + ext

def load_tracking_info() -> Dict[str, Any]:
    """Loads the tracking information from the JSON file."""
    default_structure = {"processed_files": {}}
    tracking_file = config.TRACKING_FILE
    if not os.path.exists(tracking_file):
        return default_structure

    try:
        with open(tracking_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                logger.warning(f"Tracking file {tracking_file} is empty. Initializing fresh.")
                return default_structure
            data = json.loads(content)
            if isinstance(data, dict) and "processed_files" in data and isinstance(data["processed_files"], dict):
                return data
            else:
                logger.warning(f"Tracking file {tracking_file} has invalid structure. Re-initializing.")
                return default_structure
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {tracking_file}. Re-initializing.")
        return default_structure
    except Exception as e:
        logger.error(f"Error loading tracking file {tracking_file}: {e}", exc_info=True)
        return default_structure

def save_tracking_info(data: Dict[str, Any]):
    """Saves the tracking information to the JSON file atomically."""
    tracking_file = config.TRACKING_FILE
    temp_file = tracking_file + ".tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        os.replace(temp_file, tracking_file)
    except Exception as e:
        logger.error(f"Error saving tracking file {tracking_file}: {e}", exc_info=True)
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as rm_err:
                logger.error(f"Could not remove temporary tracking file {temp_file}: {rm_err}")

def clean_text(text: str) -> str:
    """Cleans the extracted text."""
    if not text:
        return ""
    # De-hyphenate words at line breaks
    text = re.sub(r'-\n', '', text)
    # Replace escaped backslash-hyphen with just a hyphen
    text = text.replace('\\-', '-')
    # Normalize all whitespace to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_path: str, worker_logger) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
    """Extracts text, chunks it, and gathers metadata from a PDF."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    chunks_with_metadata = []
    pdf_doc_metadata = {}
    total_pages = 0

    try:
        doc_fitz = fitz.open(pdf_path)
        pdf_doc_metadata = doc_fitz.metadata
        total_pages = len(doc_fitz)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )

        pdf_filename = os.path.basename(pdf_path)
        for page_num in range(total_pages):
            page = doc_fitz.load_page(page_num)
            page_text = page.get_text("text", sort=True)
            if not page_text or not page_text.strip():
                continue

            cleaned_text = clean_text(page_text)
            page_chunks_texts = text_splitter.split_text(cleaned_text)

            for i, chunk_text in enumerate(page_chunks_texts):
                metadata = {"source": pdf_filename, "page_number": page_num + 1, "chunk_index_on_page": i}
                if pdf_doc_metadata.get('title'):
                    title_val = pdf_doc_metadata.get('title', '')
                    try:
                        metadata['title'] = (title_val.decode('utf-8', 'ignore') if isinstance(title_val, bytes) else str(title_val))[:200]
                    except Exception: pass

                chunks_with_metadata.append({"text": chunk_text, "metadata": metadata})

        doc_fitz.close()
    except Exception as e:
        worker_logger.error(f"Failed to extract text from {pdf_path}: {e}", exc_info=True)
        return [], {}, 0

    return chunks_with_metadata, pdf_doc_metadata, total_pages


def store_chunks_in_chroma(chunks: List[Dict[str, Any]], worker_logger) -> bool:
    """Stores document chunks in ChromaDB."""
    global worker_embedding_model
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    if not worker_embedding_model:
        worker_logger.error("Embedding model not available in worker. Cannot store chunks.")
        return False
    if not chunks:
        worker_logger.warning("No chunks provided to store.")
        return False

    try:
        collection_worker = Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.VECTOR_STORE_FOLDER,
            embedding_function=worker_embedding_model
        )

        documents_to_add = [Document(page_content=chunk["text"], metadata=chunk["metadata"]) for chunk in chunks]

        collection_worker.add_documents(documents=documents_to_add, batch_size=config.CHROMA_BATCH_SIZE)
        worker_logger.info(f"Successfully added {len(documents_to_add)} chunks to ChromaDB.")
        return True
    except Exception as e:
        worker_logger.error(f"Failed to store chunks in ChromaDB: {e}", exc_info=True)
        return False

def worker_process_single_pdf(pdf_path_tuple: Tuple[str, str]) -> Tuple[str, bool]:
    """
    Function executed by each worker process for a single PDF.
    """
    pdf_path, processed_pdf_path = pdf_path_tuple
    process_name = multiprocessing.current_process().name
    worker_logger = get_logger(process_name)
    pdf_filename = os.path.basename(pdf_path)

    worker_logger.info(f"Worker started for PDF: {pdf_filename}")

    if worker_embedding_model is None:
        worker_logger.error("Worker embedding model not initialized. Aborting task.")
        return (processed_pdf_path, False)

    # 1. Extract and chunk text
    chunks, pdf_metadata, total_pages = extract_text_from_pdf(pdf_path, worker_logger)
    if not chunks:
        return (processed_pdf_path, False)

    # 2. Store chunks in ChromaDB
    added_to_chroma = store_chunks_in_chroma(chunks, worker_logger)
    if not added_to_chroma:
        return (processed_pdf_path, False)

    # 3. Save metadata file
    sanitized_base_name = sanitize_filename(pdf_filename.replace('.pdf', ''))
    metadata_path = os.path.join(config.METADATA_FOLDER, f"{sanitized_base_name}_metadata.json")
    serializable_pdf_metadata = {k: (v.decode('utf-8', 'ignore') if isinstance(v, bytes) else str(v)) for k, v in pdf_metadata.items()}
    file_metadata = {
        "source_filename": pdf_filename,
        "processed_pdf_path": processed_pdf_path,
        "extracted_on": datetime.now().isoformat(),
        "total_pages": total_pages,
        "pdf_metadata": serializable_pdf_metadata,
        "total_chunk_count": len(chunks),
        "added_to_chroma": added_to_chroma
    }
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(file_metadata, f, ensure_ascii=False, indent=4)
        worker_logger.info(f"Metadata saved to '{metadata_path}'")
    except Exception as e:
        worker_logger.error(f"Failed to save metadata for {pdf_filename}: {e}")
        # Continue to move the file, but the overall process might be considered failed
        # depending on requirements. For now, we'll let it pass.

    # 4. Move processed file
    try:
        if os.path.exists(pdf_path):
            os.makedirs(os.path.dirname(processed_pdf_path), exist_ok=True)
            shutil.move(pdf_path, processed_pdf_path)
            worker_logger.info(f"Moved processed PDF to '{processed_pdf_path}'")
        else:
            worker_logger.warning(f"Source PDF '{pdf_path}' not found for moving.")
    except Exception as e:
        worker_logger.error(f"Failed to move PDF from '{pdf_path}': {e}")
        return (processed_pdf_path, False)

    gc.collect()
    worker_logger.info(f"Worker finished successfully for {pdf_filename}.")
    return (processed_pdf_path, True)

def main():
    """Main entry point for the script."""
    logger.info("Starting PDF processing script...")

    required_folders = [config.UPLOAD_FOLDER, config.PROCESSED_FOLDER, config.METADATA_FOLDER, config.VECTOR_STORE_FOLDER, config.LOG_FOLDER]
    for folder in required_folders:
        os.makedirs(folder, exist_ok=True)

    all_upload_files = glob.glob(os.path.join(config.UPLOAD_FOLDER, "*.pdf")) + glob.glob(os.path.join(config.UPLOAD_FOLDER, "*.PDF"))
    logger.info(f"Found {len(all_upload_files)} PDF files in upload folder.")
    if not all_upload_files:
        logger.info("No PDFs found to process.")
        return

    tracking_info = load_tracking_info()
    files_to_process_tuples = []
    skipped_count = 0

    for pdf_path in all_upload_files:
        processed_pdf_path = os.path.join(config.PROCESSED_FOLDER, os.path.basename(pdf_path))
        try:
            file_last_modified = os.path.getmtime(pdf_path)
        except OSError as e:
            logger.warning(f"Cannot get mod time for {pdf_path}, skipping. Error: {e}")
            skipped_count += 1
            continue

        if processed_pdf_path in tracking_info["processed_files"] and \
           tracking_info["processed_files"][processed_pdf_path].get("timestamp", 0) >= file_last_modified:
            skipped_count += 1
        else:
            files_to_process_tuples.append((pdf_path, processed_pdf_path))

    logger.info(f"Skipped {skipped_count} unchanged files.")
    if not files_to_process_tuples:
        logger.info("No new or modified files to process.")
        return

    logger.info(f"Found {len(files_to_process_tuples)} new or modified files to process.")

    # Use context manager for the pool
    try:
        # Use a number of processes based on CPU count, but not more than a reasonable limit or the number of files
        num_processes = min(config.PARSER_WORKER_COUNT, len(files_to_process_tuples))
        logger.info(f"Starting processing pool with {num_processes} workers...")

        with multiprocessing.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(config.EMBEDDING_MODEL, config.EMBEDDING_DEVICE)
        ) as pool:
            results = pool.map(worker_process_single_pdf, files_to_process_tuples)

    except Exception as pool_err:
        logger.error(f"Error during pool processing: {pool_err}", exc_info=True)
        return

    processed_count = 0
    failed_count = 0
    for i, (processed_path, success) in enumerate(results):
        source_path, _ = files_to_process_tuples[i]
        if success:
            processed_count += 1
            try:
                file_last_modified_source = os.path.getmtime(source_path) if os.path.exists(source_path) else time.time()
                tracking_info["processed_files"][processed_path] = {
                    "original_path": source_path,
                    "timestamp": file_last_modified_source,
                    "processed_on": datetime.now().isoformat()
                }
            except OSError:
                logger.warning(f"Could not get modification time for original source: {source_path}")
        else:
            failed_count += 1
            logger.error(f"Processing failed for source: {os.path.basename(source_path)}")
            # Move failed files to a separate directory
            failed_dir = os.path.join(config.BASE_DIR, "failed_uploads")
            os.makedirs(failed_dir, exist_ok=True)
            try:
                fail_dst = os.path.join(failed_dir, os.path.basename(source_path))
                if os.path.exists(source_path):
                    shutil.move(source_path, fail_dst)
                    logger.warning(f"Moved failed PDF to {fail_dst}")
            except Exception as move_err:
                logger.error(f"Could not move failed PDF '{os.path.basename(source_path)}': {move_err}")

    save_tracking_info(tracking_info)
    logger.info("Final tracking information saved.")

    logger.info("============================================")
    logger.info("PDF Processing Finished.")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed to process: {failed_count}")
    logger.info("============================================")

if __name__ == "__main__":
    main()
