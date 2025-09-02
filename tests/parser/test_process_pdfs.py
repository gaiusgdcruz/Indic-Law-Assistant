import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from parser.process_pdfs import (
    sanitize_filename,
    clean_text,
    extract_text_from_pdf,
    store_chunks_in_chroma,
    init_worker,
)
# We need to import config to mock values if necessary
from parser import config

class TestParserUtils(unittest.TestCase):

    def test_sanitize_filename(self):
        self.assertEqual(sanitize_filename("test/file?name.pdf"), "file_name.pdf")
        self.assertEqual(sanitize_filename("  a<b>c:d*e|f\\.pdf  "), "a_b_c_d_e_f_.pdf")
        self.assertEqual(sanitize_filename(""), "invalid_filename")
        self.assertEqual(sanitize_filename(".pdf"), "invalid_filename")
        self.assertEqual(sanitize_filename("...pdf"), "invalid_filename")
        self.assertEqual(sanitize_filename("  leading_trailing_spaces  .pdf"), "leading_trailing_spaces.pdf")

    def test_clean_text(self):
        self.assertEqual(clean_text("  hello   world  "), "hello world")
        self.assertEqual(clean_text("some-\nword"), "someword")
        self.assertEqual(clean_text("line1\nline2"), "line1 line2")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("Hyphen-\\-nation"), "Hyphen-nation")

class TestPDFProcessing(unittest.TestCase):

    def setUp(self):
        """Set up a mock logger for the worker functions."""
        self.mock_logger = MagicMock()

    @patch('parser.process_pdfs.fitz')
    def test_extract_text_from_pdf(self, mock_fitz):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is the text from page 1."

        mock_doc = MagicMock()
        mock_doc.metadata = {'title': 'Test PDF'}
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_doc.close = MagicMock()

        mock_fitz.open.return_value = mock_doc

        pdf_path = "/fake/path/to/test.pdf"
        chunks, metadata, total_pages = extract_text_from_pdf(pdf_path, self.mock_logger)

        mock_fitz.open.assert_called_once_with(pdf_path)
        self.assertEqual(total_pages, 1)
        self.assertEqual(metadata['title'], 'Test PDF')

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['text'], "This is the text from page 1.")
        self.assertEqual(chunks[0]['metadata']['source'], "test.pdf")
        self.assertEqual(chunks[0]['metadata']['page_number'], 1)
        self.assertEqual(chunks[0]['metadata']['title'], 'Test PDF')
        mock_doc.close.assert_called_once()

    @patch('langchain_chroma.Chroma')
    def test_store_chunks_in_chroma(self, mock_chroma_class):
        mock_collection = MagicMock()
        mock_chroma_class.return_value = mock_collection

        with patch('parser.process_pdfs.worker_embedding_model', MagicMock()) as mock_embed_model:
            mock_embed_model.embed_query.return_value = [0.1, 0.2, 0.3]

            chunks = [
                {"text": "chunk 1", "metadata": {"source": "test.pdf", "page_number": 1}},
                {"text": "chunk 2", "metadata": {"source": "test.pdf", "page_number": 2}}
            ]

            with patch.object(config, 'CHROMA_BATCH_SIZE', 50):
                success = store_chunks_in_chroma(chunks, self.mock_logger)

            self.assertTrue(success)
            mock_chroma_class.assert_called_once()
            self.assertEqual(mock_collection.add_documents.call_count, 1)
            added_docs = mock_collection.add_documents.call_args[1]['documents']
            batch_size_arg = mock_collection.add_documents.call_args[1]['batch_size']
            self.assertEqual(batch_size_arg, 50)
            self.assertEqual(len(added_docs), 2)
            self.assertEqual(added_docs[0].page_content, "chunk 1")

    @patch('langchain_chroma.Chroma')
    def test_store_chunks_in_chroma_failure(self, mock_chroma_class):
        mock_collection = MagicMock()
        mock_collection.add_documents.side_effect = Exception("DB connection failed")
        mock_chroma_class.return_value = mock_collection

        with patch('parser.process_pdfs.worker_embedding_model', MagicMock()):
            chunks = [{"text": "chunk 1", "metadata": {}}]
            success = store_chunks_in_chroma(chunks, self.mock_logger)
            self.assertFalse(success)
            self.mock_logger.error.assert_called_with("Failed to store chunks in ChromaDB: DB connection failed", exc_info=True)

class TestWorkerInitialization(unittest.TestCase):

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    def test_init_worker_success(self, mock_huggingface_embeddings):
        mock_model_instance = MagicMock()
        mock_model_instance.embed_query.return_value = [0.1, 0.2]
        mock_huggingface_embeddings.return_value = mock_model_instance

        init_worker("test-model", "cpu")

        from parser.process_pdfs import worker_embedding_model
        self.assertIsNotNone(worker_embedding_model)
        mock_huggingface_embeddings.assert_called_once_with(
            model_name="test-model",
            model_kwargs={'device': 'cpu'}
        )
        mock_model_instance.embed_query.assert_called_once_with("test")

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings', side_effect=Exception("Model not found"))
    def test_init_worker_failure(self, mock_huggingface_embeddings):
        init_worker("bad-model", "cpu")

        from parser.process_pdfs import worker_embedding_model
        self.assertIsNone(worker_embedding_model)


if __name__ == '__main__':
    unittest.main()
