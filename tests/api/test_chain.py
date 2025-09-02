import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

# Add project root for consistent imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the module to be tested
from api import chain
from langchain_core.messages import AIMessage

class TestRefactoredChain(unittest.TestCase):

    def setUp(self):
        """Reset the pipeline before each test to ensure mocks are reapplied."""
        chain._rag_pipeline = None

    @patch('langchain_ollama.ChatOllama')
    @patch('api.chain._initialize_vector_store')
    @patch('api.chain.get_reranker')
    @patch('api.chain.get_relevant_documents', new_callable=AsyncMock)
    def test_build_rag_chain_branching(self, mock_get_docs, mock_get_reranker, mock_init_vs, MockChatOllama):
        """
        Tests the branching logic by mocking all external dependencies.
        """
        async def run_test():
            # --- Setup Mocks ---
            mock_init_vs.return_value = MagicMock()
            mock_get_reranker.return_value = MagicMock()
            mock_get_docs.return_value = []

            mock_llm_instance = MagicMock()
            MockChatOllama.return_value = mock_llm_instance

            # --- Test Legal Branch ---
            mock_llm_instance.ainvoke.side_effect = [
                AIMessage(content="LEGAL"),
                AIMessage(content="rewritten legal query"),
                AIMessage(content="final legal answer")
            ]

            # get_rag_pipeline will call build_rag_chain, which will use our mocks
            test_pipeline = chain.get_rag_pipeline()
            # We need to test the core pipeline, not the caching wrapper for this test
            result = await test_pipeline.runnable.ainvoke({"query": "legal stuff"})

            mock_get_docs.assert_called_once()
            self.assertIn("final legal answer", result['answer'])

            # --- Test General Branch ---
            mock_get_docs.reset_mock()
            chain._rag_pipeline = None # Force rebuild

            mock_llm_instance.ainvoke.side_effect = [
                AIMessage(content="GENERAL"),
                AIMessage(content="final general answer")
            ]

            test_pipeline_2 = chain.get_rag_pipeline()
            result = await test_pipeline_2.runnable.ainvoke({"query": "general stuff"})

            mock_get_docs.assert_not_called()
            self.assertIn("final general answer", result['answer'])

        asyncio.run(run_test())

    @patch('api.chain.get_semantic_cache')
    def test_caching_runnable(self, mock_get_cache):
        """
        Tests the CachingRunnable with both hit and miss scenarios.
        """
        async def run_test():
            # --- Cache Hit Scenario ---
            mock_cache_hit = MagicMock()
            future_hit = asyncio.Future()
            future_hit.set_result([{'response': 'cached answer'}])
            mock_get_cache.return_value = mock_cache_hit

            mock_runnable_hit = MagicMock()
            mock_runnable_hit.ainvoke = AsyncMock()

            caching_wrapper_hit = chain.CachingRunnable(runnable=mock_runnable_hit)

            with patch('api.chain.asyncio.get_running_loop') as mock_loop_hit:
                mock_loop_hit.return_value.run_in_executor.return_value = future_hit
                result_hit = await caching_wrapper_hit.ainvoke({"query": "some query"})

            mock_loop_hit.return_value.run_in_executor.assert_called_once()
            mock_runnable_hit.ainvoke.assert_not_called()
            self.assertEqual(result_hit, {"answer": "cached answer", "sources": "From Cache"})

            # --- Cache Miss Scenario ---
            mock_cache_miss = MagicMock()
            future_miss = asyncio.Future(); future_miss.set_result(None)
            future_store = asyncio.Future(); future_store.set_result(None)
            mock_cache_miss.check.return_value = future_miss
            mock_get_cache.return_value = mock_cache_miss

            mock_runnable_miss = MagicMock()
            mock_runnable_miss.ainvoke = AsyncMock(return_value={"answer": "fresh answer"})

            caching_wrapper_miss = chain.CachingRunnable(runnable=mock_runnable_miss)

            with patch('api.chain.asyncio.get_running_loop') as mock_loop_miss:
                mock_loop_miss.return_value.run_in_executor.side_effect = [future_miss, future_store]
                result_miss = await caching_wrapper_miss.ainvoke({"query": "new query"})

            self.assertEqual(mock_loop_miss.return_value.run_in_executor.call_count, 2)
            mock_runnable_miss.ainvoke.assert_called_once()
            self.assertEqual(result_miss, {"answer": "fresh answer"})

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
