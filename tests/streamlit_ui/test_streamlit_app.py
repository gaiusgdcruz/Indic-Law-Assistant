import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

# Add project root to path for consistent imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from streamlit_ui import streamlit_app

class TestStreamlitAuth(unittest.TestCase):

    def setUp(self):
        """Set up a mock for streamlit's session_state."""
        self.session_state_patch = patch('streamlit_ui.streamlit_app.st.session_state', MagicMock())
        self.mock_session_state = self.session_state_patch.start()

        # Set default values for the mock session state
        self.mock_session_state.api_base_url = "http://fakeapi:8000"
        self.mock_session_state.token = None
        self.mock_session_state.refresh_token = None
        self.mock_session_state.is_authenticated = False

    def tearDown(self):
        """Stop the patch."""
        self.session_state_patch.stop()

    @patch('streamlit_ui.streamlit_app.httpx.AsyncClient')
    def test_login_success(self, MockAsyncClient):
        """Test successful login."""
        # --- Setup Mock ---
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "fake_access_token",
            "refresh_token": "fake_refresh_token"
        }

        mock_client_instance = MagicMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)

        # The AsyncClient context manager should return our instance
        mock_cm = MagicMock()
        mock_cm.__aenter__.return_value = mock_client_instance
        MockAsyncClient.return_value = mock_cm

        # --- Run Test ---
        result = asyncio.run(streamlit_app.login("testuser", "testpass"))

        # --- Assertions ---
        self.assertTrue(result)
        self.assertTrue(self.mock_session_state.is_authenticated)
        self.assertEqual(self.mock_session_state.token, "fake_access_token")
        self.assertEqual(self.mock_session_state.refresh_token, "fake_refresh_token")
        mock_client_instance.post.assert_called_once_with(
            "http://fakeapi:8000/token",
            json={"username": "testuser", "password": "testpass"}
        )

    @patch('streamlit_ui.streamlit_app.httpx.AsyncClient')
    def test_login_failure(self, MockAsyncClient):
        """Test failed login."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        mock_client_instance = MagicMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)

        mock_cm = MagicMock()
        mock_cm.__aenter__.return_value = mock_client_instance
        MockAsyncClient.return_value = mock_cm

        result = asyncio.run(streamlit_app.login("wronguser", "wrongpass"))

        self.assertFalse(result)
        self.assertFalse(self.mock_session_state.is_authenticated)

    @patch('streamlit_ui.streamlit_app.httpx.AsyncClient')
    def test_refresh_token_success(self, MockAsyncClient):
        """Test successful token refresh."""
        self.mock_session_state.refresh_token = "valid_refresh_token"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token"
        }

        mock_client_instance = MagicMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)

        mock_cm = MagicMock()
        mock_cm.__aenter__.return_value = mock_client_instance
        MockAsyncClient.return_value = mock_cm

        result = asyncio.run(streamlit_app.refresh_token())

        self.assertTrue(result)
        self.assertEqual(self.mock_session_state.token, "new_access_token")
        self.assertEqual(self.mock_session_state.refresh_token, "new_refresh_token")
        mock_client_instance.post.assert_called_once_with(
            "http://fakeapi:8000/token/refresh",
            json={"refresh_token": "valid_refresh_token"}
        )

if __name__ == '__main__':
    unittest.main()
