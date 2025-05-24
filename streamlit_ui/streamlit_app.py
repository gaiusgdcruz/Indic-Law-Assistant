import streamlit as st
import httpx
import json
from datetime import datetime
import asyncio
from typing import Optional, Tuple
# Replace old logging setup with the new central one
from core.logging_utils import get_logger
import os
from dotenv import load_dotenv
import time
# hashlib is no longer used as password hashing is server-side
# import hashlib

# Load environment variables
# Ensure .env is loaded from project root if streamlit_app.py is in a subdirectory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv() # Load .env from current directory or default path if not found in root

# Configure logging using the new central utility
# The old file handler 'streamlit_app.log' will be replaced by the central APP_LOG_FILE_PATH
logger = get_logger(__name__)

# Update the title and description
st.set_page_config(
    page_title="Indic Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'token' not in st.session_state:
    st.session_state.token = None
if 'refresh_token' not in st.session_state:
    st.session_state.refresh_token = None
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'language' not in st.session_state:
    st.session_state.language = "en"
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")


async def login(username: str, password: str) -> bool:
    """Authenticate with the API using the raw password."""
    try:
        # Send the raw password directly
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{st.session_state.api_base_url}/token", # Use configured API base URL
                json={"username": username, "password": password} # Send raw password
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.token = data["access_token"]
                st.session_state.refresh_token = data.get("refresh_token")  # new refresh token support
                st.session_state.is_authenticated = True
                return True
            return False
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return False

async def refresh_token() -> bool:
    """Attempt to refresh the access token using the refresh token"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{st.session_state.api_base_url}/token/refresh", # Use configured API base URL
                json={"refresh_token": st.session_state.refresh_token}
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.token = data["access_token"]
                # Optionally update the refresh token as well
                st.session_state.refresh_token = data.get("refresh_token", st.session_state.refresh_token)
                logger.info("Token refreshed successfully.")
                return True
            else:
                logger.error(f"Token refresh failed with status: {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"Refresh token error: {str(e)}")
        return False

async def process_stream_async(query: str) -> Optional[str]:
    """Process the query with streaming response asynchronously"""
    if not query.strip():
        return None

    if not st.session_state.is_authenticated:
        st.error("Please log in to use the service")
        return None

    async with httpx.AsyncClient() as client:
        response_placeholder = st.empty()
        full_response = ""

        # Format chat history properly
        formatted_chat_history = []
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                user_msg = st.session_state.chat_history[i]
                assistant_msg = st.session_state.chat_history[i + 1]
                if isinstance(user_msg, dict) and isinstance(assistant_msg, dict):
                    formatted_chat_history.append({
                        "role": "user",
                        "content": user_msg.get("content", "")
                    })
                    formatted_chat_history.append({
                        "role": "assistant",
                        "content": assistant_msg.get("content", "")
                    })

        # Define a helper to send the request
        async def send_request():
            return await client.post(
                f"{st.session_state.api_base_url}/stream", # Use configured API base URL
                json={
                    "query": query,
                    "language": st.session_state.language,
                    "chat_history": formatted_chat_history
                },
                headers={"Authorization": f"Bearer {st.session_state.token}"},
                timeout=300.0
            )

        response = await send_request()
        if response.status_code == 401:
            st.error("Your login token has expired. Attempting token refresh...")
            if await refresh_token():
                logger.info("Token refreshed. Retrying query...")
                response = await send_request()
            else:
                st.error("Token refresh failed. Please log in again.")
                st.session_state.is_authenticated = False
                st.session_state.token = None
                st.session_state.refresh_token = None
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
                return None

        logger.info(f"Got response with status: {response.status_code}")

        if response.status_code == 200:
            try:
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        json_str = line[6:].strip()
                        logger.debug(f"Received chunk: {json_str}")

                        if not json_str:
                            continue

                        try:
                            data = json.loads(json_str)

                            if data.get('type') == 'done':
                                logger.info("Received 'done' signal.")
                                break

                            if data.get('type') == 'error':
                                error_data = data.get('data', 'Unknown server error')
                                logger.error(f"Received error from API: {error_data}")
                                st.error(f"Server error: {error_data}")
                                return None

                            content = data.get('content')
                            if content:
                                if full_response and not full_response.endswith(('\n', ' ')) and not content.startswith(('\n', ' ')):
                                    full_response += " "
                                full_response += content
                                response_placeholder.markdown(full_response + "▌")

                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}, raw data: '{json_str}'")
                            continue

                response_placeholder.empty()
                if full_response:
                    logger.info(f"Final response length: {len(full_response)}")
                    return full_response.strip()
                else:
                    logger.warning("No content received in the response stream.")
                    return None
            except httpx.RemoteProtocolError as e:
                logger.error(f"Connection closed prematurely: {e}")
                if full_response:
                    return full_response.strip()
                return None
        else:
            error_msg = await response.text()
            logger.error(f"Server error ({response.status_code}): {error_msg}")
            st.error(f"Failed to get response from server: {response.status_code}")
            return None

def process_query(query: str):
    """Process the query and update chat history synchronously"""
    if not query.strip():
        return

    if not st.session_state.is_authenticated:
        st.error("Please log in to use the service")
        return

    logger.info(f"Processing new query: {query}")
    
    # Add message to chat history with required fields
    user_message = {
        "role": "user",
        "content": query,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.chat_history.append(user_message)

    # Add to messages for display
    st.session_state.messages.append({
        "timestamp": user_message["timestamp"],
        "query": query,
        "response": "",
        "error": None
    })

    with st.spinner("🤔 Thinking..."):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(process_stream_async(query))
            loop.close()
            
            if response:
                # Update chat history with assistant's response
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.chat_history.append(assistant_message)
                
                # Update messages for display
                st.session_state.messages[-1]["response"] = response
                logger.info("Updated session state with response")
            else:
                st.session_state.messages[-1]["error"] = "No response received from server"
                logger.warning("No response received from server")
                
        except Exception as e:
            error_msg = str(e)
            # Replace cryptic error with a more descriptive message for translation API errors.
            if error_msg == "'str' object is not callable":
                error_msg = ("Translation service error: a configuration error occurred where an expected "
                             "function is a string. Please check your translation API configuration.")
            logger.error(f"Error in process_query: {error_msg}", exc_info=True)
            st.session_state.messages[-1]["error"] = error_msg
            st.error(f"Error: {error_msg}")

# Update the CSS with dark mode support
st.markdown("""
    <style>
    /* Dark mode compatibility */
    .message-container {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        font-size: 1rem;
    }
    
    .user-message {
        background-color: var(--background-color);
        border: 1px solid var(--secondary-background-color);
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        margin-right: 2rem;
    }
    
    /* Improved message styling */
    .message-content {
        margin: 0.5rem 0;
        line-height: 1.6;
        color: var(--text-color);
    }
    
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .avatar {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .timestamp {
        color: var(--text-color-secondary);
        font-size: 0.8rem;
        margin-left: auto;
    }
    
    /* Code block improvements */
    .message-content code {
        background-color: var(--code-background);
        color: var(--code-text);
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 0.85em;
    }
    
    .message-content pre {
        background-color: var(--code-background);
        padding: 1rem;
        border-radius: 6px;
        overflow-x: auto;
        border: 1px solid var(--border-color);
    }
    
    /* Other elements */
    .message-content blockquote {
        border-left: 3px solid var(--secondary-background-color);
        margin: 1rem 0;
        padding-left: 1rem;
        color: var(--text-color-secondary);
    }
    
    .message-content ul, .message-content ol {
        margin: 1rem 0;
        padding-left: 2rem;
    }
    
    hr {
        margin: 2rem 0;
        opacity: 0.2;
        border-color: var(--border-color);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--secondary-background-color);
        border-radius: 4px;
    }
    
    /* Improved form styling */
    .stTextArea textarea {
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stButton button {
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# Add a sidebar for settings (optional)
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Language selector
    selected_language = st.selectbox(
        "Choose Language:",
        ["English", "Malayalam"],
        index=0 if st.session_state.language == "en" else 1
    )
    st.session_state.language = "en" if selected_language == "English" else "ml"
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main content
st.title("⚖️ Indic Law Assistant")
st.markdown("""
    <div style='background-color: var(--secondary-background-color); padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        <h4>Welcome to your AI Legal Research Assistant!</h4>
        <p>Ask questions about Indian law, legal procedures, and case precedents. I'll help you find relevant information.</p>
        <p><small>Note: This is an AI assistant and should not be considered legal advice.</small></p>
    </div>
""", unsafe_allow_html=True)

# Authentication form
if not st.session_state.is_authenticated:
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if asyncio.run(login(username, password)):
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")
else:
    # Logout button
    if st.button("Logout"):
        st.session_state.is_authenticated = False
        st.session_state.token = None
        st.session_state.refresh_token = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    # Chat interface
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_area(
            "Your question:",
            height=100,
            key="user_input",
            placeholder="Type your legal question here..."
        )
        
        cols = st.columns([0.85, 0.15])
        with cols[0]:
            submit = st.form_submit_button(
                "Send Message 📤",
                use_container_width=True,
            )
        with cols[1]:
            clear = st.form_submit_button(
                "Clear 🗑️",
                use_container_width=True,
                on_click=lambda: st.session_state.messages.clear()
            )

        if submit and query:
            process_query(query)

        # Update the message display
        if st.session_state.messages:
            for msg in reversed(st.session_state.messages):
                # User message
                st.markdown(
                    f"""
                    <div class="message-container user-message">
                        <div class="message-header">
                            <img src="https://api.dicebear.com/6.x/initials/svg?seed=User" class="avatar" />
                            <strong>You</strong>
                            <div class="timestamp">{msg['timestamp']}</div>
                        </div>
                        <div class="message-content">{msg['query']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Assistant response
                if msg.get('response'):
                    st.markdown(
                        f"""
                        <div class="message-container assistant-message">
                            <div class="message-header">
                                <img src="https://api.dicebear.com/6.x/bottts/svg?seed=Assistant" class="avatar" />
                                <strong>Assistant</strong>
                                <div class="timestamp">{msg['timestamp']}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(msg['response'])
                elif msg.get('error'):
                    st.error(f"Error: {msg['error']}")
                st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.info("👋 Send a message to start the conversation!")
