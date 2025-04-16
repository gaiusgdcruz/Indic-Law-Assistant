import streamlit as st
import requests
import json

# Set the Ollama API URL
OLLAMA_URL = "http://localhost:11434/api/generate"

# Initialize session state for chat history and cached responses
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cached_responses" not in st.session_state:
    st.session_state.cached_responses = {}
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "clear_input" in st.session_state and st.session_state.clear_input:
    st.session_state.input_field = ""  # Clear input field
    st.session_state.clear_input = False  # Reset flag

# Streamlit UI
st.title("Indic Law Assist")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["response"])

# User input field
user_input = st.text_input("Enter your prompt:", key="input_field")

if st.button("Generate Response"):
    if user_input.strip():
        with st.spinner("Processing..."):
            # Store input separately
            st.session_state.user_input = user_input

            # Check if response exists in cache
            if st.session_state.user_input in st.session_state.cached_responses:
                model_response = st.session_state.cached_responses[st.session_state.user_input]
            else:
                # Send request with streaming enabled
                payload = {
                    "model": "deepseek-r1:1.5b",
                    "prompt": st.session_state.user_input,
                    "stream": True  # Enable streaming
                }
                try:
                    response = requests.post(OLLAMA_URL, data=json.dumps(payload), stream=True)

                    if response.status_code == 200:
                        # Create an empty container for streaming output
                        message_placeholder = st.empty()
                        streamed_response = ""

                        # Read and stream response chunks
                        for chunk in response.iter_lines():
                            if chunk:
                                decoded_chunk = json.loads(chunk.decode("utf-8"))
                                streamed_response += decoded_chunk.get("response", "")

                                # Update the message in real-time
                                message_placeholder.write(streamed_response)

                        # Cache the response
                        st.session_state.cached_responses[st.session_state.user_input] = streamed_response
                        model_response = streamed_response
                    else:
                        st.error(f"Error: {response.status_code}, {response.text}")
                        model_response = "Error in response generation."

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to Ollama. Make sure it is running.")
                    model_response = "Connection error."

            st.session_state.chat_history.append({"question": st.session_state.user_input, "response": model_response})

            # Set flag to clear input on next render
            st.session_state.clear_input = True  

            st.rerun()  # Forces UI refresh
    else:
        st.warning("Please enter a prompt before clicking 'Generate Response'.")
