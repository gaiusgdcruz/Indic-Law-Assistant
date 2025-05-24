# Indic-Law-Assistant

A Retrieval-Augmented Generation (RAG) system designed for Indian legal documents using local Large Language Models via Ollama. It parses Law Document PDFs, indexes them using ChromaDB, and enables semantic query answering via a Streamlit frontend and FastAPI.

---

## 🏗️ Project Structure

This project is organized into several modules, each with its own specific role:

-   **`./` (Main Application)**: The root of the project, containing general configuration and this main README.
-   **`api/`**: Provides the FastAPI backend that serves the RAG pipeline, handles user authentication, and manages interactions with language models. ([Details in api/README.md](api/README.md))
-   **`parser/`**: Responsible for processing PDF documents, extracting text, generating embeddings, and populating the vector store. ([Details in parser/README.md](parser/README.md))
-   **`streamlit_ui/`**: Contains the Streamlit application for user interaction, allowing users to query the system and view results. ([Details in streamlit_ui/README.md](streamlit_ui/README.md))
-   **`cache/`**: Implements an alternative semantic caching mechanism using Redis and RedisVL. Note that the main API uses a different caching strategy. ([Details in cache/README.md](cache/README.md))

Each module directory contains a specific `README.md` with more detailed information about its functionality and setup.

## ✨ Prerequisites

Ensure the following are installed before running the project:

- Python 3.10+
- `pip` for dependency management
- [Ollama](https://ollama.com/) (running locally for LLM inference)
- Chrome or any browser for Streamlit

### 1. Create a Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 2. Configure Environment Variables
An example of necessary environment variables can be found in the `.env` file in the project root (this was generated if you ran the previous setup step for standardizing configuration). Populate this file with your specific settings. Key configurations for different modules (API, Parser, Cache) are loaded from these environment variables.

### 3. Install Ollama
Begin by installing [Ollama](https://ollama.com/) following the instructions provided on the website.

### 4. Pull the Model
Pull the required model using the command, for example for `deepseek-r1:1.5b`:
```bash
ollama pull deepseek-r1:1.5b
```
(Ensure the models specified in your `.env` file for `GENERATOR_MODEL`, `REWRITER_RERANKER_MODEL`, `CACHE_OLLAMA_MODEL`, and the embedding models are pulled.)

### 5. Start Ollama
Once Ollama is installed, start the Ollama service:
```bash
ollama serve
```


## 🚀 FastAPI Setup

1. **Clone the repository**

2. **Activate your virtual environment** (if not already):
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies**:
Navigate to the project root directory (where this README is located) and install all dependencies:
```bash
pip install -r requirements.txt
```
This file includes dependencies for the API, parser, and UI. Module-specific `requirements.txt` files (e.g., in `api/`, `streamlit_ui/`, `cache/`) list subsets of these for informational or isolated setup purposes, but the root `requirements.txt` is comprehensive.

4. **Navigate to the API folder**:
```bash
cd api/
```

5. **Run the FastAPI server**:
```bash
uvicorn main:app --reload
```

Visit the API docs at:
```
http://localhost:8000/docs
```

### 🔍 Query Endpoint

- **Endpoint**: `/rag_query`
- **Method**: POST
- **Payload**:
```json
{
  "query": "What is Indian Forest law?"
}
```
- **Response**:
```json
{
  "answer": "The Indian forest law is...",
  "retrieved_sources": [...],
  "rewritten_query": "...",
  "processing_time": 2.34,
  "confidence_score": 0.91
}
```

## 💬 Streamlit UI Setup

1. **Ensure API is Running**: The Streamlit UI (`streamlit_app.py`) interacts with the FastAPI backend, so make sure it's running (see FastAPI Setup above).

2. **Install dependencies** (if you haven't already done so from the root `requirements.txt`):
It's recommended to install dependencies from the root `requirements.txt` as it's comprehensive. However, if setting up in isolation:
```bash
pip install -r streamlit_ui/requirements.txt
```

3. **Navigate to the Streamlit UI folder**:
```bash
cd streamlit_ui/
```

4. **Run the Streamlit app**:
```bash
streamlit run streamlit_app.py
```
The app should launch in your browser. For more details on the UI, including the simpler `main.py` (which connects directly to Ollama), see the [streamlit_ui/README.md](streamlit_ui/README.md).
