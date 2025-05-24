# API Module

This module provides the FastAPI backend for the Indic-Law-Assistant. It handles user queries, interacts with the RAG (Retrieval Augmented Generation) pipeline, and serves responses.

## Functionality

The API module is responsible for:

-   **Authentication**: Securely managing user access with JWT tokens (`/token`, `/token/refresh`).
-   **Query Processing**:
    -   Receiving user queries via HTTP POST requests.
    -   Determining if a query is legal or general in nature.
    -   For legal queries:
        -   Rewriting the query for better retrieval.
        -   Retrieving relevant documents from the vector store.
        -   Reranking the retrieved documents.
        -   Generating an answer based on the context from documents using a Large Language Model (LLM).
    -   For general queries:
        -   Generating a response directly using an LLM.
    -   Handling chat history to provide conversational context.
-   **Streaming Responses**: Providing real-time streaming of answers to the client (`/stream`).
-   **Translation**: Optional translation of queries and responses between English and Malayalam (`/translate`, and integrated into the streaming endpoint).
-   **Health Checks**: Offering an endpoint to monitor the API's status and its core components (`/health`).
-   **Caching**: Caching responses for frequently accessed endpoints like health checks.

## Setup and Execution

The main README.md in the root of the project contains detailed setup instructions for the entire application, including the API.

To run the API server specifically:

1.  **Navigate to the API folder**:
    ```bash
    cd api/
    ```
2.  **Install dependencies** (if not already installed from the root `requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```
    *Note: It's recommended to install all dependencies from the root `requirements.txt` to ensure consistency.*
3.  **Ensure Ollama is running** and the required models are pulled (see main README).
4.  **Run the FastAPI server**:
    ```bash
    uvicorn main:app --reload
    ```
    The API will be accessible at `http://localhost:8000`, and the OpenAPI documentation at `http://localhost:8000/docs`.

## Key Files and Components

-   **`main.py`**: The entry point for the FastAPI application. It defines all the API endpoints, handles request validation, and orchestrates the overall workflow. It also includes the streaming logic and authentication.
-   **`chain.py`**: Contains the core RAG pipeline logic built using LangChain Expression Language (LCEL). It defines how queries are processed, documents are retrieved and reranked, and answers are generated. It distinguishes between legal and general queries.
-   **`config.py`**: Manages configuration settings for the API, such as Ollama model names, vector store paths, API keys, and feature flags (e.g., for translation). It loads settings from environment variables and `.env` files.
-   **`models.py`**: Defines and initializes the language models (LLMs for generation and rewriting/reranking) and embedding models. It uses `ChatOllama` for LLM interaction and `SentenceTransformerEmbeddings` for creating embeddings.
-   **`schemas.py`**: Contains Pydantic models for request and response validation, ensuring data consistency across the API. Defines structures for queries, responses, chat messages, tokens, etc.
-   **`auth.py`**: Implements authentication logic, including JWT token creation, user authentication, and dependency functions for protecting endpoints.
-   **`prompts.py`**: Stores the prompt templates used for query rewriting and answer generation.
-   **`reranker.py`**: Implements the document reranking logic to improve the relevance of retrieved documents.
-   **`vector_store.py`**: Manages the connection to and interaction with the ChromaDB vector store where document embeddings are stored.
-   **`utils.py`**: Provides utility functions used across the API, such as text formatting and translation.
-   **`logging_config.py`**: Configures logging for the API module.
-   **`scripts/`**: Contains utility scripts, like notebooks for testing specific API functionalities (e.g., `Translate_API_POC.ipynb`).
-   **`util/`**: Contains utility sub-modules, like `cache.py` for caching mechanisms (though `main.py` uses `fastapi-cache`).

## Endpoints

The primary endpoints are:

-   **`POST /token`**: Authenticates a user and returns an access token.
-   **`POST /token/refresh`**: Refreshes an expired access token.
-   **`POST /stream`**: Streams a response to a query, supporting conversational history and language options. This is the main endpoint for user interaction.
-   **`POST /rag_query`**: (Likely a non-streaming version or for specific RAG interactions) Processes a query through the RAG pipeline and returns a consolidated response. *Consider if this is still the primary non-streaming endpoint or if `/stream` handles all cases.*
-   **`GET /health`**: Checks the health of the API and its dependencies.
-   **`POST /translate`**: Translates text between English and Malayalam.

For detailed request/response schemas and to try out the endpoints, refer to the OpenAPI documentation available at `/docs` when the API is running.
