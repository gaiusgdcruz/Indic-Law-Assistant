# Cache Module

This module implements a semantic caching layer using Redis and RedisVL, designed to work with Ollama embeddings. It provides a way to cache responses from Large Language Models (LLMs) based on the semantic similarity of prompts.

**Note:** The main API (`../api/`) currently uses `fastapi-cache` with an `InMemoryBackend` for its caching needs, as configured in `api/main.py`. This module represents an alternative or supplementary semantic caching mechanism.

## Functionality

-   **Semantic Caching**: Stores LLM responses and retrieves them if a new prompt is semantically similar to a cached prompt. This can reduce latency and LLM operational costs.
-   **Custom Vectorizer**: Uses a custom text vectorizer (`cache_vectorizer.py`) that leverages Ollama embeddings (`nomic-embed-text` by default) to generate vector representations of prompts for similarity comparison.
-   **Redis Integration**: Utilizes RedisVL for efficient storage and retrieval of vectorized prompts and their corresponding responses.

## Key Files

-   **`cache_vectorizer.py`**: Defines `create_vectorizer()`, which sets up a `CustomTextVectorizer` for RedisVL. This vectorizer uses `OllamaEmbeddings` to generate text embeddings. It provides both synchronous and asynchronous embedding methods.
-   **`config.py`**: Contains configuration settings for the cache module, including:
    -   `EMBEDDING_MODEL`: The Ollama embedding model to use (e.g., "nomic-embed-text").
    -   `CACHE_NAME`: The name of the semantic cache in Redis.
    -   `DB_HOST`: The hostname for the Redis server.
    -   `OLLAMA_MODEL`: The Ollama model used for generation in the sample script (e.g., "llama3.2:1b").
-   **`sample.py`**: A command-line script demonstrating how to use the `SemanticCache`. It:
    -   Initializes the `SemanticCache` with a Redis connection and the custom vectorizer.
    -   Takes a user question as input.
    -   Checks if a semantically similar question exists in the cache.
    -   If a cache hit occurs, it returns the cached response.
    -   If not, it queries an Ollama LLM, prints the response, and stores the new prompt-response pair in the cache.
-   **`requirements.txt`**: Lists the Python dependencies specific to this cache module, such as `redisvl` and `langchain-ollama`.

## Setup

1.  **Ensure Redis is running**: This module requires a Redis server. The default configuration connects to `localhost:6379`.
2.  **Install dependencies**:
    ```bash
    pip install -r cache/requirements.txt
    ```
3.  **Configure Environment**:
    -   Set up Ollama and ensure the embedding model specified in `cache/config.py` (or environment variables) is available.
    -   You might need to set environment variables if you don't want to use the defaults in `config.py`, though the script currently reads directly from `config.py`. A `.env` file can be used with `python-dotenv` as shown in `sample.py`.

## Usage Example

The `sample.py` script provides a direct way to test the semantic cache:

```bash
python cache/sample.py
```

It will prompt you to enter a question. The script will then attempt to retrieve a response from the cache or, if not found, generate one using an Ollama LLM and store it in the cache.

## Integration Notes

-   This caching mechanism is distinct from the `fastapi-cache` used in the main `api` module.
-   To integrate this semantic cache into the main API or other parts of the application, you would need to:
    1.  Initialize the `SemanticCache` instance (similar to `sample.py`).
    2.  Before calling the LLM, use `llmcache.check(prompt=query)` to look for a cached response.
    3.  If a response is found, return it.
    4.  Otherwise, call the LLM and then store the new response using `llmcache.store(prompt=query, response=llm_response)`.
    5.  Ensure the Redis server is accessible and configured correctly for the application environment.
    6.  The embedding model (`nomic-embed-text`) used here by default differs from the one in the main API (`paraphrase-multilingual-MiniLM-L12-v2`). For effective semantic caching within the larger application context, consistent embedding models should be considered.
```
