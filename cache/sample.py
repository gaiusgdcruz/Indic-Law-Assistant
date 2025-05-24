import time
import config
from langchain_ollama import ChatOllama
from redisvl.extensions.llmcache import SemanticCache
from cache_vectorizer import vectorizer
from dotenv import load_dotenv

# Ensure config is imported relatively if sample.py is part of the cache module
from . import config 
from .cache_vectorizer import vectorizer # Ensure relative import

load_dotenv(dotenv_path=config.dotenv_path) # Load .env from the path specified in config

redis_url = f"redis://{config.CACHE_REDIS_HOST}:{config.CACHE_REDIS_PORT}/{config.CACHE_REDIS_DB}"
# Initialize Semantic Cache

try:
    llmcache = SemanticCache(
        name=config.CACHE_NAME,
        redis_url=redis_url,
        distance_threshold=0.1,
        vectorizer=vectorizer, # This vectorizer uses config.CACHE_EMBEDDING_MODEL
        connection_kwargs={
            'decode_responses': True,
            'socket_timeout': 5,
            'retry_on_timeout': True
        }
        # dimension is often inferred by redisvl based on the first vector stored,
        # or can be explicitly set if known for CACHE_EMBEDDING_MODEL.
        # For "nomic-embed-text" (default CACHE_EMBEDDING_MODEL), the dimension is 768.
        # If CACHE_EMBEDDING_MODEL is changed, this might need adjustment or to be dynamic.
        # For now, assuming the default or that the user manages this.
        # dimension=768 # Example: for nomic-embed-text
    )
except Exception as e:
    print(f"Error initializing SemanticCache: {e}")
    exit(1)

question = input("Enter your question:")

# Use CACHE_OLLAMA_MODEL for the sample script's LLM interaction
client = ChatOllama(model=config.CACHE_OLLAMA_MODEL, verbose=True)


def ask_ollama(question: str) -> str:
    response = client.invoke(input=question)
    return response.content


# Measure time for cache check
start_time = time.time()
cached_response = llmcache.check(prompt=question)
cache_time = time.time() - start_time

if cached_response:
    print("Returned from cache")
    print("Prompt:", cached_response[0]['prompt'])
    print("Response:", cached_response[0]['response'])
    print(f"Time taken for cache retrieval: {cache_time:.4f} seconds")
else:
    try:
        start_time = time.time()
        response = ask_ollama(question)
        llm_time = time.time() - start_time
        print("No response in cache. LLM generated response.")
        print("Prompt:", question)
        print("Response:", response)
        print(f"Time taken for LLM response: {llm_time:.4f} seconds")

        # Store the response in cache
        llmcache.store(prompt=question, response=response)
        print("Stored in cache successfully.")
    except Exception as e:
        print(f"Error during LLM operations: {e}")