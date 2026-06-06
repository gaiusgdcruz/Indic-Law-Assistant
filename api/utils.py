import logging
import re
import httpx
from typing import List, Union, Dict

from .config import TRANSLATION_API_URL
from langchain_core.documents import Document
from fastapi.responses import JSONResponse

from .schemas import ChatMessage

logger = logging.getLogger(__name__)


def format_docs(docs: List[Document]) -> str:
    """Combines document page contents into a single string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def parse_rerank_score(result: str) -> float:
    """Safely parses the LLM score output."""
    if not result:
        return 0.0
    match = re.search(r'\d+(\.\d+)?', result.strip())
    if match:
        try:
            score = float(match.group(0))
            return max(0.0, min(10.0, score))
        except ValueError:
            print(f"Warning: Could not parse score from '{result}'")
            return 0.0
    print(f"Warning: No numeric score found in '{result}'")
    return 0.0


def format_chat_history(messages: List[Union[ChatMessage, Dict]]) -> str:
    """Format chat history into a string.
    
    Args:
        messages: List of ChatMessage objects or dictionaries with 'role' and 'content' keys
    """
    logger.info(f"Formatting chat history. Input messages length: {len(messages)}")
    if not messages:
        logger.info("No previous conversation.")
        return "No previous conversation."

    formatted = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
        else:
            role = msg.role
            content = msg.content
            
        logger.info(f"Processing message role: {role}")
        display_role = "User" if role == "user" else "Assistant"
        formatted.append(f"{display_role}: {content[:50]}...")
    
    result = "\n".join(formatted)
    logger.info(f"Formatted chat history: {result}")
    return result


def handle_error(logger, e, message="An error occurred"):
    """Handles errors and returns a JSON response."""
    logger.error(message, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(e)})


async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text from source language to target language using the translation API.
    
    Args:
        text: The text to translate
        source_lang: Source language code (e.g., "en", "ml")
        target_lang: Target language code (e.g., "en", "ml")
        
    Returns:
        Translated text
        
    Raises:
        Exception if translation fails
    """
    # If languages are the same, return original text
    if source_lang == target_lang:
        return text
        
    # If text is empty, return empty
    if not text.strip():
        return text
    
    try:
        direction = f"{source_lang}-{target_lang}"
        logger.info(f"Translating text ({len(text)} chars) from {source_lang} to {target_lang}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TRANSLATION_API_URL,
                json={"text": text, "direction": direction},
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            
            if "translated_text" not in result:
                logger.error(f"Translation API returned unexpected response: {result}")
                return f"[Translation Error] {text}"
                
            logger.info(f"Translation successful: {len(text)} chars translated")
            return result["translated_text"]
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return f"[Translation Error: {str(e)}] {text}"