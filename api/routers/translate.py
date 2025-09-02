from fastapi import APIRouter, HTTPException

from ..schemas import TranslationRequest, TranslationResponse
from ..utils import translate_text

router = APIRouter(
    prefix="/translate",
    tags=["Translation"],
)


@router.post("", response_model=TranslationResponse)
async def translate_text_endpoint(request: TranslationRequest):
    """
    Translate text between specified languages using the configured translation service.
    """
    try:
        translated_text = await translate_text(
            request.text, request.source_lang, request.target_lang
        )
        return TranslationResponse(translated_text=translated_text)
    except Exception as e:
        error_detail = str(e)
        # Provide a more user-friendly error for a known configuration issue
        if "object is not callable" in error_detail:
            error_detail = (
                "Translation service error: A configuration error occurred. "
                "Please check the translation API setup."
            )
        raise HTTPException(status_code=500, detail=error_detail)
