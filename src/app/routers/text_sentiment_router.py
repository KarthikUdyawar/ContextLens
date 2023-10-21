"""FastAPI Router for Text Sentiment Analysis"""
from fastapi import APIRouter

from src.app.controllers.text_sentiment_controller import (
    clean_user_text,
    predict_prob_sentiment,
    predict_sentiment,
)
from src.app.interfaces.text_sentiment_interface import (
    CleanTextResponse,
    TextProbResponse,
    TextRequest,
    TextResponse,
)

router = APIRouter()


@router.post("/predict/", response_model=TextResponse)
async def predict_text_sentiment(request: TextRequest) -> TextResponse:
    """Endpoint for predicting the sentiment of a given text.

    Args:
        request (TextRequest): Request model containing input text.

    Returns:
        TextResponse: Response model containing cleaned text and sentiment label.
    """
    return predict_sentiment(request)


@router.post("/predict-prob/", response_model=TextProbResponse)
async def predict_prob_text_sentiment(request: TextRequest) -> TextProbResponse:
    """Endpoint for predicting the sentiment probabilities of a given text.

    Args:
        request (TextRequest): Request model containing input text.

    Returns:
        TextProbResponse: Response model containing cleaned text and sentiment probabilities.
    """
    return predict_prob_sentiment(request)


@router.post("/clean/", response_model=CleanTextResponse)
async def clean_text(request: TextRequest) -> CleanTextResponse:
    """Endpoint for cleaning a given text.

    Args:
        request (TextRequest): Request model containing input text.

    Returns:
        CleanTextResponse: Response model containing cleaned text.
    """
    return clean_user_text(request)
