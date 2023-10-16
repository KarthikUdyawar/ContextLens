"""Controller Functions for Text Sentiment Analysis"""
from src.app.interfaces.text_sentiment_interface import (CleanTextResponse,
                                                         TextProbResponse,
                                                         TextRequest,
                                                         TextResponse)
from src.pipeline.predict import TextSentimentClassifier

file_name = "src/model/0.1v/checkpoint_6/model_checkpoint+6.pth"
classifier = TextSentimentClassifier(file_name)


def clean_user_text(request: TextRequest) -> CleanTextResponse:
    """Clean a user's input text.

    Args:
        request (TextRequest): Input text to clean.

    Returns:
        CleanTextResponse: Cleaned text.
    """
    input_text = request.text
    return CleanTextResponse(cleaned_text=classifier.preprocess_text(input_text))


def predict_sentiment(request: TextRequest) -> TextResponse:
    """Predict the sentiment of a user's input text.

    Args:
        request (TextRequest): Input text to predict sentiment for.

    Returns:
        TextResponse: Cleaned text and predicted sentiment.
    """
    input_text = request.text
    cleaned_text = classifier.preprocess_text(input_text)
    result = classifier.classify_sentiment(input_text, return_probabilities=False)
    return TextResponse(cleaned_text=cleaned_text, sentiment=result)


def predict_prob_sentiment(request: TextRequest) -> TextResponse:
    """Predict the sentiment probabilities of a user's input text.

    Args:
        request (TextRequest): Input text to predict sentiment probabilities for.

    Returns:
        TextProbResponse: Cleaned text and sentiment probabilities.
    """
    input_text = request.text
    cleaned_text = classifier.preprocess_text(input_text)
    result = classifier.classify_sentiment(input_text, return_probabilities=True)
    return TextProbResponse(cleaned_text=cleaned_text, sentiment_prob=result)
