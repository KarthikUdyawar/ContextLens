"""Pydantic Models for Text Request and Response"""

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """A Pydantic model for incoming text requests.

    Args:
        text (str): The input text for analysis.
    """

    text: str = Field(..., example="This is a sample text for sentiment analysis.")


class TextResponse(BaseModel):
    """A Pydantic model for response with cleaned text and sentiment label.

    Args:
        cleaned_text (str): The cleaned input text.
        sentiment (str): The predicted sentiment label.
    """

    cleaned_text: str = Field(
        ..., example="this is a sample text for sentiment analysis "
    )
    sentiment: str = Field(..., example="neutral")


class TextProbResponse(BaseModel):
    """A Pydantic model for response with cleaned text and sentiment probabilities.

    Args:
        cleaned_text (str): The cleaned input text.
        sentiment_prob (list[float]): The predicted sentiment probabilities.
    """

    cleaned_text: str = Field(
        ..., example="this is a sample text for sentiment analysis "
    )
    sentiment_prob: list[float] = Field(
        ...,
        example="[0.21194206178188324,0.5761160254478455,0.2119419425725937] \
            #position [neg,neu,pos]",
    )


class CleanTextResponse(BaseModel):
    """A Pydantic model for response with cleaned text.

    Args:
        cleaned_text (str): The cleaned input text.
    """

    cleaned_text: str = Field(
        ..., example="this is a sample text for sentiment analysis "
    )
