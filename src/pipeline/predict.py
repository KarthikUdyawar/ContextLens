"""Text Sentiment Classifier for Sentiment Analysis."""

import logging
import os

import torch

from src.utils.custom_BERT_classifier import CustomBERTClassifier
from src.utils.text_dataset import TextDataset, get_tokenizer
from src.utils.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when inference is attempted but the checkpoint never loaded."""


class TextSentimentClassifier:
    """A Text Sentiment Classifier for predicting sentiment in text.

    Args:
        model_checkpoint_file (str): The path to the saved model checkpoint file.

    Methods:
        preprocess_text(input_text: str) -> str:
            Preprocess the input text for sentiment analysis.

        classify_sentiment(input_text: str,
            return_probabilities: bool = False) -> str or list:
            Predict the sentiment of the input text and return the result.

    """

    def __init__(self, model_checkpoint_file: str) -> None:
        """Initialize the TextSentimentClassifier.

        Args:
            model_checkpoint_file (str): The path to the saved model checkpoint file.
        """
        self.model = CustomBERTClassifier(num_classes=3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            print("=== GPU not found ===")
            print(f"Device found {self.device}")
        self.model.to(self.device)

        self.model_loaded = False
        if os.path.exists(model_checkpoint_file):
            try:
                # nosec B614: weights_only=True deferred — needs verification
                # against existing checkpoint format first. See DECISIONS.md.
                checkpoint = torch.load(  # nosec B614
                    model_checkpoint_file, map_location=self.device
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model_loaded = True
                print("Model loaded")
            except Exception:
                # Corrupt/incompatible checkpoint: log, stay degraded, don't crash
                # module import (FastAPI must still start and serve /health).
                logger.exception(
                    "Failed to load checkpoint at %s; serving in degraded mode.",
                    model_checkpoint_file,
                )
        else:
            print("Model not loaded.")
        self.preprocessor = TextPreprocessor()
        self.tokenizer = get_tokenizer()

    def preprocess_text(self, input_text: str) -> str:
        """Preprocess the input text for sentiment analysis.

        Args:
            input_text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        return self.preprocessor.preprocessing(input_text)

    def classify_sentiment(
        self, cleaned_text: str, return_probabilities: bool = False
    ) -> str | list[float]:
        """Predict the sentiment of already-cleaned text and return the result.

        Text is preprocessed exactly once, by the caller (via `preprocess_text`),
        not again inside this method (DECISIONS.md #3).

        Args:
            cleaned_text (str): Text that has already been run through
                `preprocess_text`.
            return_probabilities (bool, optional): Whether to return
                sentiment probabilities. Defaults to False.

        Returns:
            str or list: The predicted sentiment label or probabilities.
        """
        if not self.model_loaded:
            raise ModelNotReadyError(
                "Model checkpoint was not loaded; refusing to serve predictions "
                "from randomly-initialized weights (DECISIONS.md #4)."
            )
        text_dataset = TextDataset([cleaned_text], tokenizer=self.tokenizer)
        self.model = self.model.eval()
        with torch.no_grad():
            input_ids = text_dataset[0]["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = (
                text_dataset[0]["attention_mask"].unsqueeze(0).to(self.device)
            )
            outputs = self.model(input_ids, attention_mask)

        # `outputs` is already softmax-normalized inside CustomBERTClassifier.forward();
        # do NOT softmax again here (DECISIONS.md #1).
        y_pred_prob = outputs.cpu().numpy()[0]
        y_pred = torch.argmax(outputs, axis=1).cpu().numpy()[0]

        _result = (
            "positive" if y_pred == 2 else "negative" if y_pred == 0 else "neutral"
        )

        return y_pred_prob.tolist() if return_probabilities else _result


if __name__ == "__main__":
    # pylint: disable=invalid-name
    MODEL_FILE_PATH = "src/model/0.2v/model.pth"
    classifier = TextSentimentClassifier(MODEL_FILE_PATH)
    user_text = str(input("> "))
    print(f"User Text: {user_text}")
    clean_text = classifier.preprocess_text(user_text)
    print(f"Cleaned Text: {clean_text}")
    result = classifier.classify_sentiment(clean_text, return_probabilities=True)
    print(f"Sentiment: {result}")
