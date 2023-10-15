"""Text Sentiment Classifier for Sentiment Analysis"""
import os

import torch

from src.utils.custom_BERT_classifier import CustomBERTClassifier
from src.utils.text_dataset import TextDataset
from src.utils.text_preprocessor import TextPreprocessor


class TextSentimentClassifier:
    """A Text Sentiment Classifier for predicting sentiment in text.

    Args:
        model_checkpoint_file (str): The path to the saved model checkpoint file.

    Methods:
        preprocess_text(input_text: str) -> str:
            Preprocess the input text for sentiment analysis.

        classify_sentiment(input_text: str, return_probabilities: bool = False) -> str or list:
            Predict the sentiment of the input text and return the result.

    """

    def __init__(self, model_checkpoint_file: str):
        """Initialize the TextSentimentClassifier.

        Args:
            model_checkpoint_file (str): The path to the saved model checkpoint file.
        """
        self.model = CustomBERTClassifier(num_classes=3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            print("=== GPU not found ===")
        self.model.to(self.device)
        if os.path.exists(model_checkpoint_file):
            checkpoint = torch.load(model_checkpoint_file)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("Model loaded")
        else:
            print("Model not loaded.")
        self.preprocessor = TextPreprocessor()

    def preprocess_text(self, input_text: str) -> str:
        """Preprocess the input text for sentiment analysis.

        Args:
            input_text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        return self.preprocessor.preprocessing(input_text)

    def classify_sentiment(
        self, input_text: str, return_probabilities: bool = False
    ) -> str | list:
        """Predict the sentiment of the input text and return the result.

        Args:
            input_text (str): The text for which sentiment should be classified.
            return_probabilities (bool, optional): Whether to return sentiment probabilities.
            Defaults to False.

        Returns:
            str or list: The predicted sentiment label or probabilities.
        """
        clean_text = self.preprocess_text(input_text)
        text_dataset = TextDataset([clean_text])
        self.model = self.model.eval()
        with torch.no_grad():
            input_ids = text_dataset[0]["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = (
                text_dataset[0]["attention_mask"].unsqueeze(0).to(self.device)
            )
            outputs = self.model(input_ids, attention_mask)

        y_pred_prob = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        y_pred = torch.argmax(outputs, axis=1).cpu().numpy()[0]

        result = "positive" if y_pred == 2 else "negative" if y_pred == 0 else "neutral"

        return y_pred_prob if return_probabilities else result


if __name__ == "__main__":
    file_name = "src/model/0.1v/checkpoint_6/model_checkpoint+6.pth"
    classifier = TextSentimentClassifier(file_name)
    input_text = str(input("> "))
    print(f"Input Text: {input_text}")
    clean_text = classifier.preprocess_text(input_text)
    print(f"Cleaned Text: {clean_text}")
    result = classifier.classify_sentiment(input_text, return_probabilities=True)
    print(f"Sentiment: {result}")
