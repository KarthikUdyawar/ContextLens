"""Text Sentiment Classifier for Sentiment Analysis"""

from src.pipeline.predict import TextSentimentClassifier


def main():
    """Main function for the interactive program."""
    MODEL_FILE_PATH = "src/model/0.2v/model.pth"
    classifier = TextSentimentClassifier(MODEL_FILE_PATH)
    while True:
        print("Text Sentiment Classifier for Sentiment Analysis")
        user_text = str(input("> "))
        print(f"User Text: {user_text}", end="\n\n")

        clean_text = classifier.preprocess_text(user_text)
        print(f"Cleaned Text: {clean_text}", end="\n\n")

        result = classifier.classify_sentiment(clean_text)
        print(f"Sentiment: {result}", end="\n\n")

        result_prob = classifier.classify_sentiment(
            clean_text, return_probabilities=True
        )
        print(f"Sentiment probability: {result_prob}", end="\n\n")

        choice = input("Try again (Y/n): ")
        if choice.lower() == "n":
            break


if __name__ == "__main__":
    main()
