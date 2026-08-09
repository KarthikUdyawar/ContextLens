"""Text preprocessing for a dataset.

HTML entity conversion, mention/link removal, emoji handling, lowercasing, and more.
"""

from __future__ import annotations

import html
import os
import re

import emoji
import pandas as pd

# src/utils/text_preprocessor.py -> src/Text-Preprocessing-Data/, independent of cwd
_LOOKUP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Text-Preprocessing-Data"
)


class TextPreprocessor:
    """Converts raw text into clean text."""

    def __init__(self) -> None:
        """Initialize the preprocessor."""
        # Load abbreviation, apostrophe, and emoticon data
        self.abbreviations_df = pd.read_csv(
            os.path.join(_LOOKUP_DIR, "abbreviations.csv")
        )
        self.apostrophe_df = pd.read_csv(os.path.join(_LOOKUP_DIR, "apostrophe.csv"))
        self.emoticons_df = pd.read_csv(os.path.join(_LOOKUP_DIR, "emoticons.csv"))

        # Create dictionaries from dataframes
        self.abbreviations_dict: dict[str, str] = dict(self.abbreviations_df.values)
        self.apostrophe_dict: dict[str, str] = dict(self.apostrophe_df.values)
        self.emoticons_dict: dict[str, str] = dict(self.emoticons_df.values)

    def lookup_dict(self, text: str, dictionary: dict[str, str]) -> str:
        """Replace placeholders in the text with values from a dictionary.

        Args:
            text: The text containing placeholders to be replaced.
            dictionary: A dictionary containing placeholder-value pairs.

        Returns:
            The text with placeholders replaced by their corresponding values.
        """
        for word in text.split():
            if word in dictionary:
                text = text.replace(word, dictionary[word])
        return text

    def preprocessing(self, input_text: str) -> str:
        """Preprocess the input text for natural language processing.

        Args:
            input_text: The input text to be preprocessed.

        Returns:
            The preprocessed text.
        """
        # Step A : Converting html entities i.e. (&lt; &gt; &amp;)
        text = html.unescape(input_text)
        # Step B: Remove HTML tags
        text = re.sub(re.compile("<.*?>"), "", text)
        # Step C : Removing "@user" from all the text
        text = re.sub("@[\\w]*", "", text)
        # Step D : Remove http & https links
        text = re.sub("http://\\S+|https://\\S+", "", text)
        # Step E : Emoticon Lookup
        text = self.lookup_dict(text, self.emoticons_dict)
        # Step F : Emoji Lookup
        text = emoji.demojize(text, delimiters=(" ", " "))
        # Step G : Changing all the text into lowercase
        text = text.lower()
        # Step H : Apostrophe Lookup
        text = text.replace("’", "'")
        text = self.lookup_dict(text, self.apostrophe_dict)
        # Step I : Short Word Lookup
        text = self.lookup_dict(text, self.abbreviations_dict)
        # Step J : Replacing Punctuations, Special Characters & Numbers
        # (integers) with space
        text = re.sub(r"[^a-z]", " ", text)
        # Step K: Remove whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


if __name__ == "__main__":
    import os

    from tqdm import tqdm

    tqdm.pandas()

    print("Start load data")
    df = pd.read_parquet("data/Text_dataset.br", engine="pyarrow")
    print("Done Load data")

    print("Start process")
    preprocessor = TextPreprocessor()
    df["clean_text"] = df["text"].progress_apply(preprocessor.preprocessing)
    print("Done process")

    print("Start save")
    if not os.path.exists("data/intermediate_data"):
        os.makedirs("data/intermediate_data")

    df.to_parquet(
        "data/intermediate_data/clean_dataset.br",
        engine="pyarrow",
        compression="brotli",
        index=False,
    )
    print("Done save")
