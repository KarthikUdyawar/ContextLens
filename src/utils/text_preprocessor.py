"""This script performs text preprocessing on a dataset, including converting HTML entities, 
removing mentions, links, and emojis, converting text to lowercase, and more.
"""
# Import necessary libraries
import html
import re

import emoji
import pandas as pd


class TextPreprocessor:
    """Converts raw text into clean text"""

    def __init__(self):
        """Initialize the preprocessor"""
        # Load abbreviation, apostrophe, and emoticon data
        self.abbreviations_df = pd.read_csv(
            "data/Text-Preprocessing-Data/abbreviations.csv"
        )
        self.apostrophe_df = pd.read_csv("data/Text-Preprocessing-Data/apostrophe.csv")
        self.emoticons_df = pd.read_csv("data/Text-Preprocessing-Data/emoticons.csv")

        # Create dictionaries from dataframes
        self.abbreviations_dict = dict(self.abbreviations_df.values)
        self.apostrophe_dict = dict(self.apostrophe_df.values)
        self.emoticons_dict = dict(self.emoticons_df.values)

    def lookup_dict(self, text: str, dictionary: dict) -> str:
        """
        Replace placeholders in the text with values from a dictionary.

        Args:
            text (str): The text containing placeholders to be replaced.
            dictionary (dict): A dictionary containing placeholder-value pairs.

        Returns:
            str: The text with placeholders replaced by their corresponding values.
        """
        for word in text.split():
            if word in dictionary:
                text = text.replace(word, dictionary[word])
        return text

    def preprocessing(self, input_text: str) -> str:
        """Preprocess the input text for natural language processing.

        Args:
            input_text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text.
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
        # Step J : Replacing Punctuations, Special Characters & Numbers (integers) with space
        text = re.sub(r"[^a-z]", " ", text)
        # Step K: Remove whitespace
        text = re.sub(r"\s+", " ", text)
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
