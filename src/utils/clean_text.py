"""This script performs text preprocessing on a dataset, including converting HTML entities, 
removing mentions, links, and emojis, converting text to lowercase, and more.
"""
# Import necessary libraries
import html
import re

import emoji
import pandas as pd

# Load abbreviation, apostrophe, and emoticon data
abbreviations_df = pd.read_csv(
    "data/Text-Preprocessing-Data/abbreviations.csv"
)
apostrophe_df = pd.read_csv("data/Text-Preprocessing-Data/abbreviations.csv")
emoticons_df = pd.read_csv("data/Text-Preprocessing-Data/emoticons.csv")

# Create dictionaries from dataframes
abbreviations_dict = dict(abbreviations_df.values)
apostrophe_dict = dict(apostrophe_df.values)
emoticons_dict = dict(emoticons_df.values)


# Function to replace placeholders in text with values from a dictionary
def lookup_dict(text: str, dictionary: dict) -> str:
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


# Main text preprocessing function
def preprocessing(input_text: str) -> str:
    """Preprocess the input text for natural language processing.

    Args:
        input_text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    # Step A : Converting html entities i.e. (&lt; &gt; &amp;)
    text = html.unescape(input_text)
    # Step B : Removing "@user" from all the text
    text = re.sub("@[\w]*", "", text)
    # Step C : Remove http & https links
    text = re.sub("http://\S+|https://\S+", "", text)
    # Step D : Emoticon Lookup
    text = lookup_dict(text, emoticons_dict)
    # Step E : Emoji Lookup
    text = emoji.demojize(text)
    # Step F : Changing all the text into lowercase
    text = text.lower()
    # Step G : Apostrophe Lookup
    text = lookup_dict(text, apostrophe_dict)
    # Step H : Short Word Lookup
    text = lookup_dict(text, abbreviations_dict)
    # Step I : Replacing Punctuations, Special Characters & Numbers (integers) with space
    text = re.sub(r"[^a-z]", " ", text)
    # Step J: Remove whitespace
    text = re.sub(r"\s+", " ", text)
    return text


if __name__ == "__main__":
    from tqdm import tqdm

    tqdm.pandas()

    print("Start load data")
    df = pd.read_parquet("data/Text_dataset.br", engine="pyarrow")
    print("Done Load data")

    print("Start process")
    df["clear_text"] = df["text"].progress_apply(preprocessing)
    print("Done process")

    print("Start save")
    df.to_parquet(
        "data/Clean_text_dataset.br",
        engine="pyarrow",
        compression="brotli",
        index=False,
    )
    print("Done save")
