"""Builds the datasets"""
import os

import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

from src.utils.split_data import TrainValidTestSplitter
from src.utils.text_preprocessor import TextPreprocessor

tqdm.pandas()

PWD = os.getcwd()

BATCH_SIZE = 6 * 5000 # 6 * 5000
DATA_VERSION = "0.2v"
SOURCE_FILE_DIR = f"{PWD}/artifacts/Text_dataset.br"
DATA_VERSION_DIR = f"{PWD}/src/data/{DATA_VERSION}"

print("Start load data")
df = pd.read_parquet(SOURCE_FILE_DIR, engine="pyarrow")
df = df.sample(BATCH_SIZE, random_state=42)
print("Done Load data\n")

print("Start pre processing")
preprocessor = TextPreprocessor()
df["clean_text"] = df["text"].progress_apply(preprocessor.preprocessing)
print("Done pre processing\n")

df = df.drop_duplicates(subset=["clean_text"])


def target_encoder(text: str) -> str:
    """
    Encode text sentiment polarity.

    Args:
        text (str): The input text for sentiment analysis.

    Returns:
        str: sentiment polarity.
    """
    polarity = TextBlob(text).sentiment.polarity  # type: ignore
    return "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"


print("Start target encoding")

df["target"] = df["clean_text"].progress_apply(target_encoder)
print("Done target encoding\n")

print("Start filtering")
df["text_length"] = df["clean_text"].progress_apply(lambda x: len(str(x).split()))
filtered_df = df[(df["text_length"] != 0) & (df["text_length"] < 100)]
print("Done filtering\n")

print("Start spiting data")

X = df["clean_text"].to_frame(name="text")
y = df["target"]

data_splitter = TrainValidTestSplitter(X, y, test_size=0.3, random_state=42)

train_df, valid_df, test_df = data_splitter.get_dataframes()

print("Done spiting data\n")

print("Start optimize space")

train_df = data_splitter.optimize_dataframe(train_df)
valid_df = data_splitter.optimize_dataframe(valid_df)
test_df = data_splitter.optimize_dataframe(test_df)

print(f"{train_df.info() = }\n")
print(f"{valid_df.info() = }\n")
print(f"{test_df.info() = }\n")

print("Done optimize space\n")

print("Start saving datasets")


if not os.path.exists(DATA_VERSION_DIR):
    os.makedirs(DATA_VERSION_DIR)

train_df.to_parquet(
    os.path.join(DATA_VERSION_DIR, "train_data.parquet"),
    engine="pyarrow",
    compression="brotli",
    index=False,
)

valid_df.to_parquet(
    os.path.join(DATA_VERSION_DIR, "valid_data.parquet"),
    engine="pyarrow",
    compression="brotli",
    index=False,
)

test_df.to_parquet(
    os.path.join(DATA_VERSION_DIR, "test_data.parquet"),
    engine="pyarrow",
    compression="brotli",
    index=False,
)

print("Done saving datasets\n")
