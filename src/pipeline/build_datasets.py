"""Builds the datasets"""
import os
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from src.utils.split_data import TrainValidTestSplitter
from src.utils.text_preprocessor import TextPreprocessor

tqdm.pandas()

DATA_VERSION = "0.1v"

print("Start load data")
df = pd.read_parquet("data/Text_dataset.br", engine="pyarrow")
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

data_version_folder = f"src/data/{DATA_VERSION}"

if not os.path.exists(data_version_folder):
    os.makedirs(data_version_folder)

train_df.to_parquet(
    os.path.join(data_version_folder, "train_data.parquet"),
    engine="pyarrow",
    compression="brotli",
    index=False,
)

valid_df.to_parquet(
    os.path.join(data_version_folder, "valid_data.parquet"),
    engine="pyarrow",
    compression="brotli",
    index=False,
)

test_df.to_parquet(
    os.path.join(data_version_folder, "test_data.parquet"),
    engine="pyarrow",
    compression="brotli",
    index=False,
)
print("Done saving datasets\n")
