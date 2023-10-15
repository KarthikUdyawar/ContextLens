"""Custom Text Dataset for BERT-based Text Classification"""
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TextDataset(Dataset):
    """A custom text dataset for BERT-based text classification."""

    def __init__(self, texts: list, labels: list = None, max_length: int = 100):
        """Initialize the custom text dataset.

        Args:
            texts (list):  A list of text samples.
            labels (list, optional): A list of corresponding labels. Defaults to None.
            max_length (int, optional): Maximum sequence length. Defaults to 100.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, and labels (if available).
        """
        text = self.texts[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        if self.labels:
            label = self.labels[idx]
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.float),
            }
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
