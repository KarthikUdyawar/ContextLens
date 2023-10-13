"""Custom Text Dataset for BERT-based Text Classification"""
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """A custom text dataset for BERT-based text classification."""

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int):
        """Initialize the custom text dataset.

        Args:
            texts (list): A list of text samples.
            labels (list): A list of corresponding labels.
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer):
            The text tokenizer.
            max_length (int): Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
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
        label = self.labels[idx]

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

        if label:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.float),
            }
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
