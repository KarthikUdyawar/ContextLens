"""Custom Text Dataset for BERT-based Text Classification."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


@lru_cache(maxsize=1)
def get_tokenizer(pretrained_model: str = "bert-base-uncased") -> BertTokenizer:
    """Load (and cache) the BERT tokenizer.

    `from_pretrained` does disk/network I/O; caching means it only runs once
    per process instead of once per `TextDataset` instantiation (DECISIONS.md #6).

    Args:
        pretrained_model: Tokenizer checkpoint name.
            Defaults to "bert-base-uncased".

    Returns:
        The (cached) tokenizer instance.
    """
    # nosec B615: unpinned revision, same as custom_BERT_classifier.py
    return BertTokenizer.from_pretrained(pretrained_model)  # nosec B615


class TextDataset(Dataset):  # type: ignore[misc]
    """A custom text dataset for BERT-based text classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[Any] | None = None,
        max_length: int = 100,
        tokenizer: BertTokenizer | None = None,
    ) -> None:
        """Initialize the custom text dataset.

        Args:
            texts: A list of text samples.
            labels: A list of corresponding labels. Defaults to None.
            max_length: Maximum sequence length. Defaults to 100.
            tokenizer: Pre-loaded tokenizer to reuse.
                Defaults to the cached `get_tokenizer()` instance.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.max_length = max_length

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample from the dataset by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing input_ids, attention_mask, and
            labels (if available).
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
