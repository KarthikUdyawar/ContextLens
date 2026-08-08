"""Custom BERT-based Text Classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel


class CustomBERTClassifier(nn.Module):  # type: ignore[misc]
    """A custom BERT-based text classifier.

    Args:
        num_classes: The number of classes for classification.
        dropout_prob: Probability of dropout. Defaults to 0.2.
    """

    def __init__(self, num_classes: int, dropout_prob: float = 0.2) -> None:
        """Initialize the custom BERT-based text classifier.

        Args:
            num_classes: The number of classes for classification.
            dropout_prob: Probability of dropout. Defaults to 0.2.
        """
        super().__init__()
        # Loading pre-implemented BERT model
        # nosec B615: unpinned revision, pre-existing since v1.0.0 — see
        # DECISIONS.md for revision-pin policy status.
        self.bert = BertModel.from_pretrained("bert-base-uncased")  # nosec B615
        # Custom additional layers
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)  # Add an additional fully connected layer
        self.fc3 = nn.Linear(64, num_classes)  # Add one more fully connected layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the custom BERT-based text classifier.

        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask.

        Returns:
            Predicted class probabilities.
        """
        # BERT model
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extracting CLS token representation
        cls_output = bert_outputs["last_hidden_state"][:, 0, :]
        # Custom layers with Dropout
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
