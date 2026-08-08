"""DataFrame Batch Loader for Stratified Sampling and Batching."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class DataFrameBatchLoader:
    """A class for stratified sampling and batching of a DataFrame."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        batch_size: int,
        stratify_col: str | None = None,
        random_state: int = 42,
    ) -> None:
        """Initialize the DataFrameBatchLoader.

        Args:
            data_frame: The DataFrame to be sampled and batched.
            batch_size: The batch size for creating mini-batches.
            stratify_col: The column for stratified sampling. Defaults to None.
            random_state: Random seed for reproducibility. Defaults to 42.
        """
        self.random_state = random_state
        self.data_frame = data_frame.sample(frac=1, random_state=self.random_state)
        self.batch_size = batch_size
        self.batches: list[pd.DataFrame] = [
            self.data_frame[i : i + batch_size]
            for i in range(0, len(self.data_frame), batch_size)
        ]
        self.stratify_col = stratify_col
        if self.stratify_col is not None:
            self.perform_stratified_sampling()

    def __len__(self) -> int:
        """Get the number of mini-batches.

        Returns:
            The number of mini-batches.
        """
        return len(self.batches)

    def __getitem__(self, index: int) -> pd.DataFrame:
        """Get a mini-batch by index.

        Args:
            index: The index of the mini-batch.

        Returns:
            The mini-batch.

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= index < len(self.batches):
            return self.batches[index]
        raise IndexError("Index out of range")

    def perform_stratified_sampling(self) -> None:
        """Perform stratified sampling to shuffle and maintain class distribution."""
        y = self.data_frame[self.stratify_col]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=self.random_state
        )
        train_idx, _ = next(sss.split(self.data_frame, y))

        shuffled_data_frame = self.data_frame.iloc[train_idx].sample(
            frac=1, random_state=self.random_state
        )
        self.data_frame = shuffled_data_frame
