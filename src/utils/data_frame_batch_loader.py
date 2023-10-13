"""DataFrame Batch Loader for Stratified Sampling and Batching"""

from sklearn.model_selection import StratifiedShuffleSplit


class DataFrameBatchLoader:
    """A batch loader for DataFrame with support for stratified sampling.

    Args:
        data_frame (pandas.DataFrame): The input DataFrame to create batches from.
        batch_size (int): The size of each batch.
        stratify (bool, optional): Whether to perform stratified sampling. Defaults to False.
        stratify_col (str, optional): The column for stratified sampling.
        Required if 'stratify' is True.

    Raises:
        ValueError: If 'stratify' is True but 'stratify_col' is not provided.

    Returns:
        int: The total number of batches available for loading.
    """

    def __init__(
        self,
        data_frame,
        batch_size: int,
        stratify: bool = False,
        stratify_col: str = None,
    ):
        """Initialize the DataFrame batch loader.

        Args:
            data_frame (pandas.DataFrame): The input DataFrame to create batches from.
            batch_size (int): The size of each batch.
            stratify (bool, optional): Whether to perform stratified sampling. Defaults to False.
            stratify_col (str, optional): The column for stratified sampling.
            Required if 'stratify' is True.

        Raises:
            ValueError: If 'stratify' is True but 'stratify_col' is not provided.
        """

        # Check for stratified sampling
        if stratify:
            if stratify_col is None:
                raise ValueError(
                    "You must specify the stratify_col if stratify is True."
                )

            # Extract the target variable for stratified sampling
            y = data_frame[stratify_col]

            # Perform stratified sampling to shuffle and maintain class distribution
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, _ = next(sss.split(data_frame, y))

            # Use the shuffled indices for creating batches
            shuffled_data_frame = data_frame.iloc[train_idx].sample(
                frac=1, random_state=42
            )
            self.data_frame = shuffled_data_frame
        else:
            self.data_frame = data_frame.sample(frac=1, random_state=42)

        self.batch_size = batch_size
        self.batches = [
            self.data_frame[i : i + batch_size]
            for i in range(0, len(self.data_frame), batch_size)
        ]

    def __len__(self) -> int:
        """Get the total number of batches available for loading.

        Returns:
            int: The total number of batches available for loading.
        """

        return len(self.batches)

    def __getitem__(self, index: int):
        """Get a batch by index.

        Args:
            index (int): The index of the batch to retrieve.

        Raises:
            IndexError: If the provided index is out of range.

        Returns:
            pandas.DataFrame: A batch of data from the DataFrame.
        """

        if 0 <= index < len(self.batches):
            return self.batches[index]
        else:
            raise IndexError("Index out of range")
