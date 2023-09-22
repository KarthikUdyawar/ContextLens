""""This script performs splitting data into train, validation, and test sets."""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_datetime64_any_dtype as is_datetime


class TrainValidTestSplitter:
    """Utility class for splitting data into train, validation, and test sets."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.3,
        random_state: int | None = None,
    ):
        """
        Initialize the splitter.

        Args:
            X (pd.DataFrame): The feature DataFrame.
            y (pd.Series): The target Series.
            test_size (float, optional): The proportion of data to include in the test split. Defaults to 0.3.
            random_state (int | None, optional): Seed for random number generation. Defaults to None.
        """
        # Initialize instance variables
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def split_data(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """
        Split the data into training, validation, and test sets.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: The split datasets.
        """
        # Split the data into train and a temporary set
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            stratify=self.y,
            random_state=self.random_state,
        )

        # Further split the temporary set into validation and test sets
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=self.random_state,
        )

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def random_oversample_multi_class(
        self, X: pd.DataFrame, y: pd.Series, ratio: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Randomly oversample minority classes in the dataset.

        Args:
            X (pd.DataFrame): The feature DataFrame.
            y (pd.Series): The target Series.
            ratio (float, optional): The oversampling ratio. Defaults to 1.0.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The resampled feature and target DataFrames.
        """
        # Set a random seed for reproducibility
        np.random.seed(self.random_state)

        # Find unique classes and their counts in the target variable
        unique_classes, class_counts = np.unique(y, return_counts=True)

        # Identify the majority class
        majority_class = unique_classes[np.argmax(class_counts)]

        X_resampled_list = []
        y_resampled_list = []

        for cls in unique_classes:
            if cls == majority_class:
                continue

            # Separate majority and minority samples
            X_majority = X[y != cls]
            X_minority = X[y == cls]
            y_majority = y[y != cls]
            y_minority = y[y == cls]

            # Calculate the number of samples to generate
            n_samples = int(len(X_majority) * ratio) - len(X_minority)

            # Randomly select samples from the minority class with replacement
            random_indices = np.random.choice(
                len(X_minority), size=n_samples, replace=True
            )
            X_resampled_cls = pd.concat(
                [X_majority, X_minority.iloc[random_indices]], axis=0
            )
            y_resampled_cls = pd.concat(
                [y_majority, y_minority.iloc[random_indices]], axis=0
            )

            X_resampled_list.append(X_resampled_cls)
            y_resampled_list.append(y_resampled_cls)

        # Combine the resampled data for all classes
        X_resampled = pd.concat(X_resampled_list, axis=0)
        y_resampled = pd.concat(y_resampled_list, axis=0)

        return X_resampled, y_resampled

    def get_dataframes(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the split datasets for training, validation, and test.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training, validation, and test DataFrames.
        """
        # Split the data into training, validation, and test sets
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.split_data()

        # Randomly oversample the training set to balance classes
        X_resampled, y_resampled = self.random_oversample_multi_class(
            X_train, y_train, ratio=1.0
        )

        # Create DataFrames for training, validation, and test sets
        train_df = pd.DataFrame(X_resampled, columns=X_resampled.columns)
        train_df["target"] = y_resampled

        valid_df = pd.DataFrame(X_valid, columns=X_valid.columns)
        valid_df["target"] = y_valid

        test_df = pd.DataFrame(X_test, columns=X_test.columns)
        test_df["target"] = y_test

        # Reset the index for each DataFrame
        train_df.reset_index(drop=True, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        return train_df, valid_df, test_df

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize the memory usage of a pandas DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to be optimized.

        Returns:
            pd.DataFrame: The optimized DataFrame.
        """
        optimized_df = df.copy()

        # Define a function to optimize object columns
        def _object_optimizer(
            optimized_df: pd.DataFrame, col: str, col_data: pd.Series
        ) -> None:
            if col_data.apply(lambda x: isinstance(x, str)).any():
                optimized_df[col] = col_data.astype("string")
            if is_datetime(col_data):
                optimized_df[col] = pd.to_datetime(col_data)
            num_unique = len(col_data.unique())
            num_total = len(col_data)
            if num_unique / num_total < 0.2:
                optimized_df[col] = col_data.astype("category")

        # Define a function to optimize integer columns
        def _int_optimizer(
            optimized_df: pd.DataFrame, col: str, col_data: pd.Series
        ) -> None:
            int_type = "unsigned" if col_data.min() >= 0 else "integer"
            optimized_df[col] = pd.to_numeric(col_data, downcast=int_type)

        # Define a function to optimize float columns
        def _float_optimizer(
            optimized_df: pd.DataFrame, col: str, col_data: pd.Series
        ) -> None:
            optimized_df[col] = pd.to_numeric(col_data, downcast="float")

        # Iterate through each column in the DataFrame
        for col in optimized_df.columns:
            col_data = optimized_df[col]
            dtype = col_data.dtype

            # Convert object columns to category dtype if less than 50% unique values
            if dtype == object:
                _object_optimizer(optimized_df, col, col_data)

            # Optimize integer columns
            elif dtype == "int64":
                _int_optimizer(optimized_df, col, col_data)

            # Optimize float columns
            elif dtype == "float64":
                _float_optimizer(optimized_df, col, col_data)

        # Calculate and print memory usage before and after optimization
        before_mem = df.memory_usage().sum() / 1024**2
        after_mem = optimized_df.memory_usage().sum() / 1024**2
        print(f"Memory usage before optimization: {before_mem:.2f} MB")
        print(f"Memory usage after optimization: {after_mem:.2f} MB")

        return optimized_df
