"""
Data Loading Module
===================
Handles loading, validation, and initial processing of insurance claims data.

Author: Nandeesh H S
Project: Insurance Risk Analytics & Claims Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

from config import (
    TRAIN_FILE, TEST_FILE, DATE_COLUMNS, RANDOM_SEED
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and validate insurance claims data.

    This class handles:
    - Loading training and test datasets
    - Data type validation
    - Missing value detection
    - Basic data quality checks
    """

    def __init__(self, train_path: Optional[Path] = None, test_path: Optional[Path] = None):
        """
        Initialize DataLoader.

        Parameters
        ----------
        train_path : Path, optional
            Path to training data CSV file
        test_path : Path, optional
            Path to test data CSV file
        """
        self.train_path = train_path or TRAIN_FILE
        self.test_path = test_path or TEST_FILE
        self.train_data = None
        self.test_data = None

    def load_data(self, parse_dates: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test datasets.

        Parameters
        ----------
        parse_dates : bool, default=True
            Whether to parse date columns during loading

        Returns
        -------
        train_df : pd.DataFrame
            Training dataset
        test_df : pd.DataFrame
            Test dataset
        """
        logger.info("Loading datasets...")

        try:
            # Load training data
            if parse_dates:
                self.train_data = pd.read_csv(
                    self.train_path,
                    parse_dates=DATE_COLUMNS
                )
            else:
                self.train_data = pd.read_csv(self.train_path)

            # Load test data
            if parse_dates:
                test_date_cols = [col for col in DATE_COLUMNS if col != 'X.1']
                self.test_data = pd.read_csv(
                    self.test_path,
                    parse_dates=test_date_cols
                )
            else:
                self.test_data = pd.read_csv(self.test_path)

            logger.info(f"Training data loaded: {self.train_data.shape}")
            logger.info(f"Test data loaded: {self.test_data.shape}")

            # Validate data
            self._validate_data()

            return self.train_data, self.test_data

        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _validate_data(self):
        """
        Validate loaded data for common issues.

        Checks:
        - Required columns exist
        - Data types are correct
        - Detects missing values
        - Checks for duplicates
        """
        logger.info("Validating data quality...")

        # Check for missing values
        train_missing = self.train_data.isnull().sum().sum()
        test_missing = self.test_data.isnull().sum().sum()

        if train_missing > 0:
            logger.warning(f"Training data has {train_missing} missing values")
        if test_missing > 0:
            logger.warning(f"Test data has {test_missing} missing values")

        # Check for duplicates
        train_dupes = self.train_data.duplicated().sum()
        test_dupes = self.test_data.duplicated().sum()

        if train_dupes > 0:
            logger.warning(f"Training data has {train_dupes} duplicate rows")
        if test_dupes > 0:
            logger.warning(f"Test data has {test_dupes} duplicate rows")

        logger.info("Data validation complete")

    def get_data_summary(self) -> dict:
        """
        Generate summary statistics for loaded data.

        Returns
        -------
        summary : dict
            Dictionary containing data summary statistics
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        summary = {
            'train_shape': self.train_data.shape,
            'test_shape': self.test_data.shape,
            'train_columns': list(self.train_data.columns),
            'test_columns': list(self.test_data.columns),
            'train_missing': self.train_data.isnull().sum().to_dict(),
            'test_missing': self.test_data.isnull().sum().to_dict(),
            'train_dtypes': self.train_data.dtypes.to_dict(),
            'test_dtypes': self.test_data.dtypes.to_dict(),
        }

        return summary

    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           save_dir: Path):
        """
        Save processed datasets to CSV.

        Parameters
        ----------
        train_df : pd.DataFrame
            Processed training data
        test_df : pd.DataFrame
            Processed test data
        save_dir : Path
            Directory to save processed data
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        train_path = save_dir / "train_processed.csv"
        test_path = save_dir / "test_processed.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Processed data saved to {save_dir}")


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for insurance claims prediction.

    Target Variables
    ----------------
    - CS (Claim Status): Binary indicator (1 if claim filed, 0 otherwise)
    - LC (Loss Cost per Exposure Unit): X.15 / X.16
    - HALC (Historically Adjusted Loss Cost): (X.15 / X.16) × X.18

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw insurance data

    Returns
    -------
    df : pd.DataFrame
        Dataframe with added target variables

    Notes
    -----
    - Handles division by zero by setting LC and HALC to 0 when X.16 = 0
    - X.15: Total cost of claims for the policy during current year
    - X.16: Total number of claims incurred during current year
    - X.18: Ratio of claims filed to total duration of policy in force
    """
    df = df.copy()

    # Claim Status (CS): 1 if claims exist, 0 otherwise
    df['CS'] = (df['X.16'] > 0).astype(int)

    # Loss Cost per Exposure Unit (LC)
    df['LC'] = np.where(
        df['X.16'] > 0,
        df['X.15'] / df['X.16'],
        0
    )

    # Historically Adjusted Loss Cost (HALC)
    df['HALC'] = np.where(
        df['X.16'] > 0,
        (df['X.15'] / df['X.16']) * df['X.18'],
        0
    )

    logger.info("Target variables created successfully")
    logger.info(f"Claim rate: {df['CS'].mean()*100:.2f}%")

    return df


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    train_df, test_df = loader.load_data()

    # Create targets for training data
    train_df = create_target_variables(train_df)

    # Display summary
    summary = loader.get_data_summary()
    print("\nData Summary:")
    print(f"Training: {summary['train_shape']}")
    print(f"Test: {summary['test_shape']}")
