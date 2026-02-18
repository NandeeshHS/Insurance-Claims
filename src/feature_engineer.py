"""
Feature Engineering Module
==========================
Transforms raw insurance data into predictive features through:
- Date-based features (ages, durations)
- Derived metrics (ratios, interactions)
- Polynomial features

Author: Nandeesh H S
"""

import pandas as pd
import numpy as np
from config import REFERENCE_DATE


class FeatureEngineer:
    """
    Feature engineering for insurance claims prediction.

    Creates derived features from policy, driver, and vehicle data.
    """

    def __init__(self, reference_date: str = REFERENCE_DATE):
        """
        Initialize feature engineer.

        Args:
            reference_date: Reference date for calculating ages (YYYY-MM-DD)
        """
        self.reference_date = pd.to_datetime(reference_date)
        self.date_columns = ['X.2', 'X.3', 'X.4', 'X.5', 'X.6']

    def engineer_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.

        Args:
            df: Input DataFrame with raw features
            is_train: Whether this is training data (for logging purposes)

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        data_type = 'train' if is_train else 'test'
        print(f"Starting feature engineering for {data_type} data...")

        # Force-parse date columns (handles string dates and already-parsed datetimes)
        import pandas as pd
        for col in self.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Create date-derived features
        df = self._create_date_features(df)

        # Create derived metrics
        df = self._create_derived_features(df)

        # Create polynomial features
        df = self._create_polynomial_features(df)

        # Drop original date columns
        df = self._drop_date_columns(df)

        print(f"Feature engineering complete. New shape: {df.shape}")

        return df

    def _create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age and duration features from date columns."""

        # Driver age (from birth date X.5)
        if 'X.5' in df.columns:
            df['driver_age'] = (
                (self.reference_date - df['X.5']).dt.days / 365.25
            ).clip(15, 100).fillna(45).astype(int)

        # Driver experience (from license date X.6)
        if 'X.6' in df.columns:
            df['driver_exp'] = (
                (self.reference_date - df['X.6']).dt.days / 365.25
            ).clip(0, 80).fillna(20).astype(int)

        # Vehicle age (from manufacturing year X.22)
        if 'X.22' in df.columns:
            df['vehicle_age'] = (
                2019 - df['X.22']
            ).clip(0, 50).fillna(10).astype(int)

        # Policy duration (from start X.2 to end X.4)
        if 'X.2' in df.columns and 'X.4' in df.columns:
            df['policy_duration'] = (
                (df['X.4'] - df['X.2']).dt.days / 365.25
            ).clip(0, 10).fillna(1)

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio and interaction features."""

        # Power-to-weight ratio (vehicle performance metric)
        if 'X.23' in df.columns and 'X.28' in df.columns:
            df['power_to_weight'] = (
                df['X.23'] / (df['X.28'] + 1)
            ).fillna(0.08)

        # Premium-to-value ratio (pricing efficiency)
        if 'X.14' in df.columns and 'X.25' in df.columns:
            df['premium_to_value'] = (
                df['X.14'] / (df['X.25'] + 1)
            ).fillna(0.02)

        # Customer loyalty score (tenure ratio)
        if 'X.8' in df.columns and 'X.10' in df.columns:
            df['loyalty'] = (
                df['X.8'] / (df['X.10'] + 1)
            ).fillna(1)

        return df

    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create squared features for key variables."""

        if 'driver_age' in df.columns:
            df['driver_age_sq'] = df['driver_age'] ** 2

        if 'vehicle_age' in df.columns:
            df['vehicle_age_sq'] = df['vehicle_age'] ** 2

        return df

    def _drop_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove original date columns after feature extraction."""

        cols_to_drop = [col for col in self.date_columns if col in df.columns]
        if cols_to_drop:
            df = df.drop(cols_to_drop, axis=1)

        return df


def create_engineered_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Convenience function for feature engineering.

    Args:
        df: Input DataFrame
        is_train: Whether this is training data

    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    return engineer.engineer_features(df, is_train)
