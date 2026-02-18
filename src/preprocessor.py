"""
Data Preprocessing Module
=========================
Handles encoding, scaling, and feature selection.

Author: Nandeesh H S
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from config import RANDOM_SEED, N_FEATURES_TO_SELECT


class DataPreprocessor:
    """
    Preprocesses features for modeling.

    Handles:
    - Target variable separation
    - Categorical encoding (one-hot)
    - Feature scaling
    - Feature selection
    """

    def __init__(self, n_features_to_select: int = N_FEATURES_TO_SELECT):
        """
        Initialize preprocessor.

        Args:
            n_features_to_select: Number of features to select (if using selection)
        """
        self.n_features_to_select = n_features_to_select
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.categorical_cols = ['X.7', 'X.13', 'X.19', 'X.20', 'X.21', 'X.27']
        self.target_cols = ['X.1', 'X.15', 'X.16', 'X.17', 'X.18', 'CS', 'LC', 'HALC']
        self.fitted = False

    def prepare_features_targets(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Separate features and targets from training data.

        Args:
            train_df: Training DataFrame with targets
            test_df: Test DataFrame without targets

        Returns:
            Tuple of (X_train, X_test, targets_dict)
        """
        # Extract targets from training data
        targets = {}
        if 'CS' in train_df.columns:
            targets['CS'] = train_df['CS'].values
        if 'LC' in train_df.columns:
            targets['LC'] = train_df['LC'].values
        if 'HALC' in train_df.columns:
            targets['HALC'] = train_df['HALC'].values

        # Get feature columns
        feature_cols = [col for col in train_df.columns if col not in self.target_cols]

        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()

        return X_train, X_test, targets

    def encode_categorical(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        One-hot encode categorical variables.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (X_train_encoded, X_test_encoded)
        """
        # Filter to only existing categorical columns
        cat_cols = [col for col in self.categorical_cols if col in X_train.columns]

        if not cat_cols:
            return X_train, X_test

        print(f"Encoding {len(cat_cols)} categorical variables...")

        # One-hot encode
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=int)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=int)

        # Align test columns with train
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        print(f"Features after encoding: {X_train.shape[1]}")

        return X_train, X_test

    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        print("Scaling features...")

        # Fit on train, transform both
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

        return X_train_scaled, X_test_scaled

    def select_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y: np.ndarray,
        method: str = 'rf_importance'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Select most important features.

        Args:
            X_train: Training features
            X_test: Test features
            y: Target variable for feature selection
            method: Selection method ('rf_importance' or 'rfe')

        Returns:
            Tuple of (X_train_selected, X_test_selected, selected_feature_names)
        """
        print(f"Selecting features using {method}...")

        if method == 'rf_importance':
            # Use Random Forest feature importance
            rf = RandomForestClassifier(
                n_estimators=50,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                max_depth=10
            )
            rf.fit(X_train, y)

            # Select features above median importance
            selector = SelectFromModel(rf, threshold='median', prefit=True)
            self.feature_selector = selector

            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)

            self.selected_features = X_train.columns[selector.get_support()].tolist()

            X_train_selected = pd.DataFrame(X_train_selected, columns=self.selected_features)
            X_test_selected = pd.DataFrame(X_test_selected, columns=self.selected_features)

            print(f"Selected {len(self.selected_features)} features")

        else:
            raise ValueError(f"Unknown selection method: {method}")

        return X_train_selected, X_test_selected, self.selected_features

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_for_selection: Optional[np.ndarray] = None,
        apply_feature_selection: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Complete preprocessing pipeline.

        Args:
            train_df: Training DataFrame with targets
            test_df: Test DataFrame
            target_for_selection: Target variable for feature selection (e.g., CS)
            apply_feature_selection: Whether to apply feature selection

        Returns:
            Tuple of (X_train_final, X_test_final, targets_dict)
        """
        print("\n" + "="*80)
        print("DATA PREPROCESSING PIPELINE")
        print("="*80)

        # Step 1: Separate features and targets
        X_train, X_test, targets = self.prepare_features_targets(train_df, test_df)
        print(f"\nInitial features: {X_train.shape[1]}")

        # Step 2: Encode categorical variables
        X_train, X_test = self.encode_categorical(X_train, X_test)

        # Step 3: Scale features
        X_train, X_test = self.scale_features(X_train, X_test)

        # Step 4: Feature selection (optional)
        if apply_feature_selection and target_for_selection is not None:
            X_train, X_test, selected_features = self.select_features(
                X_train, X_test, target_for_selection
            )
            print(f"Final features after selection: {X_train.shape[1]}")
        else:
            print("Skipping feature selection")

        self.fitted = True
        print("\nPreprocessing complete!")
        print("="*80)

        return X_train, X_test, targets

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Encode
        cat_cols = [col for col in self.categorical_cols if col in X.columns]
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)

        # Scale
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Select features
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
            X_selected = pd.DataFrame(X_selected, columns=self.selected_features)
            return X_selected

        return X_scaled
