"""
Ensemble Strategies Module
==========================
Implements two modeling approaches for insurance claims:

Approach A (Two-Stage):
  1. Classify claim occurrence (CS)
  2. Predict loss cost conditional on claim (LC | CS=1)
  3. Final prediction = P(CS=1) × E[LC | CS=1]

Approach B (Direct Tweedie):
  - Directly predict loss cost using Tweedie regression
  - Handles zeros naturally via Tweedie distribution

Author: Nandeesh H S
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

from models.classifier import ClaimStatusClassifier
from models.regressor import TweedieRegressionModel
from config import RANDOM_SEED


class ApproachA:
    """
    Two-stage approach: Classification + Conditional Regression.

    Steps:
    1. Train classifier on all data to predict P(claim)
    2. Train regressor only on claims data to predict E[LC | claim]
    3. Final prediction = P(claim) × E[LC | claim]
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        """Initialize Approach A models."""
        self.random_state = random_state
        self.classifier = ClaimStatusClassifier(random_state=random_state)
        self.regressor = TweedieRegressionModel(random_state=random_state)

    def train(
        self,
        X: pd.DataFrame,
        y_cs: np.ndarray,
        y_lc: np.ndarray,
        y_halc: np.ndarray
    ) -> Dict[str, float]:
        """
        Train two-stage model.

        Args:
            X: Training features
            y_cs: Claim status (binary)
            y_lc: Loss cost
            y_halc: HALC

        Returns:
            Dictionary of training metrics
        """
        print("\n" + "="*80)
        print("APPROACH A: TWO-STAGE MODELING")
        print("="*80)

        # Step 1: Train classifier on ALL data
        print("\nStep 1: Training claim classifier on all data...")
        clf_scores = self.classifier.train_all(X, y_cs)

        # Step 2: Train regressor ONLY on claims (CS=1)
        print("\nStep 2: Training regressor on claims data only...")
        X_claims = X[y_cs == 1]
        y_lc_claims = y_lc[y_cs == 1]
        y_halc_claims = y_halc[y_cs == 1]

        print(f"Training on {len(X_claims):,} claims records (out of {len(X):,} total)")

        reg_scores = self.regressor.train_all(X_claims, y_lc_claims, y_halc_claims)

        print("\n" + "="*80)
        print("APPROACH A TRAINING COMPLETE")
        print("="*80)

        return {**clf_scores, **reg_scores}

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions using two-stage approach.

        Args:
            X: Features

        Returns:
            Tuple of (CS_pred, LC_pred, HALC_pred)
        """
        # Step 1: Predict claim probability
        CS_pred = self.classifier.predict_proba(X, use_ensemble=True)

        # Step 2: Predict loss cost given claim
        LC_given_claim = self.regressor.predict_lc(X, use_ensemble=True)
        HALC_given_claim = self.regressor.predict_halc(X, use_ensemble=True)

        # Step 3: Combine (P(claim) × E[LC | claim])
        LC_pred = CS_pred * LC_given_claim
        HALC_pred = CS_pred * HALC_given_claim

        return CS_pred, LC_pred, HALC_pred


class ApproachB:
    """
    Direct Tweedie regression approach.

    Directly predicts loss cost using Tweedie distribution,
    which naturally handles zero-inflated data.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        """Initialize Approach B models."""
        self.random_state = random_state
        self.classifier = ClaimStatusClassifier(random_state=random_state)
        self.regressor = TweedieRegressionModel(random_state=random_state)

    def train(
        self,
        X: pd.DataFrame,
        y_cs: np.ndarray,
        y_lc: np.ndarray,
        y_halc: np.ndarray
    ) -> Dict[str, float]:
        """
        Train direct Tweedie model.

        Args:
            X: Training features
            y_cs: Claim status (for classifier)
            y_lc: Loss cost
            y_halc: HALC

        Returns:
            Dictionary of training metrics
        """
        print("\n" + "="*80)
        print("APPROACH B: DIRECT TWEEDIE REGRESSION")
        print("="*80)

        # Train classifier (for CS prediction)
        print("\nTraining claim classifier...")
        clf_scores = self.classifier.train_all(X, y_cs)

        # Train regressor on ALL data (including zeros)
        print("\nTraining Tweedie regressor on all data (including zeros)...")
        reg_scores = self.regressor.train_all(X, y_lc, y_halc)

        print("\n" + "="*80)
        print("APPROACH B TRAINING COMPLETE")
        print("="*80)

        return {**clf_scores, **reg_scores}

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions using direct approach.

        Args:
            X: Features

        Returns:
            Tuple of (CS_pred, LC_pred, HALC_pred)
        """
        # Predict claim probability (for CS)
        CS_pred = self.classifier.predict_proba(X, use_ensemble=True)

        # Directly predict loss costs (Tweedie handles zeros)
        LC_pred = self.regressor.predict_lc(X, use_ensemble=True)
        HALC_pred = self.regressor.predict_halc(X, use_ensemble=True)

        return CS_pred, LC_pred, HALC_pred


class ApproachComparison:
    """
    Compares Approach A vs Approach B and selects the best.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        """Initialize comparison framework."""
        self.random_state = random_state
        self.approach_a = ApproachA(random_state=random_state)
        self.approach_b = ApproachB(random_state=random_state)
        self.best_approach = None

    def train_both_approaches(
        self,
        X: pd.DataFrame,
        y_cs: np.ndarray,
        y_lc: np.ndarray,
        y_halc: np.ndarray
    ) -> Dict[str, any]:
        """
        Train both approaches.

        Args:
            X: Training features
            y_cs: Claim status
            y_lc: Loss cost
            y_halc: HALC

        Returns:
            Dictionary with training results
        """
        # Train Approach A
        scores_a = self.approach_a.train(X, y_cs, y_lc, y_halc)

        # Train Approach B
        scores_b = self.approach_b.train(X, y_cs, y_lc, y_halc)

        return {
            'approach_a': scores_a,
            'approach_b': scores_b
        }

    def compare_on_validation(
        self,
        X: pd.DataFrame,
        y_cs: np.ndarray,
        y_lc: np.ndarray,
        y_halc: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, any]:
        """
        Compare both approaches on validation set.

        Args:
            X: Full training features
            y_cs: Claim status
            y_lc: Loss cost
            y_halc: HALC
            test_size: Validation set proportion

        Returns:
            Dictionary with comparison results and best approach
        """
        print("\n" + "="*80)
        print("COMPARING APPROACHES ON VALIDATION DATA")
        print("="*80)

        # Create validation split
        X_train, X_val, y_cs_train, y_cs_val, y_lc_train, y_lc_val, y_halc_train, y_halc_val = train_test_split(
            X, y_cs, y_lc, y_halc,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_cs
        )

        print(f"\nValidation split: {len(X_val):,} records ({test_size*100:.0f}%)")

        # Evaluate Approach A
        CS_pred_a, LC_pred_a, HALC_pred_a = self.approach_a.predict(X_val)
        mse_lc_a = mean_squared_error(y_lc_val, LC_pred_a)
        mse_halc_a = mean_squared_error(y_halc_val, HALC_pred_a)
        auc_cs_a = roc_auc_score(y_cs_val, CS_pred_a)

        # Evaluate Approach B
        CS_pred_b, LC_pred_b, HALC_pred_b = self.approach_b.predict(X_val)
        mse_lc_b = mean_squared_error(y_lc_val, LC_pred_b)
        mse_halc_b = mean_squared_error(y_halc_val, HALC_pred_b)
        auc_cs_b = roc_auc_score(y_cs_val, CS_pred_b)

        # Determine winner
        total_mse_a = mse_lc_a + mse_halc_a
        total_mse_b = mse_lc_b + mse_halc_b

        if total_mse_a < total_mse_b:
            self.best_approach = "A"
            winner_text = "APPROACH A (Two-Stage)"
        else:
            self.best_approach = "B"
            winner_text = "APPROACH B (Direct Tweedie)"

        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        print(f"\nApproach A (Two-Stage):")
        print(f"  CS AUC:      {auc_cs_a:.4f}")
        print(f"  LC MSE:      {mse_lc_a:,.2f}")
        print(f"  HALC MSE:    {mse_halc_a:,.2f}")
        print(f"  Total MSE:   {total_mse_a:,.2f}")

        print(f"\nApproach B (Direct Tweedie):")
        print(f"  CS AUC:      {auc_cs_b:.4f}")
        print(f"  LC MSE:      {mse_lc_b:,.2f}")
        print(f"  HALC MSE:    {mse_halc_b:,.2f}")
        print(f"  Total MSE:   {total_mse_b:,.2f}")

        print("\n" + "="*80)
        print(f"WINNER: {winner_text}")
        print("="*80)

        if total_mse_b < total_mse_a:
            improvement_factor = total_mse_a / total_mse_b
            print(f"\nApproach B is {improvement_factor:.1f}x better than Approach A!")

        return {
            'approach_a': {
                'cs_auc': auc_cs_a,
                'lc_mse': mse_lc_a,
                'halc_mse': mse_halc_a,
                'total_mse': total_mse_a
            },
            'approach_b': {
                'cs_auc': auc_cs_b,
                'lc_mse': mse_lc_b,
                'halc_mse': mse_halc_b,
                'total_mse': total_mse_b
            },
            'best_approach': self.best_approach
        }

    def predict_with_best(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions using the best approach.

        Args:
            X: Features

        Returns:
            Tuple of (CS_pred, LC_pred, HALC_pred)
        """
        if self.best_approach is None:
            raise ValueError("Must run compare_on_validation first to determine best approach")

        if self.best_approach == "A":
            return self.approach_a.predict(X)
        else:
            return self.approach_b.predict(X)
