"""
Classification Models Module
============================
Implements claim status prediction models:
- XGBoost Classifier
- LightGBM Classifier
- Logistic Regression

Uses Optuna for hyperparameter optimization.

Author: Nandeesh H S
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from config import (
    RANDOM_SEED,
    N_FOLDS,
    OPTUNA_N_TRIALS_CLASSIFICATION,
    OPTUNA_TIMEOUT
)

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ClaimStatusClassifier:
    """
    Ensemble of classification models for predicting claim occurrence (CS).

    Trains and combines XGBoost, LightGBM, and Logistic Regression.
    """

    def __init__(self, n_folds: int = N_FOLDS, random_state: int = RANDOM_SEED):
        """
        Initialize classifier.

        Args:
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )

        # Models
        self.xgb_model = None
        self.lgb_model = None
        self.lr_model = None

        # Performance scores
        self.xgb_score = 0.0
        self.lgb_score = 0.0
        self.lr_score = 0.0

        # Ensemble weights
        self.weights = None

    def train_xgboost(
        self, X: pd.DataFrame, y: np.ndarray, n_trials: int = OPTUNA_N_TRIALS_CLASSIFICATION
    ) -> Tuple[object, float]:
        """
        Train XGBoost classifier with Optuna optimization.

        Args:
            X: Training features
            y: Binary target (claim status)
            n_trials: Number of Optuna trials

        Returns:
            Tuple of (trained_model, best_auc_score)
        """
        print("\nTraining XGBoost Classifier...")

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': -1,
                'eval_metric': 'auc'
            }

            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=self.cv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Train final model with best parameters
        self.xgb_model = xgb.XGBClassifier(**study.best_params)
        self.xgb_model.fit(X, y)
        self.xgb_score = study.best_value

        print(f"XGBoost - Best AUC: {self.xgb_score:.4f}")

        return self.xgb_model, self.xgb_score

    def train_lightgbm(
        self, X: pd.DataFrame, y: np.ndarray, n_trials: int = OPTUNA_N_TRIALS_CLASSIFICATION
    ) -> Tuple[object, float]:
        """
        Train LightGBM classifier with Optuna optimization.

        Args:
            X: Training features
            y: Binary target (claim status)
            n_trials: Number of Optuna trials

        Returns:
            Tuple of (trained_model, best_auc_score)
        """
        print("\nTraining LightGBM Classifier...")

        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }

            model = lgb.LGBMClassifier(**params)
            scores = cross_val_score(model, X, y, cv=self.cv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Train final model with best parameters
        self.lgb_model = lgb.LGBMClassifier(**study.best_params)
        self.lgb_model.fit(X, y)
        self.lgb_score = study.best_value

        print(f"LightGBM - Best AUC: {self.lgb_score:.4f}")

        return self.lgb_model, self.lgb_score

    def train_logistic_regression(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[object, float]:
        """
        Train Logistic Regression baseline model.

        Args:
            X: Training features
            y: Binary target (claim status)

        Returns:
            Tuple of (trained_model, auc_score)
        """
        print("\nTraining Logistic Regression...")

        self.lr_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            n_jobs=-1
        )
        self.lr_model.fit(X, y)

        scores = cross_val_score(self.lr_model, X, y, cv=self.cv, scoring='roc_auc')
        self.lr_score = scores.mean()

        print(f"Logistic Regression - AUC: {self.lr_score:.4f}")

        return self.lr_model, self.lr_score

    def train_all(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Train all classification models.

        Args:
            X: Training features
            y: Binary target (claim status)

        Returns:
            Dictionary of model scores
        """
        print("\n" + "="*80)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*80)

        # Train all models
        self.train_xgboost(X, y)
        self.train_lightgbm(X, y)
        self.train_logistic_regression(X, y)

        # Calculate ensemble weights (weighted by performance)
        scores = np.array([self.xgb_score, self.lgb_score, self.lr_score])
        self.weights = scores / scores.sum()

        print("\n" + "="*80)
        print("CLASSIFICATION TRAINING COMPLETE")
        print("="*80)
        print(f"\nEnsemble Weights:")
        print(f"  XGBoost:    {self.weights[0]:.4f}")
        print(f"  LightGBM:   {self.weights[1]:.4f}")
        print(f"  Logistic:   {self.weights[2]:.4f}")

        return {
            'xgb_auc': self.xgb_score,
            'lgb_auc': self.lgb_score,
            'lr_auc': self.lr_score
        }

    def predict_proba(self, X: pd.DataFrame, use_ensemble: bool = True) -> np.ndarray:
        """
        Predict claim probabilities.

        Args:
            X: Features
            use_ensemble: If True, use weighted ensemble; if False, use XGBoost only

        Returns:
            Array of claim probabilities
        """
        if use_ensemble:
            # Weighted ensemble prediction
            prob_xgb = self.xgb_model.predict_proba(X)[:, 1]
            prob_lgb = self.lgb_model.predict_proba(X)[:, 1]
            prob_lr = self.lr_model.predict_proba(X)[:, 1]

            prob_ensemble = (
                self.weights[0] * prob_xgb +
                self.weights[1] * prob_lgb +
                self.weights[2] * prob_lr
            )

            return prob_ensemble
        else:
            # Use XGBoost only
            return self.xgb_model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble on validation data.

        Args:
            X: Validation features
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        prob_ensemble = self.predict_proba(X, use_ensemble=True)
        auc_ensemble = roc_auc_score(y, prob_ensemble)

        return {
            'ensemble_auc': auc_ensemble
        }
