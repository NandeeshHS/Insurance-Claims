"""
Regression Models Module
========================
Implements Tweedie regression for loss cost prediction:
- XGBoost with Tweedie objective
- LightGBM with Tweedie objective
- Scikit-learn TweedieRegressor (GLM)

Supports both Approach A (two-stage) and Approach B (direct).

Author: Nandeesh H S
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import optuna
from config import (
    RANDOM_SEED,
    OPTUNA_N_TRIALS_REGRESSION,
    TWEEDIE_VARIANCE_POWER_MIN,
    TWEEDIE_VARIANCE_POWER_MAX
)

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TweedieRegressionModel:
    """
    Ensemble of Tweedie regression models for loss cost prediction.

    Handles zero-inflated insurance claims data using Tweedie distribution.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Initialize regressor.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

        # Models for LC (Loss Cost)
        self.xgb_lc_model = None
        self.lgb_lc_model = None
        self.glm_lc_model = None

        # Models for HALC (Historically Adjusted Loss Cost)
        self.xgb_halc_model = None
        self.lgb_halc_model = None
        self.glm_halc_model = None

        # Performance scores
        self.xgb_lc_score = 0.0
        self.lgb_lc_score = 0.0
        self.glm_lc_score = 0.0

        # Ensemble weights
        self.lc_weights = None
        self.halc_weights = None

        # Best hyperparameters
        self.best_xgb_params = None
        self.best_lgb_params = None

    def train_xgboost_lc(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_trials: int = OPTUNA_N_TRIALS_REGRESSION
    ) -> Tuple[object, float]:
        """
        Train XGBoost Tweedie regressor for LC.

        Args:
            X: Training features
            y: Loss cost target
            n_trials: Number of Optuna trials

        Returns:
            Tuple of (trained_model, best_mse)
        """
        print("\nTraining XGBoost Tweedie for LC...")

        def objective(trial):
            params = {
                'objective': 'reg:tweedie',
                'tweedie_variance_power': trial.suggest_float(
                    'tweedie_variance_power',
                    TWEEDIE_VARIANCE_POWER_MIN,
                    TWEEDIE_VARIANCE_POWER_MAX
                ),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': -1
            }

            model = xgb.XGBRegressor(**params)
            model.fit(X, y)
            pred = model.predict(X)
            mse = mean_squared_error(y, pred)
            return mse

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Train final model with best parameters
        self.best_xgb_params = study.best_params
        self.xgb_lc_model = xgb.XGBRegressor(**self.best_xgb_params)
        self.xgb_lc_model.fit(X, y)
        self.xgb_lc_score = study.best_value

        print(f"XGBoost LC - Best MSE: {self.xgb_lc_score:.2f}")

        return self.xgb_lc_model, self.xgb_lc_score

    def train_lightgbm_lc(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_trials: int = OPTUNA_N_TRIALS_REGRESSION
    ) -> Tuple[object, float]:
        """
        Train LightGBM Tweedie regressor for LC.

        Args:
            X: Training features
            y: Loss cost target
            n_trials: Number of Optuna trials

        Returns:
            Tuple of (trained_model, best_mse)
        """
        print("\nTraining LightGBM Tweedie for LC...")

        def objective(trial):
            params = {
                'objective': 'tweedie',
                'tweedie_variance_power': trial.suggest_float(
                    'tweedie_variance_power',
                    TWEEDIE_VARIANCE_POWER_MIN,
                    TWEEDIE_VARIANCE_POWER_MAX
                ),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            pred = model.predict(X)
            mse = mean_squared_error(y, pred)
            return mse

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Train final model with best parameters
        self.best_lgb_params = study.best_params
        self.lgb_lc_model = lgb.LGBMRegressor(**self.best_lgb_params)
        self.lgb_lc_model.fit(X, y)
        self.lgb_lc_score = study.best_value

        print(f"LightGBM LC - Best MSE: {self.lgb_lc_score:.2f}")

        return self.lgb_lc_model, self.lgb_lc_score

    def train_glm_lc(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[object, float]:
        """
        Train GLM Tweedie regressor for LC.

        Args:
            X: Training features
            y: Loss cost target

        Returns:
            Tuple of (trained_model, mse)
        """
        print("\nTraining GLM Tweedie for LC...")

        self.glm_lc_model = TweedieRegressor(power=1.5, alpha=0.1, max_iter=300)
        self.glm_lc_model.fit(X, y)

        pred = self.glm_lc_model.predict(X)
        self.glm_lc_score = mean_squared_error(y, pred)

        print(f"GLM LC - MSE: {self.glm_lc_score:.2f}")

        return self.glm_lc_model, self.glm_lc_score

    def train_halc_models(self, X: pd.DataFrame, y_halc: np.ndarray) -> None:
        """
        Train HALC models using LC hyperparameters.

        Args:
            X: Training features
            y_halc: HALC target
        """
        print("\nTraining HALC models (using LC hyperparameters)...")

        # XGBoost HALC
        self.xgb_halc_model = xgb.XGBRegressor(**self.best_xgb_params)
        self.xgb_halc_model.fit(X, y_halc)

        # LightGBM HALC
        self.lgb_halc_model = lgb.LGBMRegressor(**self.best_lgb_params)
        self.lgb_halc_model.fit(X, y_halc)

        # GLM HALC
        self.glm_halc_model = TweedieRegressor(power=1.5, alpha=0.1, max_iter=300)
        self.glm_halc_model.fit(X, y_halc)

        print("HALC models trained successfully!")

    def train_all(
        self,
        X: pd.DataFrame,
        y_lc: np.ndarray,
        y_halc: np.ndarray
    ) -> Dict[str, float]:
        """
        Train all regression models for both LC and HALC.

        Args:
            X: Training features
            y_lc: Loss cost target
            y_halc: HALC target

        Returns:
            Dictionary of model scores
        """
        print("\n" + "="*80)
        print("TRAINING TWEEDIE REGRESSION MODELS")
        print("="*80)

        # Train LC models
        self.train_xgboost_lc(X, y_lc)
        self.train_lightgbm_lc(X, y_lc)
        self.train_glm_lc(X, y_lc)

        # Calculate LC ensemble weights (inverse MSE weighting)
        scores = np.array([self.xgb_lc_score, self.lgb_lc_score, self.glm_lc_score])
        inv_scores = 1.0 / scores
        self.lc_weights = inv_scores / inv_scores.sum()

        # Train HALC models
        self.train_halc_models(X, y_halc)

        # HALC uses same weights as LC
        self.halc_weights = self.lc_weights.copy()

        print("\n" + "="*80)
        print("REGRESSION TRAINING COMPLETE")
        print("="*80)
        print(f"\nLC Ensemble Weights:")
        print(f"  XGBoost:  {self.lc_weights[0]:.4f}")
        print(f"  LightGBM: {self.lc_weights[1]:.4f}")
        print(f"  GLM:      {self.lc_weights[2]:.4f}")

        return {
            'xgb_lc_mse': self.xgb_lc_score,
            'lgb_lc_mse': self.lgb_lc_score,
            'glm_lc_mse': self.glm_lc_score
        }

    def predict_lc(self, X: pd.DataFrame, use_ensemble: bool = True) -> np.ndarray:
        """
        Predict loss cost.

        Args:
            X: Features
            use_ensemble: If True, use weighted ensemble; if False, use XGBoost only

        Returns:
            Array of LC predictions
        """
        if use_ensemble:
            pred_xgb = self.xgb_lc_model.predict(X)
            pred_lgb = self.lgb_lc_model.predict(X)
            pred_glm = self.glm_lc_model.predict(X)

            pred_ensemble = (
                self.lc_weights[0] * pred_xgb +
                self.lc_weights[1] * pred_lgb +
                self.lc_weights[2] * pred_glm
            )

            return pred_ensemble
        else:
            return self.xgb_lc_model.predict(X)

    def predict_halc(self, X: pd.DataFrame, use_ensemble: bool = True) -> np.ndarray:
        """
        Predict HALC.

        Args:
            X: Features
            use_ensemble: If True, use weighted ensemble; if False, use XGBoost only

        Returns:
            Array of HALC predictions
        """
        if use_ensemble:
            pred_xgb = self.xgb_halc_model.predict(X)
            pred_lgb = self.lgb_halc_model.predict(X)
            pred_glm = self.glm_halc_model.predict(X)

            pred_ensemble = (
                self.halc_weights[0] * pred_xgb +
                self.halc_weights[1] * pred_lgb +
                self.halc_weights[2] * pred_glm
            )

            return pred_ensemble
        else:
            return self.xgb_halc_model.predict(X)

    def evaluate(self, X: pd.DataFrame, y_lc: np.ndarray, y_halc: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble on validation data.

        Args:
            X: Validation features
            y_lc: True LC values
            y_halc: True HALC values

        Returns:
            Dictionary of evaluation metrics
        """
        pred_lc = self.predict_lc(X, use_ensemble=True)
        pred_halc = self.predict_halc(X, use_ensemble=True)

        mse_lc = mean_squared_error(y_lc, pred_lc)
        mse_halc = mean_squared_error(y_halc, pred_halc)

        return {
            'lc_mse': mse_lc,
            'halc_mse': mse_halc
        }
